"""
混合检索：BM25（关键词） + 向量（语义） + RRF 融合

分词方案：
  优先使用 **jieba 搜索引擎模式**（lcut_for_search），对未登录词和长词做细粒度切分，召回更高；
  jieba 不可用时自动降级为**字符二元组**（bigram），保证模块在无 jieba 环境下仍能工作。

语料规模小时（本项目 66 chunk / 约 3.5 万字符）内存 BM25 完全够用；
若语料增长到百万级，应迁移到 Milvus 原生稀疏向量 + BM25 Function + RRFRanker。

用法：
    from utils.hybrid_retriever import BM25Index, reciprocal_rank_fusion, tokenize

    idx = BM25Index([(doc_id, text), ...])
    sparse_hits = idx.search("差旅住宿标准", top_k=10)      # [(doc_id, bm25_score)]
    fused = reciprocal_rank_fusion([dense_hits, sparse_hits], k=60)
"""
import logging
import math
import re
import warnings
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# 分词
# --------------------------------------------------------------------------

try:
    with warnings.catch_warnings():
        # jieba 内部 import pkg_resources，会在 setuptools>=81 上抛 DeprecationWarning
        warnings.simplefilter("ignore")
        import jieba

    # 关掉 "Building prefix dict..." 之类的 INFO 输出
    jieba.setLogLevel(logging.WARNING)
    JIEBA_AVAILABLE = True
except ImportError:  # pragma: no cover - 取决于运行环境
    jieba = None
    JIEBA_AVAILABLE = False
    logger.warning(
        "jieba 未安装，BM25 分词降级为字符二元组。"
        "建议安装：pip install jieba -i https://pypi.tuna.tsinghua.edu.cn/simple"
    )

# 连续中文 / 连续英数，各作为一段分别处理
_SEG_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+")

# 纯标点/符号 token（中文标点不属于 \w，会被这里滤掉）
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)

# 停用词：中文虚词 + 英文功能词。
# 为什么必须过滤：BM25 的 IDF 用了 ln(1 + ...) 平滑，恒为正，
# 因此"在所有文档里都出现"的虚词（的/是/在/如何）仍会贡献分数。
# 语料小的时候这种噪声会压过真正有区分度的实词，导致无关查询也命中。
STOPWORDS: Set[str] = {
    # ---- 中文虚词 / 高频功能词 ----
    "的", "了", "是", "在", "和", "就", "都", "而", "及", "与", "着", "或", "把", "被", "让",
    "不", "也", "很", "会", "要", "有", "没", "这", "那", "你", "我", "他", "她", "它", "们",
    "个", "之", "其", "以", "为", "上", "下", "中", "对", "从", "到", "向", "给", "等", "并",
    "可", "能", "该", "当", "还", "再", "又", "才", "只", "更", "太", "却", "均", "皆", "需",
    "一个", "一些", "一下", "没有", "我们", "你们", "他们", "咱们", "大家", "自己",
    "这个", "那个", "这些", "那些", "这样", "那样", "这里", "那里",
    "可以", "应该", "可能", "需要", "进行", "通过", "对于", "关于", "根据", "按照",
    "因为", "所以", "但是", "如果", "以及", "还是", "或者", "并且", "然后", "而且",
    "什么", "怎么", "如何", "哪些", "哪个", "为什么", "多少", "是否", "怎样",
    "已经", "一直", "一起", "一定", "一样", "以后", "以前", "现在", "时候",
    "问题", "情况", "方面", "相关", "有关", "等等", "比如", "例如", "其他", "其中",
    # ---- 英文功能词 ----
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "for", "and", "or", "but", "if", "then",
    "this", "that", "these", "those", "it", "its", "as", "by", "with", "from",
    "how", "what", "which", "who", "when", "where", "why", "can", "will", "not",
}


def tokenize(text: str, drop_stopwords: bool = True) -> List[str]:
    """
    中英混合分词。

    - 中文：jieba `lcut_for_search`（搜索引擎模式，长词会额外切出子词，提升召回）
    - 英文/数字：整体小写作为一个 token
    - 丢弃空白与纯标点 token
    - `drop_stopwords=True` 时再丢弃 STOPWORDS 里的虚词（**强烈建议开启**）

    jieba 不可用时中文降级为字符二元组：'差旅标准' -> ['差旅','旅标','标准']
    """
    if not text:
        return []

    tokens: List[str] = []
    for m in _SEG_RE.finditer(text):
        seg = m.group(0)

        if "\u4e00" <= seg[0] <= "\u9fff":
            # ---- 中文段 ----
            if JIEBA_AVAILABLE:
                tokens.extend(jieba.lcut_for_search(seg))
            else:
                # 降级：字符二元组（单字则保留单字）
                if len(seg) == 1:
                    tokens.append(seg)
                else:
                    tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
        else:
            # ---- 英文/数字段 ----
            tokens.append(seg.lower())

    # 过滤空白 / 纯标点
    tokens = [t for t in tokens if t and not _PUNCT_ONLY_RE.match(t)]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def preload() -> None:
    """
    预热 jieba 词典（首次调用会有约 1 秒的构建开销）。
    可在 agent 启动时调用，避免第一次检索卡顿。
    """
    if JIEBA_AVAILABLE:
        jieba.initialize()
        logger.info("jieba 词典预热完成")


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------

class BM25Index:
    """
    Okapi BM25。

        score(q, d) = Σ_t IDF(t) · (f(t,d) · (k1+1)) / (f(t,d) + k1 · (1 - b + b · |d|/avgdl))

        IDF(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

    参数含义：
        k1（默认 1.5）  词频饱和系数 —— 一个词出现 10 次并不比出现 3 次重要 3 倍，收益递减
        b （默认 0.75） 文档长度归一化 —— 长文档天然更容易命中词，需要惩罚
    """

    def __init__(
        self,
        docs: Sequence[Tuple[object, str]],
        k1: float = 1.5,
        b: float = 0.75,
        drop_stopwords: bool = True,
    ):
        """docs: [(doc_id, text), ...]"""
        self.k1 = float(k1)
        self.b = float(b)
        self.drop_stopwords = bool(drop_stopwords)

        self.doc_ids: List[object] = []
        self._tf: List[Counter] = []
        self._len: List[int] = []
        df: Counter = Counter()

        for doc_id, text in docs:
            toks = tokenize(text, drop_stopwords=self.drop_stopwords)
            tf = Counter(toks)
            self.doc_ids.append(doc_id)
            self._tf.append(tf)
            self._len.append(len(toks))
            for term in tf:
                df[term] += 1

        self.n = len(self.doc_ids)
        self.avgdl = (sum(self._len) / self.n) if self.n else 0.0
        # 加 1 是为了让 IDF 恒为正，避免高频词拿到负分
        self.idf: Dict[str, float] = {
            term: math.log(1 + (self.n - d + 0.5) / (d + 0.5))
            for term, d in df.items()
        }

        logger.debug(
            "BM25Index 构建完成：%d 篇，平均长度 %.1f token，词表 %d（jieba=%s）",
            self.n, self.avgdl, len(self.idf), JIEBA_AVAILABLE,
        )

    def search(self, query: str, top_k: int = 10) -> List[Tuple[object, float]]:
        """
        检索，返回 [(doc_id, bm25_score)]，按分数降序，只含 score > 0 的条目。
        """
        if not self.n:
            return []

        q_terms = tokenize(query, drop_stopwords=self.drop_stopwords)
        if not q_terms:
            return []

        scores = [0.0] * self.n
        # 去重：同一个词在 query 里出现多次不重复累加（BM25 的 qtf 通常取 1）
        for term in set(q_terms):
            idf = self.idf.get(term)
            if idf is None:          # 该词不在语料词表中
                continue
            for i, tf in enumerate(self._tf):
                f = tf.get(term)
                if not f:
                    continue
                dl = self._len[i] or 1
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[i] += idf * (f * (self.k1 + 1)) / denom

        ranked = [(self.doc_ids[i], s) for i, s in enumerate(scores) if s > 0]
        ranked.sort(key=lambda x: -x[1])
        return ranked[:top_k]

    def stats(self) -> Dict:
        """便于诊断/测试。"""
        return {
            "docs": self.n,
            "avgdl": round(self.avgdl, 2),
            "vocab": len(self.idf),
            "jieba": JIEBA_AVAILABLE,
            "drop_stopwords": self.drop_stopwords,
        }


# --------------------------------------------------------------------------
# RRF 融合
# --------------------------------------------------------------------------

def reciprocal_rank_fusion(
    rank_lists: Sequence[Sequence[Tuple[object, float]]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[object, float]]:
    """
    Reciprocal Rank Fusion。

        RRF(d) = Σ_i  w_i / (k + rank_i(d))        rank 从 1 开始

    为什么用 RRF 而不是加权求和：
        BM25 分数（无上界，取决于 IDF 量纲）和余弦相似度（-1~1）**量纲不同**，
        直接加权需要归一化，而归一化方式本身又要调参。
        RRF 只看**排名**不看分数，天然规避了这个问题，且对异常分数鲁棒。

    k=60 是原论文（Cormack et al., 2009）的经验值，也是 Milvus RRFRanker 的默认值。
    k 越大，排名差异被压得越平（高分通道的优势越不明显）。

    Args:
        rank_lists: 各召回通道的结果，每个都已按相关性**降序**排列
        k:          平滑常数
        weights:    各通道权重，默认全 1.0

    Returns:
        [(doc_id, rrf_score)]，按 rrf_score 降序
    """
    if weights is None:
        weights = [1.0] * len(rank_lists)

    fused: Dict[object, float] = {}
    for weight, ranked in zip(weights, rank_lists):
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)

    return sorted(fused.items(), key=lambda x: -x[1])
