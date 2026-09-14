"""
Rerank 精排：Cross-Encoder 对召回候选二次打分重排

两段式检索：召回（宽而快：向量 10 + BM25 10 → RRF 保留 10 条）
            → 精排（窄而准：Cross-Encoder 打分 → 取 3 条）

⚠️ 实测结论（53 条标注 query，`scripts/eval_retrieval.py`）：
   「精排重排」在本语料上是**负收益**——Hit@1 持平、Hit@3 53→52、MRR 0.959→0.950。
   根因是分数饱和：12 篇文档高度重叠（如 10_international_travel.txt 也有整节
   「报销凭证 / 报销标准」），任何差旅问题都有一堆片段「确实相关」，精排一视同仁
   给 0.86~0.99，排序接近抛硬币；而 RRF 里的 BM25 路奖励**关键词精确命中**，
   恰好更贴近「哪篇是这道题的标准答案」。
   但精排分数做**负例过滤**极强（负例 Top1 ≤0.25 vs 正例 ≥0.83），
   正好补上 min_bm25_score「区间重叠、无法分开」的窟窿。

   所以默认走 `mode="filter"`：**RRF 排序 + 精排当闸门**。

降级策略（与 jieba 缺失降级为 bigram 一致）：
   模型目录不存在 / 依赖缺失 / 打分异常 → 原样返回候选、保持 RRF 顺序、
   功能不中断；**且此时绝不做分数阈值过滤**（没有分数可过滤）。

用法：
   from utils.reranker import CrossEncoderReranker, apply_rerank

   rr = CrossEncoderReranker("data/models/bge-reranker-base")
   # 闸门：保留 RRF 顺序，用精排分数丢掉不相关的（推荐）
   top3 = apply_rerank(rr, "差旅住宿标准", candidates, top_k=3,
                       score_threshold=0.3, mode="filter")
   # 重排：完全按精排分数排序（A/B 对比用）
   top3 = apply_rerank(rr, "差旅住宿标准", candidates, top_k=3, mode="rerank")
"""
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:  # pragma: no cover - 取决于运行环境
    CrossEncoder = None
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers 不可用，Rerank 精排将被跳过")

DEFAULT_SCORE_KEY = "rerank_score"

# apply_rerank 内部用于恢复候选原顺序的临时标记，输出前会被清掉
_ORDER_KEY = "_rerank_order"


def resolve_model_path(model_name_or_path: Optional[str]) -> Optional[str]:
    """
    本地目录优先：存在则返回绝对路径，否则返回 None。
    与 embedding 模型保持一致——项目坚持本地部署，不默认联网拉取。
    """
    if not model_name_or_path:
        return None
    p = Path(model_name_or_path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return str(p.resolve()) if p.exists() else None


class CrossEncoderReranker:
    """
    Cross-Encoder 精排器。模型**懒加载**：首次 rerank() 时才载入权重，
    与项目的「懒加载 / 快速启动」原则一致（未启用精排的会话完全不付这个成本）。

    Args:
        model_name_or_path: 本地目录（推荐）或 HF 模型 ID
        device:             None=自动；本机无 CUDA 时走 CPU
        batch_size:         打分批大小
        max_length:         (query, doc) 拼接后的最大 token 数，超出由 tokenizer 截断
        allow_download:     False（默认）时本地目录不存在即降级；
                            True 则允许联网按 HF ID 拉取
        scorer:             可注入的打分函数 (query, texts) -> Sequence[float]。
                            注入后不再加载本地模型。用途有两个：
                              1. 离线单测（不需要模型文件）
                              2. 日后换成 Rerank API 后端（Cohere / Jina / 云端）
                                 而不必改动 agent.py
        score_key:          写入候选 dict 的字段名
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512,
        allow_download: bool = False,
        scorer: Optional[Callable[[str, List[str]], Sequence[float]]] = None,
        score_key: str = DEFAULT_SCORE_KEY,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.allow_download = bool(allow_download)
        self.score_key = score_key

        self._injected = scorer is not None
        self._scorer = scorer
        self._model = None
        self._load_failed = False          # 失败后不再重试，避免每次检索都刷日志
        self._resolved_path: Optional[str] = None

    # ---------------- 模型装载 ----------------

    @property
    def available(self) -> bool:
        """精排是否真的可用（会触发一次装载）。供评测/诊断脚本判断。"""
        return self._load()

    def _load(self) -> bool:
        if self._injected:
            return True
        if self._model is not None:
            return True
        if self._load_failed:
            return False

        if not CROSS_ENCODER_AVAILABLE:
            logger.warning("Rerank 不可用：sentence-transformers 未安装")
            self._load_failed = True
            return False

        self._resolved_path = resolve_model_path(self.model_name_or_path)
        if self._resolved_path is None and not self.allow_download:
            logger.warning(
                "Rerank 不可用：本地模型目录不存在（%s），本次退回 RRF 顺序。"
                "请先运行 scripts/download_reranker.py；"
                "或把 RAG_CONFIG['rerank']['allow_download'] 设为 True 允许联网拉取。",
                self.model_name_or_path,
            )
            self._load_failed = True
            return False

        target = self._resolved_path or self.model_name_or_path
        try:
            logger.info("载入 Rerank 模型（首次调用，需数秒）: %s", target)
            self._model = CrossEncoder(
                target, device=self.device, max_length=self.max_length
            )
            logger.info("Rerank 模型载入完成")
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("Rerank 模型载入失败，本次退回 RRF 顺序: %s", e)
            self._load_failed = True
            return False

    def unload(self) -> None:
        """释放模型权重（CPU 内存占用不小，进程退出前调用）。"""
        self._model = None

    # ---------------- 打分 ----------------

    def rerank(
        self,
        query: str,
        candidates: Sequence[Dict],
        top_k: Optional[int] = None,
        content_key: str = "content",
    ) -> List[Dict]:
        """
        对候选打分并按分数降序重排。**不修改候选内容**，只追加一个 score_key 字段
        （原 rrf_score / matched_by 都保留，便于排查「到底哪一步把顺序改了」）。

        降级时原样返回候选（顺序不变、**无 score_key**），调用方据此判断是否生效。
        """
        if not candidates:
            return []

        if not self._load():
            return list(candidates[:top_k] if top_k else candidates)

        texts = [str(c.get(content_key) or "") for c in candidates]
        try:
            if self._injected:
                raw = list(self._scorer(query, texts))
            else:
                raw = self._model.predict(
                    [(query, t) for t in texts],
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("Rerank 打分失败，退回 RRF 顺序: %s", e)
            return list(candidates[:top_k] if top_k else candidates)

        scores = [float(s[0]) if isinstance(s, (list, tuple)) else float(s) for s in raw]

        if len(scores) != len(candidates):
            logger.warning(
                "Rerank 打分条数不匹配（%d != %d），退回 RRF 顺序",
                len(scores), len(candidates),
            )
            return list(candidates[:top_k] if top_k else candidates)

        scored: List[Dict] = []
        for cand, s in zip(candidates, scores):
            item = dict(cand)
            item[self.score_key] = s
            scored.append(item)

        # 稳定排序：同分保持 RRF 原顺序（Python sort 稳定，天然满足）
        scored.sort(key=lambda x: -x[self.score_key])
        return scored[:top_k] if top_k else scored

    def stats(self) -> Dict:
        return {
            "model": self.model_name_or_path,
            "resolved_path": self._resolved_path,
            "loaded": self._model is not None or self._injected,
            "injected_scorer": self._injected,
            "load_failed": self._load_failed,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
        }


def apply_rerank(
    reranker: Optional[CrossEncoderReranker],
    query: str,
    candidates: Sequence[Dict],
    top_k: int,
    score_threshold: Optional[float] = None,
    mode: str = "filter",
) -> List[Dict]:
    """
    精排 + 阈值过滤 + 截断。**降级安全**的纯逻辑，抽成模块级函数是为了能离线单测。

    mode 三种取值：
        "filter"（默认）用精排分数做**准入判断**，但**保持原顺序**（即 RRF 顺序）。
                 实测结论：在 12 篇高度重叠的差旅语料上，Cross-Encoder 的排序弱于 RRF
                 （Hit@3 53→52、MRR 0.959→0.950），因为所有差旅片段对任何差旅问题都
                 「确实相关」，分数挤在 0.86~0.99 分不开；但它的分数做**负例过滤**极强
                 （负例 Top1 ≤0.25，正例 ≥0.83，中间有 0.25~0.83 的空隙）。
                 所以最优组合是「RRF 排序 + 精排当闸门」。
        "rerank" 用精排分数**重排**（原始行为，保留用于 A/B 对比与后续调优）
        "off"    完全不调用精排（等价于 reranker=None）

    执行顺序（不能颠倒）：
        1. 对**全部候选**打分（候选池一般 10 条，远大于 top_k）
        2. mode="rerank" 才改用分数顺序；"filter" 恢复原顺序
        3. 只有拿到 rerank_score 才按 score_threshold 过滤
        4. 截断到 top_k；过滤后为空 → 返回 []，上游据此答「知识库中没有相关信息」

    ⚠️ 阈值只能作用于**精排分数**。RRF 分数（0.008~0.03）与余弦分数（-1~1）
       量纲完全不同，直接套用会把结果全部过滤光——这是本项目已经踩过一次的坑
       （见 hybrid_retriever / _search_hybrid 里"相似度阈值只作用于向量路"的注释）。
    """
    if not candidates:
        return []

    if mode == "off" or reranker is None:
        return list(candidates[:top_k])

    # "filter" 模式下精排分数只被 score_threshold 使用；没设阈值就是白跑一次推理，直接跳过。
    # 这样默认配置（enabled=True 但 threshold 未标定）不产生任何额外延迟。
    if mode == "filter" and score_threshold is None:
        logger.info(
            "Rerank 为 filter 模式但 score_threshold 未设置，跳过精排（无收益、不产生延迟）"
        )
        return list(candidates[:top_k])

    # 打上原顺序标记：rerank() 内部是 dict(cand) 拷贝，标记会原样带回来，
    # 这样"保持 RRF 顺序"不必依赖 id 字段是否唯一
    indexed: List[Dict] = []
    for i, cand in enumerate(candidates):
        item = dict(cand)
        item[_ORDER_KEY] = i
        indexed.append(item)

    scored = reranker.rerank(query, indexed, top_k=None)

    # 降级路径：没有精排分数说明打分没生效，只截断，**绝不用阈值过滤**
    if not scored or reranker.score_key not in scored[0]:
        logger.info("Rerank 未生效（降级），保持 RRF 顺序返回")
        return list(candidates[:top_k])

    if mode == "rerank":
        ordered = list(scored)
    else:
        # 闸门模式：只借分数做准入，顺序仍按 RRF（精排的排序在本语料上实测更差）
        ordered = sorted(scored, key=lambda d: d.get(_ORDER_KEY, 0))

    if score_threshold is not None:
        thr = float(score_threshold)
        kept = [d for d in ordered if d[reranker.score_key] >= thr]
        if len(kept) < len(ordered):
            logger.info(
                "Rerank 阈值过滤：丢弃 %d/%d 条（阈值 %.4f）",
                len(ordered) - len(kept), len(ordered), thr,
            )
        ordered = kept

    if not ordered:
        logger.info("Rerank 后无满足阈值的片段（query=%s）", query[:50])

    out: List[Dict] = []
    for d in ordered[:top_k]:
        d.pop(_ORDER_KEY, None)      # 清掉内部标记，不污染对外的溯源字段
        out.append(d)
    return out