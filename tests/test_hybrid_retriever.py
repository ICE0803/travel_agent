"""
混合检索单测（离线，不依赖网络 / Milvus / jieba）

jieba 装或不装都能跑：分词用例按当前后端分支断言，
另外强制走一遍 bigram 降级路径，保证两条路都有覆盖。

运行：
  venv\\Scripts\\python.exe tests\\test_hybrid_retriever.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 需要模块对象本身，才能在测试里临时切换分词后端
import utils.hybrid_retriever as hr
from utils.hybrid_retriever import BM25Index, reciprocal_rank_fusion, tokenize

PASSED, FAILED = [], []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  [{detail}]" if detail and not cond else ""))


CORPUS = [
    (1, "一线城市出差住宿标准不超过500元每天，推荐企业协议酒店"),
    (2, "二线城市住宿标准不超过400元每天"),
    (3, "差旅费用报销需在出差结束后15个工作日内提交申请单"),
    (4, "航班延误超过4小时可申请改签或全额退票"),
    (5, "预订机票建议提前7天以上，旺季需提前15天"),
]


class forced_bigram:
    """上下文管理器：临时强制 tokenize 走字符二元组降级路径。"""

    def __enter__(self):
        self._saved = hr.JIEBA_AVAILABLE
        hr.JIEBA_AVAILABLE = False
        return self

    def __exit__(self, *exc):
        hr.JIEBA_AVAILABLE = self._saved
        return False


def main():
    backend = "jieba" if hr.JIEBA_AVAILABLE else "bigram(降级)"
    print("=" * 68)
    print(f"混合检索单测（离线）  当前分词后端: {backend}")
    print("=" * 68)

    print(f"\n[1] 分词：当前后端（{backend}）")
    t = tokenize("差旅报销ABC123")
    if hr.JIEBA_AVAILABLE:
        check("jieba 切成词（不是二元组）", "差旅" in t and "报销" in t, str(t))
        check("jieba 不产生跨词二元组 '旅报'", "旅报" not in t, str(t))
    else:
        check("降级路径切成二元组", "差旅" in t and "旅报" in t and "报销" in t, str(t))
    check("英文数字整体小写", "abc123" in t, str(t))
    check("单字也能成 token", tokenize("去") == ["去"], str(tokenize("去")))
    check("空串返回空", tokenize("") == [])

    print("\n[2] 分词：强制 bigram 降级路径（无论 jieba 是否安装）")
    with forced_bigram():
        tb = tokenize("差旅报销")
        check("切成二元组", tb == ["差旅", "旅报", "报销"], str(tb))
        check("单字段保留单字", tokenize("去") == ["去"], str(tokenize("去")))
        check("英文数字仍整体", "abc123" in tokenize("差旅ABC123"), str(tokenize("差旅ABC123")))
        # 降级路径下 BM25 仍应正常工作
        idx_b = BM25Index(CORPUS)
        rb = idx_b.search("航班延误", top_k=2)
        check("降级路径 BM25 可用", bool(rb) and rb[0][0] == 4, str(rb))

    print("\n[3] 分词：停用词过滤")
    t_stop = tokenize("这个标准是什么", drop_stopwords=True)
    t_keep = tokenize("这个标准是什么", drop_stopwords=False)
    check("过滤后虚词消失", "这个" not in t_stop and "什么" not in t_stop, str(t_stop))
    check("过滤后实词保留", "标准" in t_stop, str(t_stop))
    check("不过滤时虚词仍在", "这个" in t_keep or "什么" in t_keep, str(t_keep))
    check("纯标点返回空", tokenize("，。！？") == [])
    check("停用词表非空", len(hr.STOPWORDS) > 50, str(len(hr.STOPWORDS)))

    print("\n[4] BM25：关键词精确命中应排第一")
    idx = BM25Index(CORPUS)
    r = idx.search("住宿标准", top_k=3)
    check("有结果", len(r) > 0)
    check("doc1 或 doc2 排第一（都含住宿标准）", r and r[0][0] in (1, 2), str(r))

    print("\n[5] BM25：罕见词权重更高（IDF 生效）")
    r2 = idx.search("航班延误", top_k=3)
    check("doc4 排第一", r2 and r2[0][0] == 4, str(r2))

    print("\n[6] BM25：完全无关的查询返回空")
    r3 = idx.search("量子计算机", top_k=3)
    check("无命中", r3 == [], str(r3))

    print("\n[7] BM25：结果按分数降序")
    r4 = idx.search("出差", top_k=5)
    check("分数单调不增",
          all(r4[i][1] >= r4[i + 1][1] for i in range(len(r4) - 1)), str(r4))

    print("\n[8] RRF：两路都命中的文档排名应上升")
    # doc A 在通道1排第1、通道2排第3；doc B 只在通道1排第2
    ch1 = [("A", 9.0), ("B", 8.0), ("C", 1.0)]
    ch2 = [("C", 5.0), ("D", 4.0), ("A", 3.0)]
    fused = reciprocal_rank_fusion([ch1, ch2], k=60)
    scores = dict(fused)
    check("A 因两路命中而得分最高", fused[0][0] == "A", str(fused))
    check("A 得分 = 1/61 + 1/63",
          abs(scores["A"] - (1 / 61 + 1 / 63)) < 1e-9, str(scores["A"]))
    check("C 两路也命中", "C" in scores)

    print("\n[9] RRF：只在一路出现的文档也会保留")
    check("B 被保留", "B" in scores)
    check("D 被保留", "D" in scores)

    print("\n[10] RRF：权重生效")
    w = reciprocal_rank_fusion([ch1, ch2], k=60, weights=[10.0, 1.0])
    order = [d for d, _ in w]
    check("加大通道1权重后 B 排在 D 之前",
          order.index("B") < order.index("D"), str(w))

    print("\n[11] RRF：空输入不报错")
    check("全空返回空", reciprocal_rank_fusion([[], []]) == [])
    check("单通道可用", len(reciprocal_rank_fusion([ch1])) == 3)

    print("\n[12] 边界：空语料 / 空查询")
    empty = BM25Index([])
    check("空语料 search 返回空", empty.search("x") == [])
    check("空查询返回空", idx.search("") == [])
    check("纯标点查询返回空", idx.search("，。！") == [])

    print(f"\n通过 {len(PASSED)} / {len(PASSED) + len(FAILED)}")
    if FAILED:
        print("失败项：")
        for f in FAILED:
            print(f"  - {f}")
    print("=" * 68)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
