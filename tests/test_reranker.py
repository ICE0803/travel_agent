"""
Rerank 精排单测（离线，不依赖网络 / 模型文件 / Milvus）

用 scorer 参数注入假打分器，覆盖全部逻辑分支：
三种 mode、排序、截断、稳定排序、降级、阈值过滤，
以及最容易出错的两条规则——
  a. 降级时**绝不能用阈值过滤**（没有分数可过滤，硬过滤会把结果全清空）
  b. mode="filter" 必须**保持原顺序**（精排只当准入闸门，不参与排序）

运行：
  venv\\Scripts\\python.exe tests\\test_reranker.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from utils.reranker import CrossEncoderReranker, apply_rerank, resolve_model_path

PASSED, FAILED = [], []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  [{detail}]" if detail and not cond else ""))


# 候选按 RRF 顺序给出：id 越小 RRF 排名越靠前
CANDS = [
    {"id": 1, "content": "一线城市出差住宿标准500元", "rrf_score": 0.030, "matched_by": "vector+bm25"},
    {"id": 2, "content": "报销需15个工作日内提交", "rrf_score": 0.028, "matched_by": "vector"},
    {"id": 3, "content": "绿色出行倡议", "rrf_score": 0.026, "matched_by": "vector"},
    {"id": 4, "content": "会员等级评定", "rrf_score": 0.024, "matched_by": "bm25"},
]

# 假分数：故意让 id=2 拿到最高分（RRF 排第 2），用来验证 mode 的差别
FAKE_SCORES = {1: 0.10, 2: 0.95, 3: 0.02, 4: 0.05}
_SCORE_BY_TEXT = {c["content"]: FAKE_SCORES[c["id"]] for c in CANDS}


def fake_scorer(query, texts):
    """确定性打分器：按文本内容返回固定分数，不加载任何模型。"""
    return [_SCORE_BY_TEXT.get(t, 0.0) for t in texts]


class CountingScorer:
    """包装一个打分器并计数调用次数，用来证明"该跳过精排时确实没调用"。"""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, query, texts):
        self.calls += 1
        return self.fn(query, texts)


def boom_scorer(query, texts):
    raise RuntimeError("模拟打分后端故障")


def wrong_len_scorer(query, texts):
    return [0.5]      # 条数不匹配


def ids(items):
    return [c["id"] for c in items]


def main():
    print("=" * 68)
    print("Rerank 精排单测（离线，注入假打分器）")
    print("=" * 68)

    rr = CrossEncoderReranker(scorer=fake_scorer)

    print("\n[1] rerank()：按分数降序重排")
    out = rr.rerank("差旅住宿标准", CANDS)
    check("available=True 且不加载模型", rr.available is True)
    check("排序 = 分数降序", ids(out) == [2, 1, 4, 3], str(ids(out)))
    check("每条都带 rerank_score", all("rerank_score" in c for c in out))
    check("分数正确", [round(c["rerank_score"], 2) for c in out] == [0.95, 0.10, 0.05, 0.02])

    print("\n[2] rerank()：保留原有字段（可追溯性）")
    check("id 保留", ids(out) == [2, 1, 4, 3])
    check("rrf_score 保留", out[0]["rrf_score"] == 0.028)
    check("matched_by 保留", out[0]["matched_by"] == "vector")

    print("\n[3] rerank()：不修改入参（纯函数语义）")
    check("原列表未被写入 rerank_score", all("rerank_score" not in c for c in CANDS))
    check("原列表顺序未变", ids(CANDS) == [1, 2, 3, 4])

    print("\n[4] rerank()：top_k 截断")
    check("top_k=2 只返回 2 条", len(rr.rerank("q", CANDS, top_k=2)) == 2)
    check("截断取的是最高分", ids(rr.rerank("q", CANDS, top_k=2)) == [2, 1])

    print("\n[5] rerank()：同分保持原顺序（稳定排序）")
    flat = CrossEncoderReranker(scorer=lambda q, ts: [0.5] * len(ts))
    check("全部同分时顺序不变", ids(flat.rerank("q", CANDS)) == [1, 2, 3, 4])

    print("\n[6] apply_rerank mode='rerank'：用精排分数排序")
    got = apply_rerank(rr, "q", CANDS, top_k=3, mode="rerank")
    check("取到精排 Top-3", ids(got) == [2, 1, 4], str(ids(got)))
    check("带 rerank_score", all("rerank_score" in c for c in got))

    print("\n[7] apply_rerank mode='filter'：保持 RRF 顺序（核心语义）")
    got = apply_rerank(rr, "q", CANDS, top_k=3, score_threshold=0.08, mode="filter")
    check("顺序 = RRF 顺序（id=1 在 id=2 之前，与重排模式的 [2,1] 相反）",
          ids(got) == [1, 2], str(ids(got)))
    check("低于阈值的被丢掉", 3 not in ids(got) and 4 not in ids(got))
    check("带 rerank_score", all("rerank_score" in c for c in got))
    check("内部标记 _rerank_order 已清除",
          all("_rerank_order" not in c for c in got))

    print("\n[8] apply_rerank：过滤后为空 → 返回 []")
    check("阈值 0.99 全部被过滤", apply_rerank(rr, "q", CANDS, top_k=3,
                                             score_threshold=0.99, mode="filter") == [])
    check("重排模式同理", apply_rerank(rr, "q", CANDS, top_k=3,
                                       score_threshold=0.99, mode="rerank") == [])

    print("\n[9] apply_rerank mode='filter' 且未设阈值：跳过精排（零开销）")
    cnt = CountingScorer(fake_scorer)
    rr_cnt = CrossEncoderReranker(scorer=cnt)
    got = apply_rerank(rr_cnt, "q", CANDS, top_k=3, score_threshold=None, mode="filter")
    check("打分器未被调用", cnt.calls == 0, f"calls={cnt.calls}")
    check("返回原顺序前 3 条", ids(got) == [1, 2, 3], str(ids(got)))
    check("无 rerank_score", all("rerank_score" not in c for c in got))

    print("\n[10] apply_rerank mode='off'：完全不调用精排")
    got = apply_rerank(rr_cnt, "q", CANDS, top_k=2, mode="off")
    check("打分器未被调用", cnt.calls == 0, f"calls={cnt.calls}")
    check("返回原顺序", ids(got) == [1, 2], str(ids(got)))

    print("\n[11] 降级：模型路径不存在")
    bad = CrossEncoderReranker("data/models/__definitely_not_here__")
    check("available=False", bad.available is False)
    got = apply_rerank(bad, "q", CANDS, top_k=2, mode="rerank")
    check("退回原顺序、不抛异常", ids(got) == [1, 2], str(ids(got)))
    check("无 rerank_score（调用方据此识别降级）",
          all("rerank_score" not in c for c in got))

    print("\n[12] 降级 + 阈值：**阈值绝不能被应用**（防误伤，最重要）")
    got = apply_rerank(bad, "q", CANDS, top_k=4, score_threshold=0.99, mode="filter")
    check("降级时不因阈值过滤而清空", len(got) == 4, str(ids(got)))
    check("顺序保持原样", ids(got) == [1, 2, 3, 4], str(ids(got)))

    print("\n[13] 打分器抛异常 → 退回原顺序")
    got = apply_rerank(CrossEncoderReranker(scorer=boom_scorer), "q", CANDS,
                       top_k=3, mode="rerank")
    check("不抛异常且顺序不变", ids(got) == [1, 2, 3], str(ids(got)))
    check("无 rerank_score", all("rerank_score" not in c for c in got))

    print("\n[14] 打分条数不匹配 → 退回原顺序")
    got = apply_rerank(CrossEncoderReranker(scorer=wrong_len_scorer), "q", CANDS,
                       top_k=3, mode="rerank")
    check("不抛异常且顺序不变", ids(got) == [1, 2, 3], str(ids(got)))

    print("\n[15] 边界：空候选 / reranker=None")
    check("空候选返回空", apply_rerank(rr, "q", [], top_k=3, mode="rerank") == [])
    check("空候选 + filter 返回空",
          apply_rerank(rr, "q", [], top_k=3, score_threshold=0.1, mode="filter") == [])
    check("reranker=None 只截断",
          ids(apply_rerank(None, "q", CANDS, top_k=2)) == [1, 2])
    check("rerank 空候选返回空", rr.rerank("q", []) == [])

    print("\n[16] resolve_model_path：本地目录优先")
    check("不存在的路径返回 None", resolve_model_path("data/models/__nope__") is None)
    check("空值返回 None", resolve_model_path(None) is None and resolve_model_path("") is None)
    check("存在的目录返回绝对路径", resolve_model_path(".") is not None)
    check("绝对路径也支持",
          resolve_model_path(str(Path(".").resolve())) is not None)

    print("\n[17] stats()：降级状态可诊断")
    check("injected_scorer=True", rr.stats()["injected_scorer"] is True)
    check("load_failed=True（路径不存在）", bad.stats()["load_failed"] is True)

    print(f"\n通过 {len(PASSED)} / {len(PASSED) + len(FAILED)}")
    if FAILED:
        print("失败项：")
        for f in FAILED:
            print(f"  - {f}")
    print("=" * 68)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
