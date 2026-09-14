#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rerank 延迟实测：CPU 上精排到底吃掉多少时间

用法：
  venv\\Scripts\\python.exe scripts\\bench_rerank.py
  venv\\Scripts\\python.exe scripts\\bench_rerank.py --candidates 5 --max-length 256
"""
import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from utils.reranker import CrossEncoderReranker


def load_agent():
    spec = importlib.util.spec_from_file_location(
        "rag_bench_agent",
        ROOT / ".claude" / "skills" / "ask-question" / "script" / "agent.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["rag_bench_agent"] = module
    spec.loader.exec_module(module)
    return module.RAGKnowledgeAgent(name="RAGKnowledgeAgent", model=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=10)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    from config import RAG_CONFIG

    agent = load_agent()
    if not getattr(agent, "initialized", False):
        print("✗ RAG Agent 未初始化：请先运行 "
              "python .claude/skills/ask-question/script/init_knowledge_base.py")
        return 1
    if agent._bm25 is None:
        print("✗ BM25 索引未构建（hybrid.enabled 是否为 false？）")
        return 1

    rr = CrossEncoderReranker(
        model_name_or_path=(RAG_CONFIG.get("rerank") or {}).get("model"),
        max_length=args.max_length,
    )

    queries = ["差旅住宿标准是多少", "报销需要准备哪些材料", "会员等级怎么评定"]

    # ⚠️ 关键：search_knowledge() 受 final_top_k 截断（默认 3）。
    # 不放宽这两个值的话，hits[:10] 最多只能拿到 3 条，
    # 会打印「候选数=10」却只测了 3 条 —— 延迟被低估约 3 倍。
    agent._hybrid_cfg["final_top_k"] = args.candidates
    agent._hybrid_cfg["rrf_candidates"] = args.candidates

    # 1) 模型载入耗时
    t = time.time()
    if not rr.available:
        print("✗ 精排模型不可用，先跑 scripts/download_reranker.py")
        return 1
    print(f"模型载入耗时: {time.time() - t:.2f}s")

    # 2) 单次精排耗时
    lat, cand_lens, cand_ns = [], [], []
    for q in queries:
        hits = agent.search_knowledge(q)
        cands = hits[: args.candidates]
        if not cands:
            continue
        cand_ns.append(len(cands))
        cand_lens.append(sum(len(c["content"]) for c in cands) / len(cands))
        for _ in range(args.rounds):
            t = time.time()
            rr.rerank(q, cands)
            lat.append(time.time() - t)

    if not lat:
        print("✗ 没取到候选，检查知识库是否已初始化")
        return 1

    if min(cand_ns) != args.candidates:
        print(f"⚠ 实际候选数 {min(cand_ns)}~{max(cand_ns)}，少于请求的 {args.candidates}"
              f"（说明 RRF 池本身不够宽）")

    print(f"\n候选数={args.candidates}（实际 {min(cand_ns)}~{max(cand_ns)}）  "
          f"max_length={args.max_length}  候选平均字数={statistics.mean(cand_lens):.0f}")
    print(f"单次精排: 中位数 {statistics.median(lat):.3f}s  "
          f"最小 {min(lat):.3f}s  最大 {max(lat):.3f}s  （{len(lat)} 次）")

    # 换算成最终 top-3 的成本参考：精排是对整个候选池打分，与最终返回几条无关
    print(f"\n延迟口径：精排要对**整个候选池**打分，与最终返回 3 条无关。")
    print(f"  → 一次 RAG 问答的额外成本 ≈ {statistics.median(lat):.3f}s（上表数字）")
    print(f"  → 想降下来就调小 --candidates 或 --max-length，"
          f"并回跑 scripts/eval_retrieval.py 确认精度没掉")
    print(f"  → 注意：README 的「响应时间 15s」指标要把这个数加进 RAG 那一段")

    agent.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())