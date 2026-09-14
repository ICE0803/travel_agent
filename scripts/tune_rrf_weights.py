#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RRF 通道权重调优：扫描 dense:sparse 权重比，输出 Hit@k / MRR 对比表

背景（实测）：
    纯 BM25        Hit@1=46/53  Hit@3=53/53  MRR=0.928
    混合(RRF)      Hit@1=49/53  Hit@3=51/53  MRR=0.943
  单靠 BM25 的 Hit@3 是满分，把向量路加进来做**等权** RRF 反而掉了 2 条 ——
  说明关键词路的排序更准，被向量路稀释了。这个脚本就是来定权重比的。

原理：RRF(d) = Σ w_i / (k + rank_i(d))，只有 w_dense : w_sparse 的**比值**有意义。

用法：
  venv\\Scripts\\python.exe scripts\\tune_rrf_weights.py
  venv\\Scripts\\python.exe scripts\\tune_rrf_weights.py --grid 1:1,1:2,1:5   # 自定义（短列表）

注意：默认网格写在脚本里，命令行**不用传参数**即可跑 —— 某些 Windows shell 下
传 6 个以上逗号分隔值会触发进程创建失败。
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import scripts.eval_retrieval as ev

# 默认网格：dense 固定 1.0，sparse 从等权一路加到"关键词主导"
DEFAULT_GRID = [(1.0, 1.0), (1.0, 1.5), (1.0, 2.0), (1.0, 3.0), (1.0, 5.0), (1.0, 10.0)]


def parse_grid(text):
    pairs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"'{item}' 格式应为 dense:sparse，例如 1:2")
        d, s = item.split(":", 1)
        pairs.append((float(d), float(s)))
    return pairs


def main():
    ap = argparse.ArgumentParser(description="RRF 通道权重扫描")
    ap.add_argument("--grid", default=None,
                    help="逗号分隔的 dense:sparse 列表（默认用脚本内置网格）")
    ap.add_argument("--fine", type=float, default=None, metavar="SPARSE",
                    help="在指定 sparse 权重附近细扫（如 --fine 1.5），"
                         "用来确认峰值不是噪声尖峰。只传一个数，不受长逗号列表的 shell 问题影响")
    ap.add_argument("--k", type=int, default=3, help="Top-K（默认 3）")
    args = ap.parse_args()
    k = args.k

    try:
        if args.fine is not None:
            # 以候选值为中心、±30% 范围取 5 个点，检查峰是否够宽
            grid = [(1.0, round(args.fine * f, 4)) for f in (0.7, 0.85, 1.0, 1.15, 1.3)]
        elif args.grid:
            grid = parse_grid(args.grid)
        else:
            grid = list(DEFAULT_GRID)
    except ValueError as e:
        print(f"✗ --grid 解析失败：{e}")
        return 1

    print("=" * 78)
    print("RRF 通道权重扫描（混合检索，不含精排）")
    print("=" * 78)

    agent = ev.load_agent()
    if not agent.initialized:
        print("✗ RAG Agent 未初始化：请先跑 "
              "python .claude/skills/ask-question/script/init_knowledge_base.py")
        return 1
    if agent._bm25 is None:
        print("✗ BM25 索引未构建（hybrid.enabled 是否为 false？）")
        return 1

    cfg = agent._hybrid_cfg
    saved = (cfg.get("dense_weight", 1.0), cfg.get("sparse_weight", 1.0))
    print(f"\n当前配置 dense_weight={saved[0]}  sparse_weight={saved[1]}  "
          f"rrf_candidates={cfg.get('rrf_candidates')}  max_per_doc={cfg.get('max_per_doc')}")
    print(f"相似度阈值={agent.similarity_threshold}  "
          f"min_bm25_score={cfg.get('min_bm25_score')}\n")

    # 参考行：纯 BM25（不走 RRF，直接取 BM25 ranking）
    ref = {}
    for name, fn in (("纯向量", ev.search_vector), ("纯 BM25", ev.search_bm25)):
        h1, hk, mrr, _ = ev.evaluate(agent, fn, k, ev.CASES)
        ref[name] = (h1, hk, mrr)

    print(f"{'dense:sparse':<16}{'Hit@1':<12}{f'Hit@{k}':<12}{'MRR'}")
    print("-" * 78)
    for name, (h1, hk, mrr) in ref.items():
        print(f"{name + '（参考）':<16}{h1}/{len(ev.CASES):<9}"
              f"{hk}/{len(ev.CASES):<9}{mrr:.3f}")
    print("-" * 78)

    rows = []
    t0 = time.time()
    try:
        for dw, sw in grid:
            cfg["dense_weight"] = dw
            cfg["sparse_weight"] = sw
            h1, hk, mrr, _ = ev.evaluate(agent, ev.search_hybrid, k, ev.CASES)
            rows.append((dw, sw, h1, hk, mrr))
            mark = "  ← 当前" if (dw, sw) == saved else ""
            print(f"{f'{dw:g}:{sw:g}':<16}{h1}/{len(ev.CASES):<9}"
                  f"{hk}/{len(ev.CASES):<9}{mrr:.3f}{mark}")
    finally:
        cfg["dense_weight"], cfg["sparse_weight"] = saved

    if not rows:
        print("✗ 没有可用结果")
        return 1

    # 判定：Hit@3 优先，其次 MRR，再其次 Hit@1
    best_key = max((r[3], r[4], r[2]) for r in rows)
    tied = sorted([r for r in rows if (r[3], r[4], r[2]) == best_key], key=lambda r: r[1])
    # 平局时取 sparse 权重**最居中**的那个：指标是台阶状的（权重微调会让某一条
    # query 的并列关系翻转，只差 ±1 条），选平台边缘容易过拟合到测试集，选中间更稳。
    best = tied[len(tied) // 2]
    base = next((r for r in rows if (r[0], r[1]) == (1.0, 1.0)), None)
    print("\n" + "=" * 78)
    print(f"最优：dense:sparse = {best[0]:g}:{best[1]:g}  "
          f"Hit@1={best[2]}/{len(ev.CASES)}  Hit@{k}={best[3]}/{len(ev.CASES)}  MRR={best[4]:.3f}")
    if len(tied) > 1:
        print(f"（{len(tied)} 组指标完全相同：sparse {tied[0][1]:g} ~ {tied[-1][1]:g}，"
              f"已取最居中的 {best[1]:g}；平台边缘容易过拟合到这 53 条测试集）")
    if base:
        print(f"对比等权基线（1:1）：Hit@1 {base[2]} → {best[2]}（{best[2] - base[2]:+d}）　"
              f"Hit@{k} {base[3]} → {best[3]}（{best[3] - base[3]:+d}）　"
              f"MRR {base[4]:.3f} → {best[4]:.3f}（{best[4] - base[4]:+.3f}）")
        if best[3] == base[3] and abs(best[4] - base[4]) < 1e-9:
            print("⚠ 扫描没有任何改进 —— 权重不是这个语料的杠杆，保持 1:1 即可。")
        elif best[3] == base[3]:
            print(f"⚠ Hit@{k} 完全没动（{best[3]}/{len(ev.CASES)}）—— "
                  f"权重救不了那几条失败用例，它只改善了排名质量（Hit@1 / MRR）。")
            print("  53 条样本上 ±1 条已在噪声区间，建议只用 --fine 确认平台够宽再采用。")
    print(f"\n要把结果写进 config.py → RAG_CONFIG['hybrid']：\n"
          f"    \"dense_weight\": {best[0]:g},\n"
          f"    \"sparse_weight\": {best[1]:g},")
    print(f"\n（扫描 {len(grid)} 组权重，耗时 {time.time() - t0:.1f}s；只改内存，未写回 config）")

    agent.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
