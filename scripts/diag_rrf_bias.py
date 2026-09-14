#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RRF chunk 数偏置诊断。

要验证的假设：
    混合检索的 Hit@3 (51/53) 低于纯 BM25 (53/53)，根因不是「向量路语义质量差」，
    而是 RRF 按 **chunk 累加** 计分 —— 一篇文档若有多个 chunk 被召回，就各自
    贡献一份分数，等于给「chunk 多的文档」变相加权。语料里体积最大（chunk 最多）
    的文档因此被系统性抬高，体积最小的文档被系统性压低。

    推论（可证伪）：
      H1  期望文档是语料中体积最小的那几篇的 query，更容易失败
      H2  失败 query 的期望文档在融合后的 RRF 排名里，落在 ≥3 篇「chunk 更多」
          的文档之后
      H3  若在 **融合前** 对每条通道按文档去重（每篇文档只以最佳 chunk 投票），
          期望文档的 RRF 名次会上升，Hit@3 回升

═══════════════════════════════════════════════════════════════════════════════
结论（2026-09-14 实测）：
  H1  成立 —— 语料 90 chunk / 12 篇，单篇 5~10 个，投票机会相差 2.00 倍；
             两条失败用例的期望文档 chunk 数**恰好都是最小值 5**
  H2  成立 —— 逐票分解是决定性证据。例：「国际长途航班可以订什么舱位」的期望文档
             是 BM25 第 1 名（10.32，领先第 2 名 7.55 一大截）却只有 2 票，
             融合后排 #5；赢家 03_booking_guide 在任何一路都没进过前 2，靠 4 票
             拿到两倍分数。「紧急出差可以后补审批吗」期望文档 1 票 vs 04_faq 11 票。
  H3  **证伪** —— 融合前完全去重使 Hit@1 50→43、MRR 0.9528→0.8679，弄坏 7 条
             只修好 1 条（配对符号检验 p = 0.016，显著变差）；限流到 2/3 票完全无变化。
             根因：多 chunk 命中**既是偏置、也是真实信号**，硬去重把信号一起扔了。

  额外发现（真正的根因）：那两条"失败"是**标注过窄** —— 期望文档只是多个正确答案
  之一（「紧急出差可以后补审批吗」的答案出现在 12/04/01 三篇里）。按 eval_retrieval.py
  docstring 规定的元组标注修正后，混合检索 Hit@3 达到 53/53、未命中归零。

  完整记录见 README「注意事项 → RRF 的 chunk 数偏置」。
═══════════════════════════════════════════════════════════════════════════════

本脚本：
    A. 打印每篇文档的 chunk 数分布（偏置的来源）
    B. 对目标 query 逐条展开 dense / sparse 原始排名 + RRF 逐票分解 + 期望文档落点
    C. 跑全量 53 条，对比「现状」与「融合前文档级去重 / 每文档限 N 票」

只读诊断，不修改任何生产代码或配置。

用法：
  venv\\Scripts\\python.exe scripts\\diag_rrf_bias.py
  venv\\Scripts\\python.exe scripts\\diag_rrf_bias.py --skip-trace   # 只跑 A + C 段
  venv\\Scripts\\python.exe scripts\\diag_rrf_bias.py --topn 8
"""
import argparse
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from utils.hybrid_retriever import reciprocal_rank_fusion  # noqa: E402


def load_eval_module():
    """加载 eval_retrieval.py（复用它的 load_agent / CASES / evaluate，保证口径一致）。"""
    spec = importlib.util.spec_from_file_location(
        "eval_retrieval_mod", ROOT / "scripts" / "eval_retrieval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_retrieval_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# 与生产 search_knowledge 逐字一致的召回（见 agent.py:480-552）
# --------------------------------------------------------------------------

def raw_channels(agent, query):
    """返回 (dense_ranked, sparse_ranked, thr, min_bm25)，与生产同参数。"""
    cfg = agent._hybrid_cfg or {}
    kd = int(cfg.get("top_k_dense", 10))
    ks = int(cfg.get("top_k_sparse", 10))
    min_bm25 = float(cfg.get("min_bm25_score", 0.0) or 0.0)
    thr = agent.similarity_threshold

    dense = agent._vector_search(query, kd)
    dense_raw = list(dense)
    if thr is not None:
        dense = [d for d in dense if d["distance"] >= thr]

    sparse = [(cid, s) for cid, s in agent._bm25.search(query, ks) if s >= min_bm25]
    return dense_raw, dense, sparse, thr, min_bm25


def parent_of(agent, cid):
    rec = agent._corpus.get(cid) or {}
    return (rec.get("metadata") or {}).get("parent_doc", "?")


def dedup_by_doc(agent, ranked):
    """按文档去重：每篇文档只保留排名最靠前的那一个 chunk（保序）。"""
    seen, out = set(), []
    for cid, score in ranked:
        p = parent_of(agent, cid)
        if p in seen:
            continue
        seen.add(p)
        out.append((cid, score))
    return out


def fusion_with_dedup(agent, query):
    """反事实：融合前对两条通道各自按文档去重，再走标准 RRF。"""
    cfg = agent._hybrid_cfg or {}
    rrf_k = int(cfg.get("rrf_k", 60))
    dense_w = float(cfg.get("dense_weight", 1.0))
    sparse_w = float(cfg.get("sparse_weight", 1.0))

    _raw, dense, sparse, _thr, _mb = raw_channels(agent, query)
    d_ranked = [(d["id"], d["distance"]) for d in dense]
    return reciprocal_rank_fusion(
        [dedup_by_doc(agent, d_ranked), dedup_by_doc(agent, sparse)],
        k=rrf_k,
        weights=[dense_w, sparse_w],
    )


def cap_votes(agent, ranked, cap):
    """同一通道内，每篇文档最多保留 cap 个 chunk 参与投票（cap=1 即完全去重）。"""
    if not cap:
        return list(ranked)
    seen = Counter()
    out = []
    for cid, score in ranked:
        p = parent_of(agent, cid)
        if seen[p] >= cap:
            continue
        seen[p] += 1
        out.append((cid, score))
    return out


def make_search_capped(agent, cap):
    """反事实：融合前对每条通道做「每文档最多 cap 票」限流，再走标准 RRF。"""
    def _fn(_agent, query, k_):
        cfg = _agent._hybrid_cfg or {}
        rrf_k = int(cfg.get("rrf_k", 60))
        dense_w = float(cfg.get("dense_weight", 1.0))
        sparse_w = float(cfg.get("sparse_weight", 1.0))

        _raw, dense, sparse, _thr, _mb = raw_channels(_agent, query)
        d_ranked = [(d["id"], d["distance"]) for d in dense]
        fused = reciprocal_rank_fusion(
            [cap_votes(_agent, d_ranked, cap), cap_votes(_agent, sparse, cap)],
            k=rrf_k,
            weights=[dense_w, sparse_w],
        )
        out = []
        for cid, score in fused[:k_]:
            rec = _agent._corpus.get(cid)
            if rec is None:
                continue
            out.append({
                "id": cid,
                "content": rec["content"],
                "metadata": rec["metadata"],
                "distance": None,
                "rrf_score": round(score, 6),
            })
        return out
    return _fn


def make_search_dedup(agent, k):
    """构造一个可直接喂给 evaluate() 的检索回调：融合前文档级去重 → 取 Top-K。

    注意 evaluate() 的调用约定是 fn(agent, query, k)，且**异常被静默吞掉**
    （eval_retrieval.py:243）——签名写错会安静地返回 0/53 而不是报错。
    """
    def _fn(_agent, query, k_):
        fused = fusion_with_dedup(_agent, query)
        out = []
        for cid, score in fused[:k_]:
            rec = _agent._corpus.get(cid)
            if rec is None:
                continue
            out.append({
                "id": cid,
                "content": rec["content"],
                "metadata": rec["metadata"],
                "distance": None,
                "rrf_score": round(score, 6),
            })
        return out
    return _fn


# --------------------------------------------------------------------------
# A. chunk 数分布
# --------------------------------------------------------------------------

def part_a_chunk_distribution(agent):
    print("=" * 78)
    print("A. 每篇文档的 chunk 数分布（RRF 偏置的来源）")
    print("=" * 78)

    dist = Counter(parent_of(agent, cid) for cid in agent._corpus)
    total = sum(dist.values())
    print(f"\n语料共 {total} 个 chunk / {len(dist)} 篇文档\n")
    print(f"{'文档':<34}{'chunk 数':>9}{'占比':>9}   {'投票机会(相对最小)':>18}")
    print("-" * 78)

    lo = min(dist.values())
    for doc, n in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"{doc:<34}{n:>9}{n / total * 100:>8.1f}%   {n / lo:>17.2f}x")

    print(f"\n最多 {max(dist.values())} chunk / 最少 {lo} chunk "
          f"→ 投票机会相差 {max(dist.values()) / lo:.2f} 倍")
    return dist


# --------------------------------------------------------------------------
# B. 单条 query 的完整展开
# --------------------------------------------------------------------------

def part_b_trace(agent, queries, topn):
    cfg = agent._hybrid_cfg or {}
    rrf_k = int(cfg.get("rrf_k", 60))
    dense_w = float(cfg.get("dense_weight", 1.0))
    sparse_w = float(cfg.get("sparse_weight", 1.0))
    final_k = int(cfg.get("final_top_k", 3))

    for query, expect in queries:
        exp_set = {expect} if isinstance(expect, str) else set(expect)
        print("\n" + "=" * 78)
        print(f"B. query: {query}")
        print(f"   期望文档: {', '.join(sorted(exp_set))}")
        print("=" * 78)

        dense_raw, dense, sparse, thr, min_bm25 = raw_channels(agent, query)

        print(f"\n-- 向量路原始召回 (top_k_dense={len(dense_raw)}, "
              f"相似度阈值={thr}) --")
        print(f"{'rank':>4}  {'相似度':>8}  {'过阈值':>6}  文档 / chunk")
        d_rank = {}
        for i, d in enumerate(dense_raw, 1):
            ok = "✓" if thr is None or d["distance"] >= thr else "✗"
            p = (d.get("metadata") or {}).get("parent_doc", "?")
            print(f"{i:>4}  {d['distance']:>8.4f}  {ok:>6}  {p} / {d['id']}")
        for i, d in enumerate(dense, 1):
            d_rank[d["id"]] = i

        print(f"\n-- BM25 路原始召回 (top_k_sparse={len(sparse)}, "
              f"min_bm25_score={min_bm25}) --")
        print(f"{'rank':>4}  {'bm25':>8}  文档 / chunk")
        s_rank = {}
        for i, (cid, s) in enumerate(sparse, 1):
            print(f"{i:>4}  {s:>8.4f}  {parent_of(agent, cid)} / {cid}")
            s_rank[cid] = i

        # ---- 逐票分解 ----
        votes = defaultdict(list)          # parent_doc -> [(channel, rank, cid, contribution)]
        for d in dense:
            cid = d["id"]
            votes[parent_of(agent, cid)].append(
                ("dense", d_rank[cid], cid, dense_w / (rrf_k + d_rank[cid])))
        for cid, _ in sparse:
            votes[parent_of(agent, cid)].append(
                ("bm25", s_rank[cid], cid, sparse_w / (rrf_k + s_rank[cid])))

        print(f"\n-- 每篇文档的 RRF 投票明细（k={rrf_k}, "
              f"权重 dense:sparse={dense_w}:{sparse_w}）--")
        print(f"{'文档':<34}{'票数':>5}{'RRF 合计':>11}   明细")
        print("-" * 78)
        totals = {p: sum(v[3] for v in vs) for p, vs in votes.items()}
        for p, tot in sorted(totals.items(), key=lambda x: -x[1]):
            detail = " ".join(
                f"{ch}#{r}(+{c:.5f})" for ch, r, _cid, c in
                sorted(votes[p], key=lambda x: -x[3])
            )
            mark = "  <-- 期望" if p in exp_set else ""
            print(f"{p:<34}{len(votes[p]):>5}{tot:>11.5f}   {detail}{mark}")

        # ---- 融合排名（生产口径：不去重，截断 final_k）----
        d_ranked = [(d["id"], d["distance"]) for d in dense]
        fused = reciprocal_rank_fusion(
            [d_ranked, sparse], k=rrf_k, weights=[dense_w, sparse_w])

        print(f"\n-- RRF 融合排名（生产：max_per_doc=0，截断到 {final_k}）--")
        print(f"{'rank':>4}  {'RRF':>9}  文档 / chunk{'':<10}状态")
        hit_pos = None
        for i, (cid, sc) in enumerate(fused, 1):
            p = parent_of(agent, cid)
            flag = ""
            if p in exp_set and hit_pos is None:
                hit_pos = i
                flag = "  <== 期望文档首次出现"
            cut = "  [超出 Top-%d]" % final_k if i == final_k + 1 else ""
            print(f"{i:>4}  {sc:>9.5f}  {p} / {cid}{flag}{cut}")

        # ---- 反事实：融合前文档级去重 ----
        fused_dd = fusion_with_dedup(agent, query)
        print(f"\n-- 反事实：融合前按文档去重 --")
        print(f"{'rank':>4}  {'RRF':>9}  文档 / chunk")
        hit_pos_dd = None
        for i, (cid, sc) in enumerate(fused_dd, 1):
            p = parent_of(agent, cid)
            flag = ""
            if p in exp_set and hit_pos_dd is None:
                hit_pos_dd = i
                flag = "  <== 期望文档首次出现"
            print(f"{i:>4}  {sc:>9.5f}  {p} / {cid}{flag}")

        # ---- 结论 ----
        in_prod = hit_pos is not None and hit_pos <= final_k
        in_dd = hit_pos_dd is not None and hit_pos_dd <= final_k
        print(f"\n>> 期望文档融合名次：现状 #{hit_pos}（Top-{final_k} 内：{'是' if in_prod else '否'}）"
              f" → 去重后 #{hit_pos_dd}（{'是' if in_dd else '否'}）")

        if hit_pos:
            above = {parent_of(agent, c) for c, _ in fused[:hit_pos - 1]} - exp_set
            exp_doc = next(iter(exp_set))
            exp_votes = len(votes.get(exp_doc, []))
            print(f"   压在其上的文档（{len(above)} 篇）：")
            for p in sorted(above, key=lambda x: -totals.get(x, 0)):
                print(f"     - {p:<32} {len(votes.get(p, [])):>2} 票 / "
                      f"RRF {totals.get(p, 0):.5f}")
            print(f"   期望文档自身：{exp_votes} 票 / RRF {totals.get(exp_doc, 0):.5f}")


# --------------------------------------------------------------------------
# C. 全量对比
# --------------------------------------------------------------------------

def part_c_full_compare(agent, ev):
    k = 3
    cases = ev.CASES

    print("\n" + "=" * 78)
    print("C. 全量对比：现状 vs 融合前文档级去重")
    print("=" * 78)

    h1a, hka, mrra, per_a = ev.evaluate(agent, ev.search_hybrid, k, cases)
    h1c, hkc, mrrc, _ = ev.evaluate(agent, ev.search_bm25, k, cases)

    variants = [
        ("每文档最多 1 票（=去重）", make_search_capped(agent, 1)),
        ("每文档最多 2 票",          make_search_capped(agent, 2)),
        ("每文档最多 3 票",          make_search_capped(agent, 3)),
    ]
    results = []
    for label, fn in variants:
        results.append((label, fn, ev.evaluate(agent, fn, k, cases)))

    n = len(cases)
    print(f"\n{'方案':<28}{'Hit@1':>12}{'Hit@3':>12}{'MRR':>10}")
    print("-" * 78)
    print(f"{'现状（生产配置）':<28}{f'{h1a}/{n}':>12}{f'{hka}/{n}':>12}{mrra:>10.4f}   <== baseline")
    for label, _fn, (h1, hk, m, _p) in results:
        print(f"{label:<28}{f'{h1}/{n}':>12}{f'{hk}/{n}':>12}{m:>10.4f}")
    print(f"{'纯 BM25（对照，无 RRF）':<28}{f'{h1c}/{n}':>12}{f'{hkc}/{n}':>12}{mrrc:>10.4f}")

    # 逐条变化 + 配对符号检验（只看 Hit@1 是否命中）
    for label, _fn, (_h1, _hk, _m, per_b) in results:
        print(f"\n-- 逐条变化：现状 -> {label} --")
        better = worse = 0
        for (q, _exp), a, b in zip(cases, per_a, per_b):
            ra, rb = a.get("rank"), b.get("rank")
            if ra == rb:
                continue
            a1, b1 = (ra == 1), (rb == 1)
            if a1 == b1:
                continue
            if b1:
                better += 1
                print(f"  ↑ {q[:34]:<36} #{ra} -> #{rb}")
            else:
                worse += 1
                print(f"  ↓ {q[:34]:<36} #{ra} -> #{rb}")
        tot = better + worse
        if tot:
            # 双侧符号检验
            from math import comb
            p = 2 * sum(comb(tot, i) for i in range(0, min(better, worse) + 1)) / (2 ** tot)
            p = min(p, 1.0)
            print(f"  Hit@1 翻转 {better} 好 / {worse} 坏 → 双侧符号检验 p = {p:.3f}"
                  f"{'（显著）' if p < 0.05 else '（不显著）'}")
        else:
            print("  （Hit@1 无翻转）")

    per_b = results[0][2][3]
    print(f"\n-- 期望文档 chunk 数与成败的关系 --")
    dist = Counter(parent_of(agent, cid) for cid in agent._corpus)
    mn, mx = min(dist.values()), max(dist.values())
    fails = [(q, e) for (q, e), a in zip(cases, per_a) if a.get("rank") is None]
    if fails:
        print(f"  现状失败的 {len(fails)} 条（语料 chunk 数范围 {mn}~{mx}）：")
        for q, e in fails:
            doc = e if isinstance(e, str) else e[0]
            print(f"    {q[:34]:<36} {doc:<32} chunk {dist.get(doc, '?')}")
    else:
        print("  现状无失败用例")

    # results: [(label, fn, (hit1, hitk, mrr, per_case)), ...]
    return per_a, {label: r[3] for label, _fn, r in results}


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topn", type=int, default=10, help="每路展示条数")
    ap.add_argument("--skip-full", action="store_true", help="跳过 C 段全量对比")
    ap.add_argument("--skip-trace", action="store_true", help="跳过 B 段逐条展开")
    args = ap.parse_args()

    ev = load_eval_module()
    print("加载 RAGKnowledgeAgent（会建 BM25 索引）…")
    agent = ev.load_agent()
    print(f"OK：语料 {len(agent._corpus)} chunk")

    dist = part_a_chunk_distribution(agent)

    # 目标：报告里混合检索未命中的两条 + 两条对照（纯 BM25 也命中、混合也命中的）
    target = [
        ("国际长途航班可以订什么舱位", "01_travel_standards.txt"),
        ("紧急出差可以后补审批吗",     "12_seasonal_policies.txt"),
    ]
    controls = [
        ("酒店取消预订有什么政策",     "03_booking_guide.txt"),
        ("会员等级是怎么评定的",       "09_member_benefits.txt"),
    ]
    if not args.skip_trace:
        part_b_trace(agent, target, args.topn)
        print("\n\n" + "#" * 78)
        print("# 对照组（混合检索原本就命中的 query）")
        print("#" * 78)
        part_b_trace(agent, controls, args.topn)

    if not args.skip_full:
        part_c_full_compare(agent, ev)

    print("\n诊断完成。")


if __name__ == "__main__":
    main()
