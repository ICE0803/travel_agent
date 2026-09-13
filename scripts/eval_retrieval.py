#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检索效果评测：纯向量 vs 纯 BM25 vs 混合（RRF）

用途：
  1. 回答「混合检索到底有没有用」——给出 Hit@k / MRR 三路对比
  2. 为 min_bm25_score 标定阈值 —— 用负例查询的分数分布找分界点
  3. 把简历里的「准确率 XX%」变成可复现的数字

用法：
  venv\\Scripts\\python.exe scripts\\eval_retrieval.py
  venv\\Scripts\\python.exe scripts\\eval_retrieval.py --k 5
  venv\\Scripts\\python.exe scripts\\eval_retrieval.py --no-save

标注集说明：
  期望文档可以写**一个字符串**，也可以写**元组**表示多个文档都算对。
  语料本身有交叉（FAQ 覆盖了很多主题，会员权益与报销规定也互相呼应），
  遇到「两篇都答得上」的情况就用元组标注，否则评测结果会掺入标注噪声。
  CASES 里的期望文档是主要答案所在文档，只要求出现在 Top-K 中即算命中。
  想扩到 100+ 条时，直接往 CASES 里加即可。
"""
import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# --------------------------------------------------------------------------
# 标注集：query -> 期望命中的源文档
# 构造原则：优先挑**该文档独有**的内容，减少与其他文档的歧义
# --------------------------------------------------------------------------
CASES = [
    # ---- 01 差旅标准与规定 ----
    ("超标住宿费由谁审批",              "01_travel_standards.txt"),
    ("西藏新疆出差住宿标准能上浮多少",    "01_travel_standards.txt"),
    ("三线及以下城市的住宿标准",         "01_travel_standards.txt"),
    ("国际长途航班可以订什么舱位",        "01_travel_standards.txt"),

    # ---- 02 报销规定 ----
    ("出差结束后多少天内要提交报销",      "02_reimbursement_policy.txt"),
    ("报销发票的抬头有什么要求",         "02_reimbursement_policy.txt"),
    ("报销需要准备哪些材料",            "02_reimbursement_policy.txt"),
    ("报销审批要经过哪几个环节",         "02_reimbursement_policy.txt"),
    ("跨年度出差什么时候必须完成报销",    "02_reimbursement_policy.txt"),

    # ---- 03 预订指南 ----
    ("火车票退票费怎么收取",            "03_booking_guide.txt"),
    ("酒店一般几点可以办理入住",         "03_booking_guide.txt"),
    ("租车取车需要提供什么证件",         "03_booking_guide.txt"),
    ("国际机票建议提前多久预订",         "03_booking_guide.txt"),
    ("酒店取消预订有什么政策",          "03_booking_guide.txt"),

    # ---- 04 FAQ ----
    ("出差可以携带家属吗",              "04_faq.txt"),
    ("出差期间生病医疗费能报销吗",        "04_faq.txt"),
    # 04 与 09 都讲了"个人里程/积分抵扣部分不予报销"，两篇都算对
    ("可以用个人里程积分订机票吗",        ("04_faq.txt", "09_member_benefits.txt")),
    ("出差申请单丢失了怎么办",           "04_faq.txt"),
    ("车辆违章罚款可以报销吗",           "04_faq.txt"),

    # ---- 05 紧急处理 ----
    ("托运行李丢失怎么处理",            "05_emergency_procedures.txt"),
    ("护照在国外丢了怎么办",            "05_emergency_procedures.txt"),
    ("酒店发生火灾怎么疏散",            "05_emergency_procedures.txt"),
    ("出差期间遭遇抢劫怎么办",           "05_emergency_procedures.txt"),
    ("出差时地震了怎么应对",            "05_emergency_procedures.txt"),
    ("食物中毒如何处理",               "05_emergency_procedures.txt"),

    # ---- 06 平台使用指南 ----
    ("阿里商旅APP怎么下载",            "06_platform_guide.txt"),
    ("平台登录密码有什么要求",           "06_platform_guide.txt"),
    ("怎么在平台上提交差旅申请",         "06_platform_guide.txt"),
    ("平台上怎么改签机票",              "06_platform_guide.txt"),

    # ---- 07 城市指南 ----
    ("北京机场到市区怎么走",            "07_city_specific_tips.txt"),
    ("上海出差推荐住哪个区域",           "07_city_specific_tips.txt"),
    ("广州的天气有什么特点",            "07_city_specific_tips.txt"),

    # ---- 08 环保倡议 ----
    ("公司有哪些绿色出行倡议",           "08_environmental_initiatives.txt"),
    ("怎么减少一次性用品的使用",         "08_environmental_initiatives.txt"),
    ("垃圾分类有什么规定",              "08_environmental_initiatives.txt"),
    ("绿色会议应该怎么做",              "08_environmental_initiatives.txt"),

    # ---- 09 会员权益 ----
    ("会员等级是怎么评定的",            "09_member_benefits.txt"),
    ("积分有有效期吗",                 "09_member_benefits.txt"),
    ("积分可以转让给同事吗",            "09_member_benefits.txt"),
    ("会员等级会影响报销标准吗",         "09_member_benefits.txt"),

    # ---- 10 国际差旅 ----
    ("国际差旅伙食补贴标准是多少",        "10_international_travel.txt"),
    ("出国审批需要经过哪些层级",         "10_international_travel.txt"),
    ("境外就医理赔需要哪些材料",         "10_international_travel.txt"),
    ("归国后差旅备用金要退还吗",         "10_international_travel.txt"),
    ("护照有效期有什么要求",            "10_international_travel.txt"),

    # ---- 11 景点指南 ----
    ("北京有哪些必去景点",              "11_city_attractions.txt"),
    ("成都大熊猫基地几点去最好",         "11_city_attractions.txt"),
    ("西安兵马俑怎么安排",              "11_city_attractions.txt"),
    ("杭州西湖半天怎么玩",              "11_city_attractions.txt"),

    # ---- 12 特殊时期政策 ----
    ("广交会期间要提前多久预订",         "12_seasonal_policies.txt"),
    ("旺季住宿标准能上浮多少",           "12_seasonal_policies.txt"),
    ("出差期间被隔离费用由谁承担",        "12_seasonal_policies.txt"),
    ("紧急出差可以后补审批吗",           "12_seasonal_policies.txt"),
]

# 负例：知识库里不该有答案的查询，用于标定 min_bm25_score
NEGATIVE_CASES = [
    "量子计算机的原理是什么",
    "如何做红烧肉",
    "明天股市会涨吗",
    "帮我写一首诗",
    "推荐一部好看的电影",
    "今天天气怎么样",          # 这属于 query-info 技能，不是知识库
]


def load_agent():
    """动态加载技能插件里的 RAGKnowledgeAgent 并初始化（会建 BM25 索引）。"""
    spec = importlib.util.spec_from_file_location(
        "rag_eval_agent",
        ROOT / ".claude" / "skills" / "ask-question" / "script" / "agent.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["rag_eval_agent"] = module
    spec.loader.exec_module(module)
    return module.RAGKnowledgeAgent(name="RAGKnowledgeAgent", model=None)


def source_of(hit: dict, corpus: dict) -> str:
    """从检索结果里取出源文档名。"""
    meta = hit.get("metadata") or {}
    if meta.get("parent_doc"):
        return meta["parent_doc"]
    # 仅 BM25 命中的条目：metadata 来自 corpus，兜底再查一次
    rec = corpus.get(hit.get("id"))
    if rec:
        return (rec.get("metadata") or {}).get("parent_doc", "")
    return ""


# --------------------------------------------------------------------------
# 三种检索模式
# --------------------------------------------------------------------------

def search_vector(agent, query, k):
    """纯向量：向量检索 + 相似度阈值过滤（改造前的生产行为）。"""
    hits = agent._vector_search(query, k)
    thr = agent.similarity_threshold
    if thr is not None:
        hits = [h for h in hits if h["distance"] >= thr]
    return hits


def search_bm25(agent, query, k):
    """纯 BM25：只用关键词路。"""
    out = []
    for cid, score in agent._bm25.search(query, k):
        rec = agent._corpus.get(cid)
        if not rec:
            continue
        out.append({
            "id": cid,
            "content": rec["content"],
            "metadata": rec["metadata"],
            "distance": None,
            "bm25_score": score,
        })
    return out


def search_hybrid(agent, query, k):
    """混合：向量 + BM25 → RRF 融合（复用 Agent 的生产实现）。"""
    saved = agent._hybrid_cfg.get("final_top_k")
    agent._hybrid_cfg["final_top_k"] = k
    try:
        return agent.search_knowledge(query)
    finally:
        if saved is None:
            agent._hybrid_cfg.pop("final_top_k", None)
        else:
            agent._hybrid_cfg["final_top_k"] = saved


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------

def expected_set(expect) -> set:
    """把期望文档统一成集合，支持单个字符串或元组（多个都算对）。"""
    if isinstance(expect, str):
        return {expect}
    return set(expect)


def fmt_expect(expect) -> str:
    """报告里显示期望文档。"""
    if isinstance(expect, str):
        return f"`{expect}`"
    return " 或 ".join(f"`{e}`" for e in expect)


def evaluate(agent, fn, k, cases):
    """返回 (hit1, hitk, mrr, per_case)。"""
    hit1 = hitk = 0
    rr_sum = 0.0
    per_case = []

    for query, expect in cases:
        exp = expected_set(expect)
        try:
            hits = fn(agent, query, k)
        except Exception as e:  # noqa: BLE001
            per_case.append({"query": query, "expect": expect, "got": [], "rank": None,
                             "error": f"{type(e).__name__}: {e}"})
            continue

        sources = [source_of(h, agent._corpus) for h in hits]
        rank = None
        for i, s in enumerate(sources, start=1):
            if s in exp:
                rank = i
                break
        if rank == 1:
            hit1 += 1
        if rank is not None:
            hitk += 1
            rr_sum += 1.0 / rank
        per_case.append({"query": query, "expect": expect, "got": sources, "rank": rank})

    n = len(cases)
    mrr = rr_sum / n if n else 0.0
    return hit1, hitk, mrr, per_case


def negative_scores(agent, k):
    """负例的最高分：分别看向量余弦与 BM25 分数，用于标定阈值。"""
    rows = []
    for q in NEGATIVE_CASES:
        try:
            v = agent._vector_search(q, k)
            v_top = max((h["distance"] for h in v), default=0.0)
        except Exception:
            v_top = 0.0
        try:
            b = agent._bm25.search(q, k)
            b_top = b[0][1] if b else 0.0
        except Exception:
            b_top = 0.0
        rows.append({"query": q, "vector_top": v_top, "bm25_top": b_top})
    return rows


def main():
    ap = argparse.ArgumentParser(description="检索效果评测")
    ap.add_argument("--k", type=int, default=3, help="Top-K（默认 3，与生产 final_top_k 一致）")
    ap.add_argument("--no-save", action="store_true", help="不保存报告文件")
    args = ap.parse_args()
    k = args.k

    print("=" * 78)
    print("检索效果评测：纯向量 vs 纯 BM25 vs 混合（RRF）")
    print("=" * 78)

    print("\n[1/4] 初始化 RAG Agent 并构建 BM25 索引...")
    t0 = time.time()
    agent = load_agent()
    if not agent.initialized:
        print("  ✗ Agent 未初始化：请先运行 "
              "python .claude/skills/ask-question/script/init_knowledge_base.py")
        return 1
    if agent._bm25 is None:
        print("  ✗ BM25 索引未构建（hybrid.enabled 是否为 false？）")
        return 1
    st = agent._bm25.stats()
    print(f"  ✓ {len(agent._corpus)} 个 chunk，词表 {st['vocab']}，"
          f"分词={'jieba' if st['jieba'] else 'bigram'}，耗时 {time.time() - t0:.1f}s")
    print(f"  相似度阈值={agent.similarity_threshold}  "
          f"hybrid 配置={json.dumps(agent._hybrid_cfg, ensure_ascii=False)}")

    print(f"\n[2/4] 评测 {len(CASES)} 条标注 query（Top-{k}）...")
    modes = [
        ("纯向量", search_vector),
        ("纯 BM25", search_bm25),
        ("混合(RRF)", search_hybrid),
    ]
    results = {}
    for name, fn in modes:
        hit1, hitk, mrr, per_case = evaluate(agent, fn, k, CASES)
        results[name] = {"hit1": hit1, "hitk": hitk, "mrr": mrr, "per_case": per_case}
        print(f"  {name:<12} Hit@1={hit1:>2}/{len(CASES)}  "
              f"Hit@{k}={hitk:>2}/{len(CASES)}  MRR={mrr:.3f}")

    print(f"\n[3/4] 负例分析（{len(NEGATIVE_CASES)} 条不该命中的查询）...")
    negs = negative_scores(agent, k)
    for r in negs:
        print(f"  {r['query']:<20} 向量最高余弦={r['vector_top']:.3f}  "
              f"BM25 最高分={r['bm25_top']:.2f}")

    # 正例的 BM25 分数区间（用于找分界点）
    pos_bm25 = []
    for query, _ in CASES:
        hits = agent._bm25.search(query, k)
        if hits:
            pos_bm25.append(hits[0][1])
    neg_bm25 = [r["bm25_top"] for r in negs if r["bm25_top"] > 0]

    print(f"\n[4/4] 阈值标定参考")

    # ---- 向量路：阈值能否把负例挡在门外 ----
    neg_vec = [r["vector_top"] for r in negs]
    print(f"\n  【向量路】阈值 = {agent.similarity_threshold}")
    if agent.similarity_threshold is not None and neg_vec:
        over = [r for r in negs if r["vector_top"] >= agent.similarity_threshold]
        print(f"    负例最高余弦 = {max(neg_vec):.3f}")
        if over:
            print(f"    ⚠ 有 {len(over)} 条负例 >= 阈值，会被向量路放行：")
            for r in over:
                print(f"        {r['query']}  ({r['vector_top']:.3f})")
        else:
            print(f"    ✓ {len(negs)} 条负例全部低于阈值，向量路的防幻觉过滤有效")

    # ---- BM25 路：绝对分数阈值能否分开 ----
    print(f"\n  【BM25 路】当前 min_bm25_score = {agent._hybrid_cfg.get('min_bm25_score')}")
    if pos_bm25 and neg_bm25:
        print(f"    正例 BM25 最高分区间: {min(pos_bm25):.2f} ~ {max(pos_bm25):.2f}")
        print(f"    负例 BM25 最高分区间: {min(neg_bm25):.2f} ~ {max(neg_bm25):.2f}")
        lo, hi = max(neg_bm25), min(pos_bm25)
        if hi > lo:
            print(f"    → 可分，建议阈值取 ({lo:.2f}, {hi:.2f}) 中点 ≈ {(lo + hi) / 2:.2f}")
        else:
            print(f"    ✗ 区间重叠（负例最高 {lo:.2f} >= 正例最低 {hi:.2f}），"
                  f"绝对分数阈值无法分开")
            print(f"      → 根因：BM25 的 IDF 用了 ln(1+...) 平滑恒为正，"
                  f"且小语料下稀有词命中会拿高分")
            print(f"      → 建议改用「至少命中 N 个查询实词」的准入方式，"
                  f"或依赖上游意图识别把无关问题拦在 RAG 之前")
    else:
        print("    数据不足，无法标定")

    # ---- 报告 ----
    print("\n" + "=" * 78)
    print(f"结果汇总（Top-{k}，{len(CASES)} 条标注 query）")
    print("=" * 78)
    print(f"{'检索模式':<14}{'Hit@1':<10}{f'Hit@{k}':<10}{'MRR'}")
    print("-" * 78)
    for name, _ in modes:
        r = results[name]
        print(f"{name:<14}{r['hit1']}/{len(CASES):<8}"
              f"{r['hitk']}/{len(CASES):<8}{r['mrr']:.3f}")

    base = results["纯向量"]
    print(f"\n混合检索相对纯向量的提升：")
    print(f"  Hit@{k}: {base['hitk']} → {results['混合(RRF)']['hitk']}"
          f"  ({results['混合(RRF)']['hitk'] - base['hitk']:+d})")
    print(f"  MRR:    {base['mrr']:.3f} → {results['混合(RRF)']['mrr']:.3f}"
          f"  ({results['混合(RRF)']['mrr'] - base['mrr']:+.3f})")

    # 混合检索独有的收益：纯向量漏掉、混合找回的
    vec_miss = {c["query"] for c in base["per_case"] if c["rank"] is None}
    hyb_hit = {c["query"] for c in results["混合(RRF)"]["per_case"] if c["rank"] is not None}
    gained = vec_miss & hyb_hit
    if gained:
        print(f"\n混合检索比纯向量多召回的 {len(gained)} 条（这就是关键词路的贡献）：")
        for q in sorted(gained):
            row = next(c for c in results["混合(RRF)"]["per_case"] if c["query"] == q)
            print(f"  + {q:<22} → 命中排第 {row['rank']}")

    # 纯向量有、混合没有的（回归风险）
    vec_hit = {c["query"] for c in base["per_case"] if c["rank"] is not None}
    hyb_miss = {c["query"] for c in results["混合(RRF)"]["per_case"] if c["rank"] is None}
    lost = vec_hit & hyb_miss
    if lost:
        print(f"\n⚠ 混合检索丢失的 {len(lost)} 条（需检查 RRF 权重或阈值）：")
        for q in sorted(lost):
            print(f"  - {q}")

    # 未命中明细
    print(f"\n未命中明细（混合检索）：")
    misses = [c for c in results["混合(RRF)"]["per_case"] if c["rank"] is None]
    if not misses:
        print("  无")
    for c in misses:
        print(f"  ✗ {c['query']:<22} 期望 {fmt_expect(c['expect'])}")
        print(f"      实际 Top{k}={c['got']}")
        if c.get("error"):
            print(f"      错误: {c['error']}")

    # ---- 保存 ----
    if not args.no_save:
        out_dir = ROOT / "tests" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"retrieval_eval_{datetime.now():%Y%m%d_%H%M%S}.md"
        lines = [
            "# 检索效果评测报告",
            "",
            f"**评测时间**: {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"**Top-K**: {k}　**标注 query 数**: {len(CASES)}　"
            f"**负例数**: {len(NEGATIVE_CASES)}",
            "",
            "## 结果",
            "",
            f"| 检索模式 | Hit@1 | Hit@{k} | MRR |",
            "|---|---|---|---|",
        ]
        for name, _ in modes:
            r = results[name]
            lines.append(f"| {name} | {r['hit1']}/{len(CASES)} | "
                         f"{r['hitk']}/{len(CASES)} | {r['mrr']:.3f} |")
        lines += ["", "## 未命中明细（混合检索）", ""]
        if not misses:
            lines.append("无")
        for c in misses:
            lines.append(f"- **{c['query']}** → 期望 {fmt_expect(c['expect'])}，"
                         f"实际 {c['got']}")
        lines += ["", "## min_bm25_score 标定", ""]
        if pos_bm25 and neg_bm25:
            lines.append(f"- 正例 BM25 最高分: {min(pos_bm25):.2f} ~ {max(pos_bm25):.2f}")
            lines.append(f"- 负例 BM25 最高分: {min(neg_bm25):.2f} ~ {max(neg_bm25):.2f}")
            lo, hi = max(neg_bm25), min(pos_bm25)
            lines.append(f"- 可分区间: {'({:.2f}, {:.2f})'.format(lo, hi) if hi > lo else '重叠，无法用绝对阈值分开'}")
        lines += ["", "## 检索明细", "",
                  f"| query | 期望文档 | 纯向量 | 纯BM25 | 混合 |", "|---|---|---|---|---|"]
        for i, (query, expect) in enumerate(CASES):
            cells = []
            for name, _ in modes:
                c = results[name]["per_case"][i]
                cells.append(f"#{c['rank']}" if c["rank"] else "✗")
            lines.append(f"| {query} | {fmt_expect(expect)} | {' | '.join(cells)} |")

        out_file.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n报告已保存: {out_file.relative_to(ROOT)}")

    agent.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
