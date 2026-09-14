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

# ⚠️ RAG_CONFIG 必须在**模块级**导入。
# 之前它只在 build_reranker() 内部用 `from config import RAG_CONFIG` 局部导入，
# 而 main() 里又引用了它 → NameError: name 'RAG_CONFIG' is not defined，
# 评测跑完 2/4、3/4 之后崩溃，汇总表和报告文件全都没生成。
from config import RAG_CONFIG
from utils.reranker import CrossEncoderReranker, resolve_model_path

# 评测用的精排器，由 main() 装载后注入（evaluate() 的回调签名固定，用模块级变量）
_RERANKER = None

# 候选池宽度取自 config，保证评测与生产跑的是同一条流水线
# （否则会出现「评测用 10 条候选、生产用 3 条」的不可复现数字）
RRF_CANDIDATES = int((RAG_CONFIG.get("hybrid") or {}).get("rrf_candidates", 10))

# --------------------------------------------------------------------------
# 标注集：query -> 期望命中的源文档
# 构造原则：优先挑**该文档独有**的内容，减少与其他文档的歧义
# --------------------------------------------------------------------------
CASES = [
    # ---- 01 差旅标准与规定 ----
    ("超标住宿费由谁审批",              "01_travel_standards.txt"),
    ("西藏新疆出差住宿标准能上浮多少",    "01_travel_standards.txt"),
    ("三线及以下城市的住宿标准",         "01_travel_standards.txt"),
    # 01 行31「国际长途航线（4小时以上）：可预订高端经济舱」
    # 与 10 行46「航程4小时以上：可预订高端经济舱」写的**是同一条规则**，两篇都算对
    ("国际长途航班可以订什么舱位",        ("01_travel_standards.txt", "10_international_travel.txt")),

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
    # 三篇都**直接回答了**该问题（逐行核对原文，非按检索结果倒推）：
    #   12 行115/116「可先出差后补审批，需在出发前告知直属主管；返回后3个工作日内完成」
    #   04 行6      「紧急出差可以特事特办，返回后3日内补齐审批手续」
    #   01 行19     「紧急情况下可先出差后补审批，返回后3日内完成」
    ("紧急出差可以后补审批吗",           ("12_seasonal_policies.txt", "04_faq.txt", "01_travel_standards.txt")),
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


def positive_cosine_stats(agent, kd):
    """
    正例里「**期望文档自己**的最高余弦」—— 这是向量阈值**不能切掉**的下限。

    与精排闸门同一口径：阈值要保住的是期望文档最好的那个 chunk，而不是召回列表的
    top-1 —— top-1 往往是错文档，它对阈值不构成约束。

    注意 `_vector_search` 本身不过阈值，所以这里拿到的是原始余弦。
    只统计进入向量路候选池（top_k_dense）的部分，与真实流水线一致。
    """
    rows = []
    for query, expect in CASES:
        exp = expected_set(expect)
        try:
            hits = agent._vector_search(query, kd)
        except Exception:
            hits = []
        own = [h["distance"] for h in hits if source_of(h, agent._corpus) in exp]
        rows.append({
            "query": query,
            "own_max": max(own) if own else None,
            "top1": max((h["distance"] for h in hits), default=0.0),
        })
    return rows


def build_reranker():
    """按 RAG_CONFIG['rerank'] 构建精排器；不可用则返回 None。"""
    cfg = RAG_CONFIG.get("rerank") or {}
    model = cfg.get("model")
    if resolve_model_path(model) is None and not cfg.get("allow_download"):
        print(f"  ✗ 精排模型不存在：{model}")
        print("    先下载：venv\\Scripts\\python.exe scripts\\download_reranker.py")
        return None

    rr = CrossEncoderReranker(
        model_name_or_path=model,
        device=cfg.get("device"),
        batch_size=int(cfg.get("batch_size", 16)),
        max_length=int(cfg.get("max_length", 512)),
        allow_download=bool(cfg.get("allow_download", False)),
    )
    if not rr.available:
        print("  ✗ 精排模型装载失败（详见日志）")
        return None
    return rr


def _with_rerank(agent, k, mode="rerank", threshold=None):
    """上下文管理器：临时给 agent 挂上精排器，退出时恢复，保证其他三列基线不被污染。"""
    import contextlib

    @contextlib.contextmanager
    def _cm():
        saved_reranker = agent._reranker
        saved_cfg = agent._rerank_cfg
        saved_final = agent._hybrid_cfg.get("final_top_k")
        saved_cand = agent._hybrid_cfg.get("rrf_candidates")
        agent._reranker = _RERANKER
        agent._rerank_cfg = {"mode": mode, "score_threshold": threshold}
        agent._hybrid_cfg["final_top_k"] = k
        agent._hybrid_cfg["rrf_candidates"] = max(k, RRF_CANDIDATES)
        try:
            yield
        finally:
            agent._reranker = saved_reranker
            agent._rerank_cfg = saved_cfg
            for key, val in (("final_top_k", saved_final), ("rrf_candidates", saved_cand)):
                if val is None:
                    agent._hybrid_cfg.pop(key, None)
                else:
                    agent._hybrid_cfg[key] = val
    return _cm()


def search_hybrid_rerank(agent, query, k):
    """混合 + 精排**重排**：RRF 保宽候选池 → 精排排序 → 取 k（A/B 对比用）。"""
    with _with_rerank(agent, k, mode="rerank"):
        return agent.search_knowledge(query)


def search_hybrid_gate(agent, query, k, threshold):
    """混合 + 精排**闸门**：排序仍用 RRF，只用精排分数丢掉低相关片段（推荐用法）。"""
    with _with_rerank(agent, k, mode="filter", threshold=threshold):
        return agent.search_knowledge(query)


def main():
    ap = argparse.ArgumentParser(description="检索效果评测")
    ap.add_argument("--k", type=int, default=3, help="Top-K（默认 3，与生产 final_top_k 一致）")
    ap.add_argument("--no-save", action="store_true", help="不保存报告文件")
    ap.add_argument("--max-per-doc", type=int, default=None,
                    help="覆盖 hybrid.max_per_doc（0=不去重），用于 A/B 对比去重效果")
    ap.add_argument("--sweep", nargs="?", const="auto", default=None,
                    help="扫描 similarity_threshold：裸用 --sweep 会按实测可分区间自动取点；"
                         "也可显式给逗号列表（注意某些 shell 下长列表会传参失败）")
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

    # --max-per-doc 覆盖要在打印 hybrid 配置之前生效，否则打印的是旧值
    if args.max_per_doc is not None:
        agent._hybrid_cfg["max_per_doc"] = args.max_per_doc
        print(f"  ⚙ --max-per-doc 覆盖为 {args.max_per_doc}（0 = 不去重）")

    st = agent._bm25.stats()
    print(f"  ✓ {len(agent._corpus)} 个 chunk，词表 {st['vocab']}，"
          f"分词={'jieba' if st['jieba'] else 'bigram'}，耗时 {time.time() - t0:.1f}s")
    print(f"  相似度阈值={agent.similarity_threshold}  "
          f"hybrid 配置={json.dumps(agent._hybrid_cfg, ensure_ascii=False)}")
    print(f"  候选池 rrf_candidates={RRF_CANDIDATES}（评测与生产同一条流水线）")

    # 放在 agent 初始化之后，避免 [1/4] 的输出被 [1.5/4] 的标题截断（顺序错乱）
    print("\n[1.5/4] 装载 Rerank 精排模型...")
    global _RERANKER
    _RERANKER = build_reranker()
    if _RERANKER is not None:
        rst = _RERANKER.stats()
        print(f"  ✓ 精排就绪：{rst['resolved_path'] or rst['model']}"
              f"  device={rst['device'] or 'auto'}  max_length={rst['max_length']}")
        print(f"  生产配置 rerank={json.dumps(RAG_CONFIG.get('rerank') or {}, ensure_ascii=False)}")
    else:
        print("  ⚠ 精排不可用，本次只跑三路对比（这是降级，不是失败）")

    print(f"\n[2/4] 评测 {len(CASES)} 条标注 query（Top-{k}）...")
    modes = [
        ("纯向量", search_vector),
        ("纯 BM25", search_bm25),
        ("混合(RRF)", search_hybrid),
    ]
    if _RERANKER is not None:
        modes.append(("混合+Rerank", search_hybrid_rerank))
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
    rerank_calib = None      # 供报告段落使用

    # ---- 向量路：阈值能否既挡住负例、又不切掉正例 ----
    neg_vec = [r["vector_top"] for r in negs]
    kd_vec = int(agent._hybrid_cfg.get("top_k_dense", 10))
    pos_vec = positive_cosine_stats(agent, kd_vec)
    own_vec = [r["own_max"] for r in pos_vec if r["own_max"] is not None]
    cur_thr = agent.similarity_threshold

    print(f"\n  【向量路】当前 similarity_threshold = {cur_thr}")
    if neg_vec and own_vec:
        print(f"    正例·期望文档最高余弦: {min(own_vec):.3f} ~ {max(own_vec):.3f}"
              f"　（{len(own_vec)}/{len(CASES)} 条能取到，其余未进 top_k_dense={kd_vec}）")
        print(f"    负例·Top-1 最高余弦  : {min(neg_vec):.3f} ~ {max(neg_vec):.3f}")
        lo, hi = max(neg_vec), min(own_vec)
        if hi > lo:
            rec_v = lo + (hi - lo) * 0.5
            print(f"    ✓ 可分！可分区间 ({lo:.3f}, {hi:.3f})，建议阈值 ≈ {rec_v:.3f}")
        else:
            print(f"    ✗ 区间重叠（负例最高 {lo:.3f} >= 正例最低 {hi:.3f}），"
                  f"单一余弦阈值无法两头兼顾")
            print("      → 余弦阈值只能二选一：保正例召回 还是 挡负例。"
                  "挡负例更适合交给精排闸门（见下方【精排路】）")
        if cur_thr is not None:
            margin = float(cur_thr) - max(neg_vec)
            flag = "　⚠ 余量过小，防幻觉过滤已接近失效" if margin < 0.03 else ""
            print(f"    当前阈值 {cur_thr} 距负例上限 {max(neg_vec):.3f} 的余量 = "
                  f"{margin:+.3f}{flag}")
            lost = [r["query"] for r in pos_vec
                    if r["own_max"] is not None and r["own_max"] < float(cur_thr)]
            if lost:
                print(f"    ⚠ 有 {len(lost)} 条正例的期望文档最高余弦 < 当前阈值，会被切掉：")
                for q in lost[:8]:
                    print(f"        {q}")
    else:
        print("    数据不足，无法标定")

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

    # ---- 精排路：能否用精排分数分开正负例（替代失效的 BM25 绝对阈值）----
    if _RERANKER is not None:
        rcfg = RAG_CONFIG.get("rerank") or {}
        print(f"\n  【精排路】mode = {rcfg.get('mode')}　"
              f"score_threshold = {rcfg.get('score_threshold')}")
        print("    口径：正例看**期望文档自己的最高分**（闸门要保住的就是它）；"
              "负例看召回池最高分")

        def _pools(q):
            """一次召回同时取出 RRF 顺序的候选 + 各候选精排分（按 id 关联）。"""
            rrf = search_hybrid(agent, q, RRF_CANDIDATES)
            rr = search_hybrid_rerank(agent, q, RRF_CANDIDATES)
            if not rr or _RERANKER.score_key not in rr[0]:
                return rrf, None
            return rrf, {h.get("id"): h[_RERANKER.score_key] for h in rr}

        def _sim_gate(rrf, sid, thr):
            """复刻 filter 模式语义：保持 RRF 顺序 → 按分数过滤 → 截断到 k。"""
            if sid is None:
                return None
            return [h for h in rrf if sid.get(h.get("id"), -1.0) >= thr][:k]

        # 单次遍历取齐；后面的统计与闸门验证全部复用，不再重复跑推理
        pos_pool = [(q, expect) + _pools(q) for q, expect in CASES]
        neg_pool = [(q,) + _pools(q) for q in NEGATIVE_CASES]

        pos_exp = []
        for _q, expect, rrf, sid in pos_pool:
            if sid is None:
                continue
            exp = expected_set(expect)
            own = [sid[h.get("id")] for h in rrf
                   if h.get("id") in sid and source_of(h, agent._corpus) in exp]
            if own:
                pos_exp.append(max(own))

        neg_top = [max(sid.values()) for _q, _rrf, sid in neg_pool if sid]

        if pos_exp and neg_top:
            print(f"    正例·期望文档最高分: {min(pos_exp):.3f} ~ {max(pos_exp):.3f}"
                  f"　（{len(pos_exp)}/{len(CASES)} 条能取到）")
            print(f"    负例·召回池最高分  : {min(neg_top):.3f} ~ {max(neg_top):.3f}")
            lo, hi = max(neg_top), min(pos_exp)
            if hi > lo:
                rec = lo + (hi - lo) * 0.25        # 保守取法：先保正例召回
                print(f"    ✓ 可分！建议 score_threshold ≈ {rec:.3f}"
                      f"（可分区间 ({lo:.3f}, {hi:.3f})，中点 {(lo + hi) / 2:.3f}，"
                      f"保守取 {rec:.3f}）")

                kept_pos = sum(
                    1 for _q, expect, rrf, sid in pos_pool
                    if any(source_of(h, agent._corpus) in expected_set(expect)
                           for h in (_sim_gate(rrf, sid, rec) or []))
                )
                blocked = sum(1 for _q, rrf, sid in neg_pool
                              if not (_sim_gate(rrf, sid, rec) or []))
                base_blocked = sum(1 for q in NEGATIVE_CASES
                                   if not search_hybrid(agent, q, k))
                print(f"    → 闸门实测（threshold={rec:.3f}）："
                      f"正例保住 {kept_pos}/{len(CASES)}，"
                      f"负例拦掉 {blocked}/{len(NEGATIVE_CASES)}"
                      f"（不开精排时已拦 {base_blocked}/{len(NEGATIVE_CASES)}）")

                rerank_calib = {
                    "mode": rcfg.get("mode"),
                    "threshold_cfg": rcfg.get("score_threshold"),
                    "recommended": rec,
                    "lo": lo, "hi": hi,
                    "pos_min": min(pos_exp), "pos_max": max(pos_exp),
                    "neg_min": min(neg_top), "neg_max": max(neg_top),
                    "kept_pos": kept_pos, "blocked": blocked,
                    "base_blocked": base_blocked,
                }
            else:
                print(f"    ✗ 仍重叠（负例最高 {lo:.3f} >= 正例最低 {hi:.3f}）")
                print("      → 说明 bge-reranker-base 对本语料的判别力不够，"
                      "考虑升级 bge-reranker-v2-m3（但 CPU 延迟会显著上升）")
        else:
            print("    数据不足，无法标定")

    # ---- similarity_threshold 扫描（放在标定之后，才能用上刚测出的可分区间）----
    # 注意：必须放在【向量路】标定之后 —— 这里用到 kd_vec / own_vec / neg_vec / cur_thr，
    # 提前到 [2/4] 之后会让它们成为"尚未赋值就被闭包引用"的 free variable（NameError）。
    if args.sweep:
        if str(args.sweep).strip().lower() == "auto":
            cands = []
            if own_vec and neg_vec:
                lo_v, hi_v = max(neg_vec), min(own_vec)
                if hi_v > lo_v:
                    # 在实测可分区间上均分 5 点
                    cands = [round(lo_v + (hi_v - lo_v) * f, 4)
                             for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
            if not cands:
                cands = [0.35, 0.40, 0.45, 0.50, 0.55]
            if cur_thr is not None:
                cands.append(float(cur_thr))
            cands = sorted(set(cands))
            print(f"\n  （--sweep auto：按可分区间自动取点）")
        else:
            try:
                cands = [float(x) for x in str(args.sweep).split(",") if x.strip()]
            except ValueError:
                print(f"\n  ✗ --sweep 解析失败：{args.sweep}"
                      f"（裸用 --sweep 即可自动取点）")
                cands = []

        if cands:
            print(f"\n[4.5/4] similarity_threshold 扫描"
                  f"（混合检索，max_per_doc={agent._hybrid_cfg.get('max_per_doc')}）...")
            # 原始向量检索结果与阈值无关，只取一次复用（否则每个候选值都要重跑一遍）
            raw_vec = {q: agent._vector_search(q, kd_vec) for q, _ in CASES}
            saved_thr = agent.similarity_threshold
            print(f"  {'阈值':<10}{'Hit@1':<12}{f'Hit@{k}':<12}{'MRR':<10}{'向量路平均条数'}")
            print("  " + "-" * 58)
            try:
                for t in cands:
                    agent.similarity_threshold = t
                    n_vec = sum(
                        len([d for d in raw_vec[q] if d["distance"] >= t])
                        for q, _ in CASES
                    ) / float(len(CASES))
                    h1, hk, m, _ = evaluate(agent, search_hybrid, k, CASES)
                    mark = "  ← 当前" if cur_thr is not None and abs(t - float(cur_thr)) < 1e-9 else ""
                    print(f"  {t:<10.4f}{h1}/{len(CASES):<9}{hk}/{len(CASES):<9}"
                          f"{m:<10.3f}{n_vec:.1f}{mark}")
            finally:
                agent.similarity_threshold = saved_thr
            print(f"  生产阈值仍为 {saved_thr}（扫描只改内存，未写回 config.py）")
            print("  判定：挑 Hit@3 最高、其次 MRR 最高的那一行写进 config.py")

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

        lines += ["", "## Rerank 精排标定（score_threshold 的来源）", ""]
        if rerank_calib:
            c = rerank_calib
            lines += [
                f"- 生产配置 mode: `{c['mode']}`　score_threshold: `{c['threshold_cfg']}`",
                f"- 正例·期望文档最高分: {c['pos_min']:.3f} ~ {c['pos_max']:.3f}",
                f"- 负例·召回池最高分: {c['neg_min']:.3f} ~ {c['neg_max']:.3f}",
                f"- 可分区间: ({c['lo']:.3f}, {c['hi']:.3f})"
                f"　→ 建议 `score_threshold = {c['recommended']:.3f}`（保守取法，先保正例召回）",
                f"- 闸门实测（threshold={c['recommended']:.3f}）: "
                f"正例保住 {c['kept_pos']}/{len(CASES)}，"
                f"负例拦掉 {c['blocked']}/{len(NEGATIVE_CASES)}"
                f"（不开精排时已拦 {c['base_blocked']}/{len(NEGATIVE_CASES)}）",
            ]
        else:
            lines.append("精排未启用或分数无法分开正负例，本次未产出建议阈值。")

        lines += ["", "## 检索明细", "",
                  "| query | 期望文档 | " + " | ".join(n for n, _ in modes) + " |",
                  "|---" * (2 + len(modes)) + "|"]
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
