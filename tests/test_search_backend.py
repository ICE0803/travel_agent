#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
query-info 搜索后端切换测试（离线，不依赖网络 / 不需要真实 API Key）

验证 InformationQueryAgent 的多后端回退编排（Tavily → DDGS）：
  1. Tavily 可用时走 Tavily，并标注 engine="tavily"
  2. Tavily 不可用（未配置/超时/配额）时自动回退 DDGS
  3. 后端返回 0 条结果时也继续尝试下一个
  4. 全部后端失败时返回 query_success=False，并列出各后端失败原因
  5. backend 强制单通道时不回退
  6. auto_order 可配置、非法值容错
  7. 成功回退时用 results.fallback 暴露静默降级
  8. 可疑域名过滤 + 结果条数截断

运行：
  venv\\Scripts\\python.exe tests\\test_search_backend.py
"""
import asyncio
import importlib.util
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Windows 控制台默认 GBK，强制 UTF-8 以便输出中文/对勾
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

AGENT_PATH = PROJECT_ROOT / ".claude" / "skills" / "query-info" / "script" / "agent.py"

spec = importlib.util.spec_from_file_location("query_info_agent_under_test", AGENT_PATH)
agent_mod = importlib.util.module_from_spec(spec)
sys.modules["query_info_agent_under_test"] = agent_mod
spec.loader.exec_module(agent_mod)

InformationQueryAgent = agent_mod.InformationQueryAgent
SearchBackendUnavailable = agent_mod.SearchBackendUnavailable
SEARCH_CONFIG = agent_mod.SEARCH_CONFIG

TAVILY_HIT = [
    {"title": "北京天气", "snippet": "晴 25 度", "url": "https://weather.com.cn/beijing"},
    {"title": "低质站", "snippet": "垃圾内容", "url": "http://spam.tk/xx"},
    {"title": "北京旅游", "snippet": "攻略", "url": "https://mafengwo.cn/beijing"},
    {"title": "北京交通", "snippet": "地铁", "url": "https://bjsubway.com/"},
    {"title": "北京美食", "snippet": "烤鸭", "url": "https://meituan.com/bj"},
    {"title": "第六条", "snippet": "应被截断", "url": "https://example.com/6"},
]
DDGS_HIT = [{"title": "DDGS 结果", "snippet": "来自 bing", "url": "https://bing.com/result"}]

PASSED, FAILED = [], []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  [{detail}]" if detail and not cond else ""))


def make_agent(tavily=None, ddgs=None):
    """构造一个把两个后端都替换成桩的 agent（不发真实请求）。"""
    agent = InformationQueryAgent(name="InformationQueryAgent", model=None)

    def _tavily(q):
        if isinstance(tavily, Exception):
            raise tavily
        return list(tavily or [])

    def _ddgs(q):
        if isinstance(ddgs, Exception):
            raise ddgs
        return list(ddgs or [])

    async def _summary(q, results):
        return f"摘要({len(results)} 条)"

    agent._tavily_search_sync = _tavily
    agent._ddgs_search_sync = _ddgs
    agent._summarize_search_results = _summary
    return agent


def set_backend(value, auto_order=None):
    SEARCH_CONFIG["backend"] = value
    if auto_order is not None:
        SEARCH_CONFIG["auto_order"] = auto_order


async def main():
    # 保存用户配置，测试结束原样还原 —— 测试不应依赖也不应污染 config.py 的设置
    orig_backend = SEARCH_CONFIG.get("backend")
    orig_order = SEARCH_CONFIG.get("auto_order")

    print("=" * 72)
    print("query-info 搜索后端切换测试（离线）")
    print("=" * 72)
    print(f"（config.py 当前值：backend={orig_backend!r}, auto_order={orig_order!r}；"
          f"下列用例均显式指定顺序）")

    print("\n[1] Tavily 可用 → 走 Tavily，过滤可疑域名并截断到 max_results")
    set_backend("auto", ["tavily", "ddgs"])
    res = await make_agent(tavily=TAVILY_HIT)._web_search("北京天气")
    r = res["results"]
    check("query_success=True", res["query_success"] is True)
    check("engine=tavily", r.get("engine") == "tavily", str(r.get("engine")))
    check("可疑域名 .tk 被过滤", all(".tk" not in s["url"] for s in r["sources"]))
    check(f"截断到 {SEARCH_CONFIG['max_results']} 条",
          len(r["sources"]) == SEARCH_CONFIG["max_results"], f"实际 {len(r['sources'])}")
    check("sources 字段统一为 title/snippet/url",
          all(set(s) == {"title", "snippet", "url"} for s in r["sources"]))

    print("\n[2] Tavily 失败（未配置 Key）→ 自动回退 DDGS")
    set_backend("auto", ["tavily", "ddgs"])
    res = await make_agent(
        tavily=SearchBackendUnavailable("未配置 Tavily API Key"),
        ddgs=DDGS_HIT,
    )._web_search("北京天气")
    r = res["results"]
    check("query_success=True", res["query_success"] is True)
    check("engine=ddgs（已回退）", r.get("engine") == "ddgs", str(r.get("engine")))
    check("拿到 DDGS 结果", len(r["sources"]) == 1 and "bing.com" in r["sources"][0]["url"])

    print("\n[3] 后端返回 0 条结果 → 同样换下一个（不终止链路）")
    set_backend("auto", ["tavily", "ddgs"])
    res = await make_agent(tavily=[], ddgs=DDGS_HIT)._web_search("北京天气")
    check("engine=ddgs", res["results"].get("engine") == "ddgs")
    check("记录了 tavily 的 0 条结果",
          any("0 条结果" in a for a in res["results"].get("fallback", [])),
          str(res["results"].get("fallback")))

    print("\n[4] 两个后端都失败 → 报错并列出各自原因")
    set_backend("auto", ["tavily", "ddgs"])
    res = await make_agent(
        tavily=SearchBackendUnavailable("Tavily 返回 HTTP 401"),
        ddgs=SearchBackendUnavailable("DDGS 各后端均未返回结果"),
    )._web_search("北京天气")
    check("query_success=False", res["query_success"] is False)
    attempts = res["results"].get("attempts", [])
    check("attempts 含 2 个后端", len(attempts) == 2, str(attempts))
    check("原因含 tavily", any("tavily" in a for a in attempts))
    check("原因含 ddgs", any("ddgs" in a for a in attempts))

    print("\n[5] backend='tavily' → 只试 Tavily，不回退")
    set_backend("tavily")
    res = await make_agent(
        tavily=SearchBackendUnavailable("Tavily 返回 HTTP 429"),
        ddgs=DDGS_HIT,
    )._web_search("北京天气")
    check("未回退到 DDGS", res["query_success"] is False and "sources" not in res["results"])
    check("只有 1 条 attempts", len(res["results"].get("attempts", [])) == 1)

    print("\n[6] backend='ddgs' → 不调用 Tavily")
    set_backend("ddgs")
    called = {"tavily": False}
    agent = make_agent(ddgs=DDGS_HIT)
    _orig_t = agent._tavily_search_sync

    def _tracked(q):
        called["tavily"] = True
        return _orig_t(q)

    agent._tavily_search_sync = _tracked
    res = await agent._web_search("北京天气")
    check("Tavily 未被调用", called["tavily"] is False)
    check("engine=ddgs", res["results"].get("engine") == "ddgs")

    print("\n[7] 结果全被可疑域名过滤 → 明确报错而不是空 sources")
    set_backend("auto", ["tavily", "ddgs"])
    res = await make_agent(tavily=[{"title": "x", "snippet": "y", "url": "http://a.tk/1"}])._web_search("q")
    check("query_success=False", res["query_success"] is False)
    check("提示被过滤", "过滤" in res["results"].get("message", ""))

    print("\n[8] auto_order 可配置：把 ddgs 提到最前，就不再调 Tavily")
    set_backend("auto", ["ddgs", "tavily"])
    called = {"t": False}
    agent = make_agent(tavily=TAVILY_HIT, ddgs=DDGS_HIT)
    _orig_t = agent._tavily_search_sync
    agent._tavily_search_sync = lambda q: (called.__setitem__("t", True), _orig_t(q))[1]
    res = await agent._web_search("北京天气")
    check("engine=ddgs", res["results"].get("engine") == "ddgs")
    check("Tavily 未被调用", called["t"] is False)
    check("_search_order 生效", agent._search_order() == ["ddgs", "tavily"],
          str(agent._search_order()))

    print("\n[9] auto_order 写错后端名时，忽略非法项而不是整链失效")
    set_backend("auto", ["typo_backend", "tavily"])
    check("过滤掉非法项", make_agent()._search_order() == ["tavily"],
          str(make_agent()._search_order()))
    set_backend("auto", ["nonsense"])
    check("全非法时回落到默认顺序",
          make_agent()._search_order() == ["tavily", "ddgs"],
          str(make_agent()._search_order()))

    print("\n[10] 成功回退时，results.fallback 暴露前序失败（避免静默降级）")
    set_backend("auto", ["tavily", "ddgs"])
    res = await make_agent(
        tavily=SearchBackendUnavailable("Tavily 请求失败（网络不通或超时）"),
        ddgs=DDGS_HIT,
    )._web_search("北京天气")
    r = res["results"]
    check("engine=ddgs", r.get("engine") == "ddgs")
    check("results.fallback 存在", "fallback" in r, str(list(r.keys())))
    check("fallback 记录了 tavily 的失败原因",
          any("tavily" in a for a in r.get("fallback", [])), str(r.get("fallback")))

    print("\n[11] 首选后端直接成功时，不应有 fallback 字段")
    set_backend("auto", ["tavily", "ddgs"])
    res = await make_agent(tavily=TAVILY_HIT, ddgs=DDGS_HIT)._web_search("北京天气")
    check("无 fallback 字段", "fallback" not in res["results"], str(list(res["results"].keys())))

    print("\n[12] 真实 _tavily_search_sync：未配置 Key 时给出可操作提示（不发网络请求）")
    tcfg = SEARCH_CONFIG.setdefault("tavily", {})
    saved_key = tcfg.get("api_key")
    saved_env = os.environ.pop("TAVILY_API_KEY", None)
    try:
        tcfg["api_key"] = ""
        try:
            InformationQueryAgent(name="x", model=None)._tavily_search_sync("北京天气")
            check("未配置 Key 应抛 SearchBackendUnavailable", False, "没有抛出异常")
        except SearchBackendUnavailable as e:
            msg = str(e)
            check("抛 SearchBackendUnavailable", True)
            check("提示环境变量名 TAVILY_API_KEY", "TAVILY_API_KEY" in msg, msg)
            check("提示 config.py 位置", "config.py" in msg, msg)
        except Exception as e:  # noqa: BLE001
            check("异常类型应为 SearchBackendUnavailable", False, f"{type(e).__name__}: {e}")
    finally:
        tcfg["api_key"] = saved_key
        if saved_env is not None:
            os.environ["TAVILY_API_KEY"] = saved_env

    # 还原用户配置（测试不应污染 config.py 的设置）
    SEARCH_CONFIG["backend"] = orig_backend
    SEARCH_CONFIG["auto_order"] = orig_order

    print(f"通过 {len(PASSED)} / {len(PASSED) + len(FAILED)}")
    if FAILED:
        print("失败项：")
        for f in FAILED:
            print(f"  - {f}")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
