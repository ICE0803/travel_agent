#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
搜索后端接入自检

对 config.SEARCH_CONFIG 里配置的每个后端逐一实测，直接告诉你哪个能用、卡在哪。
覆盖：Tavily / DDGS

用法：
  venv\\Scripts\\python.exe scripts\\check_search_api.py            # 全部后端
  venv\\Scripts\\python.exe scripts\\check_search_api.py tavily     # 只测 Tavily
  venv\\Scripts\\python.exe scripts\\check_search_api.py "" "杭州天气"
"""
import json
import os
import socket
import sys
from urllib.parse import urlparse

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    import httpx
except ImportError:
    print("✗ httpx 未安装：pip install httpx")
    sys.exit(1)

try:
    from config import SEARCH_CONFIG
except ImportError:
    print("✗ 无法从 config.py 读取 SEARCH_CONFIG")
    sys.exit(1)

OK, BAD, WARN = "✓", "✗", "!"


def mask(v: str) -> str:
    if not v:
        return "(空)"
    if len(v) <= 8:
        return f"{v[:2]}***"
    return f"{v[:4]}...{v[-4:]}"


def cred(cfg: dict, cfg_key: str, env_name: str) -> tuple:
    """返回 (值, 来源)。环境变量优先。"""
    env_val = os.environ.get(env_name, "").strip()
    cfg_val = str(cfg.get(cfg_key) or "").strip()
    if env_val:
        return env_val, f"环境变量 {env_name}"
    if cfg_val:
        return cfg_val, "config.py"
    return "", "未配置"


def registry_hint(env_name: str) -> str:
    """
    典型坑：setx 设置成功了，但当前进程读不到 —— 因为终端继承自已运行的父进程环境。
    此时给出针对性提示，而不是让用户干瞪眼。
    """
    if os.name != "nt":
        return ""
    try:
        import winreg
    except ImportError:
        return ""
    found = None
    for root, label in ((winreg.HKEY_CURRENT_USER, "用户级"),
                        (winreg.HKEY_LOCAL_MACHINE, "系统级")):
        try:
            with winreg.OpenKey(root, "Environment") as k:
                val, _ = winreg.QueryValueEx(k, env_name)
                if str(val).strip():
                    found = (label, str(val).strip())
                    break
        except (FileNotFoundError, OSError):
            continue
    if not found:
        return ""
    label, val = found
    return (
        f"      ⚠ 检测到 {label}环境变量 {env_name} 其实已存在（{mask(val)}），\n"
        f"        但当前进程读不到。这是 Windows 环境变量传播问题：\n"
        f"        setx 只对【之后新启动】的进程生效；若终端继承自已运行的父进程\n"
        f"        （Windows Terminal / VS Code / 资源管理器未刷新），重开标签页甚至\n"
        f"        重开窗口都可能仍旧读不到。三个解决办法（任选）：\n"
        f"          1) 不用环境变量，直接填进 config.py（最稳，改一行即可）：\n"
        f"             把上面这个值粘到 config.py 的 SEARCH_CONFIG 对应字段里\n"
        f"             （config.py 已被 .gitignore 忽略，Key 不会进仓库）\n"
        f"          2) 完全退出 Windows Terminal / VS Code 等父进程再重开（不是新开标签页）\n"
        f"          3) 重启电脑"
    )


def check_tcp(host: str, timeout: float = 6.0):
    try:
        with socket.create_connection((host, 443), timeout=timeout):
            return True, ""
    except Exception as e:
        return False, str(e)


# ------------------------------------------------------------------- Tavily
def test_tavily(query: str):
    print(f"\n{'─' * 70}\n[Tavily]  api_key\n{'─' * 70}")
    cfg = SEARCH_CONFIG.get("tavily", {}) or {}
    key, key_src = cred(cfg, "api_key", "TAVILY_API_KEY")
    print(f"  api_key : {mask(key)}   来源: {key_src}")
    if not key:
        print(f"  {WARN} 未配置 → 跳过。")
        hint = registry_hint("TAVILY_API_KEY")
        if hint:
            print(hint)
        else:
            print(f"      配置方式（任选）：")
            print(f"          1) 填进 config.py 的 SEARCH_CONFIG['tavily']['api_key']（推荐，最稳）")
            print(f"          2) setx TAVILY_API_KEY \"tvly-你的Key\"，然后完全重开终端")
            print(f"      Key 获取：https://app.tavily.com")
        return False

    if not key.startswith("tvly-"):
        print(f"  {WARN} 提示：Tavily 的 Key 通常以 'tvly-' 开头，请确认没复制错。")

    endpoint = cfg.get("endpoint", "https://api.tavily.com/search")
    host = urlparse(endpoint).hostname or "api.tavily.com"
    ok, err = check_tcp(host)
    print(f"  网络    : {OK if ok else BAD} {host} {'' if ok else '— ' + err}")
    if not ok:
        print(f"  {BAD} 网络不通，无需继续。")
        return False

    body = {
        "query": query,
        "max_results": 3,
        "search_depth": cfg.get("search_depth", "basic"),
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }
    if cfg.get("topic"):
        body["topic"] = cfg["topic"]

    try:
        resp = httpx.post(
            endpoint,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body,
            timeout=float(cfg.get("timeout", 15.0)),
        )
    except Exception as e:
        print(f"  {BAD} 请求异常: {type(e).__name__}: {e}")
        return False

    print(f"  HTTP    : {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json() or {}
        results = data.get("results") or []
        print(f"  {OK} 可用，返回 {len(results)} 条，"
              f"耗时 {data.get('response_time', '?')}s")
        for r in results[:3]:
            print(f"      - {(r.get('title') or '')[:60]}")
            print(f"        {r.get('url', '')}")
        print(f"  {OK} 本次消耗 1 credit（basic）；免费额度 1000 credits/月")
        return True

    detail = (resp.text or "")[:300]
    try:
        payload = resp.json() or {}
        detail = payload.get("detail") or payload.get("error") or detail
        if isinstance(detail, (dict, list)):
            detail = json.dumps(detail, ensure_ascii=False)
    except Exception:
        pass
    print(f"      detail: {detail}")

    if resp.status_code == 401:
        print(f"  {BAD} API Key 无效或已撤销 → 到 https://app.tavily.com 重新复制")
    elif resp.status_code == 429:
        print(f"  {WARN} 额度用尽或请求过频 → 免费 1000 credits/月，basic 检索 1 credit/次")
    elif resp.status_code == 432:
        print(f"  {BAD} 账户额度/权限不足")
    elif resp.status_code == 400:
        print(f"  {BAD} 请求参数有误")
    return False


# --------------------------------------------------------------------- DDGS
def test_ddgs(query: str, live: bool = True):
    print(f"\n{'─' * 70}\n[DDGS]  无需 Key（抓取公开页面）\n{'─' * 70}")
    try:
        from ddgs import DDGS
    except ImportError:
        print(f"  {BAD} 未安装 → pip install ddgs")
        return False
    if not live:
        print(f"  (跳过实测，加 --ddgs 开启)")
        return True

    cfg = SEARCH_CONFIG.get("ddgs", {}) or {}
    backends = cfg.get("backends") or ["bing", "duckduckgo", "auto"]
    working = []
    for b in backends + [x for x in ("bing", "yandex", "auto") if x not in backends]:
        try:
            raw = list(DDGS().text(query, max_results=3, safesearch=cfg.get("safesearch", "on"),
                                   region=cfg.get("region", "cn-zh"), backend=b))
            if raw:
                working.append(b)
                print(f"  {OK} {b:<12} → {len(raw)} 条")
            else:
                print(f"  {WARN} {b:<12} → 0 条")
        except Exception as e:
            print(f"  {BAD} {b:<12} → {type(e).__name__}")
    if working:
        print(f"\n  {OK} 可用后端：{', '.join(working)}")
        suggested = working if "auto" in working else working + ["auto"]
        print(f"      建议 config.py 的 SEARCH_CONFIG['ddgs']['backends'] 按此顺序配置：")
        print(f"      \"backends\": {json.dumps(suggested, ensure_ascii=False)}")
        return True
    print(f"\n  {BAD} 所有后端均不可用")
    return False


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    which = (args[0] if args else "all").lower()
    query = (args[1] if len(args) > 1 else "杭州天气")
    live_ddgs = "--ddgs" in sys.argv or which in ("all", "ddgs")

    print("=" * 70)
    print("搜索后端接入自检")
    print("=" * 70)
    print(f"查询词   : {query}")
    print(f"backend  : {SEARCH_CONFIG.get('backend')}")
    print(f"auto_order: {SEARCH_CONFIG.get('auto_order')}")

    results = {}
    if which in ("all", "tavily"):
        results["tavily"] = test_tavily(query)
    if which in ("all", "ddgs"):
        results["ddgs"] = test_ddgs(query, live=live_ddgs)

    print(f"\n{'=' * 70}\n汇总\n{'=' * 70}")
    for name, ok in results.items():
        print(f"  {OK if ok else BAD} {name}")

    usable = [n for n, ok in results.items() if ok]
    if usable:
        print(f"\n{OK} 可用后端：{', '.join(usable)}")
        # 只建议实测可用的后端，避免把没凭据/不通的后端又加回顺序里
        print(f"  建议 config.py 设：\"backend\": \"auto\"，")
        print(f"       \"auto_order\": {json.dumps(usable, ensure_ascii=False)}")
    else:
        print(f"\n{BAD} 没有可用后端。请检查 Key / 网络配置。")
    print(f"\n业务侧回归：venv\\Scripts\\python.exe tests\\test_search_backend.py")
    return 0 if usable else 1


if __name__ == "__main__":
    sys.exit(main())
