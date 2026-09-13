"""
信息查询智能体 - 真实检索版

支持：天气（wttr.in）、网络搜索（多后端可插拔：Tavily / DDGS）

数据来源：
- 天气：wttr.in（免费，无需 API Key）
- 搜索（按 config.SEARCH_CONFIG["auto_order"] 依次尝试，前一个失败自动换下一个）：
    1. Tavily Search API（需 API Key，免费 1000 credits/月）
    2. DDGS（抓取公开页面，无需 Key，需 pip install ddgs）
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from typing import Optional, Union, List, Dict, Any
import asyncio
import json
import logging
import re
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

try:
    from config import SEARCH_CONFIG
except ImportError:  # 兼容尚未更新 config.py 的旧部署
    SEARCH_CONFIG = {"backend": "ddgs", "tavily": {}, "ddgs": {}, "max_results": 5}

logger = logging.getLogger(__name__)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not installed. Install with: pip install httpx")

# 尝试导入 duckduckgo_search (旧包名) 或 ddgs (新包名)
try:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.warning("ddgs not installed. Install with: pip install ddgs")


class SearchBackendUnavailable(Exception):
    """单个搜索后端不可用（未配置 / 网络失败 / 配额用尽），调用方据此回退到下一个后端。"""

# 疑似垃圾/低质域名：多为 SEO 或不良站，不展示给用户
_SUSPICIOUS_DOMAIN_PATTERN = re.compile(
    r"\.(cc|tk|ml|ga|cf|gq|xyz|top|work|click|link|pw|buzz)(/|$)",
    re.I
)
# 域名主体若为长随机字母（无明显词），则过滤
_RANDOM_DOMAIN_PATTERN = re.compile(r"^[a-z0-9]{10,}$", re.I)


def _is_suspicious_url(url: str) -> bool:
    """过滤疑似垃圾/不良站点（如部分 .cc/.tk 等易被滥用的域名）。"""
    if not url or not url.startswith("http"):
        return True
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc or ""
        # 去掉端口
        host = host.split(":")[0].lower()
        if not host:
            return True
        # 可疑 TLD
        if _SUSPICIOUS_DOMAIN_PATTERN.search(host):
            return True
        # 主域名部分（最后一个 . 之前若还有多段则取倒数第二段之前）
        parts = host.rsplit(".", 2)
        name = parts[0] if parts else ""
        if len(name) >= 10 and _RANDOM_DOMAIN_PATTERN.match(name):
            return True
        return False
    except Exception:
        return False


class InformationQueryAgent(AgentBase):
    """
    信息查询智能体（真实检索版）

    核心功能：
    - 天气查询 - 使用 wttr.in 免费 API（无需搜索，结果可靠）
    - 网络搜索 - 使用 DDGS（开启 safesearch，过滤可疑来源）

    注意：
    - 差旅标准查询由独立的 RAGKnowledgeAgent 处理
    """

    def __init__(self, name: str = "InformationQueryAgent", model=None, **kwargs):
        super().__init__()
        self.name = name
        self.model = model
        from utils.skill_loader import SkillLoader
        self.skill_loader = SkillLoader()

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        if x is None:
            return Msg(name=self.name, content=json.dumps({"query_success": False}), role="assistant")

        # 解析输入
        content = x.content if not isinstance(x, list) else x[-1].content

        if isinstance(content, str):
            try:
                data = json.loads(content)
                context = data.get("context", {})
                user_query = context.get("rewritten_query", "") or content
            except json.JSONDecodeError:
                user_query = content
        else:
            user_query = str(content)

        # 天气类问题优先走 wttr.in，避免通用搜索返回低质结果
        if self._is_weather_query(user_query):
            logger.info(f"Weather query: {user_query}")
            try:
                result = await self._weather_query(user_query)
                return Msg(name=self.name, content=json.dumps(result, ensure_ascii=False), role="assistant")
            except Exception as e:
                logger.warning(f"Weather query failed, fallback to web search: {e}")
                result = None
        else:
            result = None

        if result is None:
            logger.info(f"Web search query: {user_query}")
            try:
                result = await self._web_search(user_query)
            except Exception as e:
                logger.error(f"Query failed: {e}")
                result = {
                    "query_type": "网络搜索",
                    "query_success": False,
                    "results": {"error": str(e)},
                }

        return Msg(name=self.name, content=json.dumps(result, ensure_ascii=False), role="assistant")

    def _is_weather_query(self, query: str) -> bool:
        """简单判断是否为天气类问题。"""
        q = (query or "").strip()
        if not q:
            return False
        return "天气" in q or "气温" in q or "下雨" in q or "预报" in q

    async def _weather_query(self, query: str) -> Dict[str, Any]:
        """
        使用 wttr.in 免费 API 查询天气（无需 API Key，结果可靠）。
        支持中文城市名，如：杭州、北京。
        """
        import asyncio
        try:
            import httpx
        except ImportError:
            return {
                "query_type": "天气查询",
                "query_success": False,
                "results": {"message": "需要安装 httpx: pip install httpx"},
            }

        # 从问题中提取城市（简单取第一个常见城市名或整句前 10 字中连续中文）
        city = self._extract_city_from_query(query)
        if not city:
            return {
                "query_type": "天气查询",
                "query_success": False,
                "results": {"message": "未识别到城市，请说明具体城市，如：杭州下周的天气怎么样？"},
            }

        url = f"https://wttr.in/{city}?format=j1"
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: httpx.get(url, timeout=10.0, headers={"User-Agent": "curl/7.64.1"}),
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"wttr.in request failed: {e}")
            return {
                "query_type": "天气查询",
                "query_success": False,
                "results": {"message": f"天气接口暂时不可用: {e}", "sources": [{"url": "https://wttr.in", "title": "wttr.in"}]},
            }

        try:
            current = data.get("current_condition", [{}])[0]
            temp_c = current.get("temp_C", "?")
            wdesc = current.get("weatherDesc", [{}])
            desc = (wdesc[0].get("value") if wdesc else None) or "—"
            humidity = current.get("humidity", "?")
            weather_text = f"{city}当前天气：{desc}，气温 {temp_c}°C，湿度 {humidity}%。"
            forecasts = []
            for day in data.get("weather", [])[:5]:
                date = day.get("date", "")
                maxtemp = day.get("maxtempC", "?")
                mintemp = day.get("mintempC", "?")
                h = (day.get("hourly") or [{}])[0] if day.get("hourly") else {}
                daily_desc = (h.get("weatherDesc") or [{}])[0].get("value", "—") if h else "—"
                forecasts.append(f"{date}: {daily_desc}，{mintemp}~{maxtemp}°C")
            if forecasts:
                weather_text += " 未来几日：" + "；".join(forecasts[:3])
            return {
                "query_type": "天气查询",
                "query_success": True,
                "results": {
                    "summary": weather_text,
                    "sources": [{"url": "https://wttr.in", "title": "wttr.in"}],
                },
            }
        except Exception as e:
            logger.warning(f"Parse wttr.in response failed: {e}")
            return {
                "query_type": "天气查询",
                "query_success": False,
                "results": {"message": "天气数据解析失败", "sources": [{"url": "https://wttr.in", "title": "wttr.in"}]},
            }

    def _extract_city_from_query(self, query: str) -> str:
        """从问题中提取城市名（简单实现：常见城市列表匹配）。"""
        common_cities = [
            "北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "武汉", "西安", "苏州",
            "天津", "重庆", "厦门", "青岛", "大连", "宁波", "无锡", "长沙", "郑州", "济南",
            "哈尔滨", "沈阳", "昆明", "合肥", "福州", "石家庄", "南昌", "贵阳", "太原", "南宁",
        ]
        q = (query or "").strip()
        for city in common_cities:
            if city in q:
                return city
        # 否则取前 2～6 个连续中文字作为可能城市名
        m = re.search(r"[\u4e00-\u9fa5]{2,6}", q)
        return m.group(0).strip() if m else ""

    # ---------- 网络搜索：多后端可插拔（Tavily / DDGS） ----------

    def _search_order(self) -> List[str]:
        """
        决定后端尝试顺序。

        - backend 为具体后端名（tavily / ddgs）时，只试它
        - backend="auto" 时按 SEARCH_CONFIG["auto_order"] 依次尝试
        """
        backend = str((SEARCH_CONFIG or {}).get("backend", "auto")).lower()
        if backend in ("tavily", "ddgs"):
            return [backend]

        order = (SEARCH_CONFIG or {}).get("auto_order")
        if not isinstance(order, list) or not order:
            order = ["tavily", "ddgs"]
        # 只保留已知后端，防止配置写错导致整条链路静默失效
        known = [b for b in order if b in ("tavily", "ddgs")]
        return known or ["tavily", "ddgs"]

    def _tavily_search_sync(self, query: str) -> List[Dict[str, str]]:
        """
        Tavily Search API（同步实现，由 asyncio.to_thread 调度）。

        文档：https://docs.tavily.com/documentation/api-reference/endpoint/search
        控制台：https://app.tavily.com （免费 1000 credits/月）

        与 DDGS 的区别：Tavily 返回的是**已抽取好的正文**（content），
        比搜索摘要长得多，因此按 max_content_chars 截断后再交给 LLM 摘要。
        """
        if not HTTPX_AVAILABLE:
            raise SearchBackendUnavailable("httpx 未安装：pip install httpx")

        cfg = (SEARCH_CONFIG or {}).get("tavily", {}) or {}
        api_key = str(cfg.get("api_key") or "").strip() or os.environ.get("TAVILY_API_KEY", "").strip()
        if not api_key:
            raise SearchBackendUnavailable(
                "未配置 Tavily API Key：请设置环境变量 TAVILY_API_KEY，"
                "或在 config.py 的 SEARCH_CONFIG['tavily']['api_key'] 填写"
            )

        body: Dict[str, Any] = {
            "query": query,
            "max_results": max(1, min(20, int(cfg.get("max_results", 10) or 10))),
            "search_depth": cfg.get("search_depth", "basic"),
            "include_answer": False,        # 摘要由本项目的 LLM 环节统一生成
            "include_raw_content": False,   # 只要抽取后的正文，不要原始 HTML
            "include_images": False,
        }
        if cfg.get("topic"):
            body["topic"] = cfg["topic"]
        if cfg.get("country"):
            body["country"] = cfg["country"]

        try:
            resp = httpx.post(
                cfg.get("endpoint", "https://api.tavily.com/search"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=float(cfg.get("timeout", 15.0)),
            )
        except Exception as e:
            raise SearchBackendUnavailable(f"Tavily 请求失败（网络不通或超时）: {e}") from e

        if resp.status_code != 200:
            detail = (resp.text or "")[:200]
            try:
                payload = resp.json()
                detail = (payload.get("detail") or payload.get("error") or detail)
                if isinstance(detail, (dict, list)):
                    detail = json.dumps(detail, ensure_ascii=False)[:200]
            except Exception:
                pass

            hint = ""
            if resp.status_code == 401:
                hint = "（API Key 无效或已被撤销，请到 app.tavily.com 重新复制）"
            elif resp.status_code == 429:
                hint = "（额度用尽或请求过频：免费 1000 credits/月，basic 检索 1 credit/次）"
            elif resp.status_code == 432:
                hint = "（Tavily 账户额度/权限不足）"
            elif resp.status_code == 400:
                hint = "（请求参数有误）"
            raise SearchBackendUnavailable(
                f"Tavily 返回 HTTP {resp.status_code}{hint} {detail}".strip()
            )

        try:
            data = resp.json()
        except Exception as e:
            raise SearchBackendUnavailable(f"Tavily 响应解析失败: {e}") from e

        limit_chars = int(cfg.get("max_content_chars", 500) or 500)
        results: List[Dict[str, str]] = []
        for item in data.get("results") or []:
            url = (item.get("url") or "").strip()
            if not url:
                continue
            content = item.get("content") or ""
            if limit_chars > 0 and len(content) > limit_chars:
                content = content[:limit_chars] + "…"
            results.append({
                "title": item.get("title", ""),
                "snippet": content,
                "url": url,
            })

        if not results:
            raise SearchBackendUnavailable("Tavily 未返回任何结果")
        return results

    def _ddgs_search_sync(self, query: str) -> List[Dict[str, str]]:
        """DDGS 兜底检索（同步实现，由 asyncio.to_thread 调度）。"""
        if not DDGS_AVAILABLE:
            raise SearchBackendUnavailable("ddgs 未安装：pip install ddgs")

        cfg = (SEARCH_CONFIG or {}).get("ddgs", {}) or {}
        backends = cfg.get("backends") or ["bing", "duckduckgo", "auto"]
        last_error = None

        for backend in backends:
            try:
                raw = DDGS().text(
                    query,
                    max_results=int(cfg.get("max_results", 10) or 10),
                    safesearch=cfg.get("safesearch", "on"),
                    region=cfg.get("region", "cn-zh"),
                    backend=backend,
                )
            except Exception as e:
                last_error = f"{backend}: {e}"
                logger.debug(f"DDGS backend {backend} failed: {e}")
                continue

            results = [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                }
                for r in raw
                if r.get("href")
            ]
            if results:
                return results

        raise SearchBackendUnavailable(
            f"DDGS 各后端均未返回结果（{last_error}）" if last_error else "DDGS 各后端均未返回结果"
        )

    async def _web_search(self, query: str) -> Dict[str, Any]:
        """
        网络搜索：按 SEARCH_CONFIG['backend'] / ['auto_order'] 依次尝试各后端。

        - 任一后端成功即返回，并在 results.engine 标注实际通道（tavily | ddgs）
        - 全部失败时返回 query_success=False，并附带每个后端的失败原因，便于排查
        - 成功结果统一归一化为 {title, snippet, url}，并过滤可疑域名后截断
        """
        raw_results: List[Dict[str, str]] = []
        used_engine: Optional[str] = None
        attempts: List[str] = []

        runners = {
            "tavily": self._tavily_search_sync,
            "ddgs": self._ddgs_search_sync,
        }

        for engine in self._search_order():
            runner = runners.get(engine)
            if runner is None:
                attempts.append(f"{engine}: 未知后端")
                continue
            try:
                raw_results = await asyncio.to_thread(runner, query)
                if not raw_results:
                    # 后端可用但没搜到东西，同样换下一个试试，别让空结果终止整条链路
                    attempts.append(f"{engine}: 返回 0 条结果")
                    logger.info(f"搜索后端 {engine} 返回 0 条结果，尝试下一个")
                    continue
                used_engine = engine
                logger.info(f"Search via {engine}: {len(raw_results)} raw results")
                break
            except SearchBackendUnavailable as e:
                attempts.append(f"{engine}: {e}")
                logger.info(f"搜索后端 {engine} 不可用，尝试下一个: {e}")
            except Exception as e:
                attempts.append(f"{engine}: {e}")
                logger.warning(f"搜索后端 {engine} 异常: {e}")

        if not raw_results:
            return {
                "query_type": "网络搜索",
                "query_success": False,
                "results": {
                    "message": "未找到相关结果",
                    "attempts": attempts,
                },
            }

        # 过滤可疑来源并截断
        limit = int((SEARCH_CONFIG or {}).get("max_results", 5) or 5)
        results: List[Dict[str, str]] = []
        for result in raw_results:
            if _is_suspicious_url(result.get("url", "")):
                continue
            results.append(result)
            if len(results) >= limit:
                break

        if not results:
            return {
                "query_type": "网络搜索",
                "query_success": False,
                "results": {
                    "message": "未找到相关结果（结果均被可疑域名过滤）",
                    "engine": used_engine,
                },
            }

        # 使用 LLM 总结搜索结果
        summary = await self._summarize_search_results(query, results)

        payload: Dict[str, Any] = {
            "summary": summary,
            "sources": results,
            "engine": used_engine,
        }
        if attempts:
            # 成功了但前面有后端失败 —— 暴露出来，避免静默降级无人察觉
            payload["fallback"] = attempts

        return {
            "query_type": "网络搜索",
            "query_success": True,
            "results": payload,
        }

    async def _summarize_search_results(self, query: str, results: List[Dict]) -> str:
        """
        使用 LLM 总结搜索结果

        Args:
            query: 用户查询
            results: 搜索结果列表

        Returns:
            总结文本
        """
        if not results:
            return "未找到相关信息"

        # 构建搜索结果文本
        results_text = ""
        for i, result in enumerate(results, 1):
            results_text += f"\n{i}. {result['title']}\n{result['snippet']}\n"

        # 获取当前时间
        from datetime import datetime
        current_date = datetime.now().strftime("%Y年%m月%d日")
        weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][datetime.now().weekday()]

        # 动态读取 Prompt 指令 (Progressive Disclosure)
        skill_instruction = self.skill_loader.get_skill_content("query-info")
        if not skill_instruction:
            skill_instruction = "请直接回答用户的问题，保持简洁。"

        prompt = f"""根据以下搜索结果，简洁地回答用户的问题。

【当前时间】
{current_date} {weekday}
（用户查询中的相对时间请基于此日期理解，如"明天"、"2月28日"等）

【用户问题】
{query}

【搜索结果】
{results_text}

【任务说明】
{skill_instruction}
"""

        try:
            response = await self.model([{"role": "user", "content": prompt}])

            # 获取响应文本 - 处理异步生成器
            text = ""
            if hasattr(response, '__aiter__'):
                # 异步生成器，需要迭代获取内容
                async for chunk in response:
                    if isinstance(chunk, str):
                        text = chunk
                    elif hasattr(chunk, 'content'):
                        if isinstance(chunk.content, str):
                            text = chunk.content
                        elif isinstance(chunk.content, list):
                            for item in chunk.content:
                                if isinstance(item, dict) and item.get('type') == 'text':
                                    text = item.get('text', '')
            elif hasattr(response, 'text'):
                text = response.text
            elif hasattr(response, 'content'):
                text = response.content
            elif isinstance(response, dict) and 'content' in response:
                text = response['content']
            else:
                text = str(response) if response else ""

            return text.strip() if text else "无法生成摘要"
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return "搜索成功，但摘要生成失败"
