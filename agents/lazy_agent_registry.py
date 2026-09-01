#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
懒加载智能体注册器
基于 .claude/skills 目录结构的插件化加载机制
"""
import os
import sys
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from agentscope.agent import AgentBase

class LazyAgentRegistry:
    """
    懒加载智能体注册器 - 插件化版本
    
    自动扫描 .claude/skills 下的技能目录，动态加载 script/agent.py
    """

    def __init__(self, model, cache: Dict, memory_manager=None):
        """
        初始化懒加载注册器

        Args:
            model: 共享的 LLM 模型实例
            cache: 用于缓存已加载智能体的字典
            memory_manager: 记忆管理器 (可选，用于注入给需要它的 Agent)
        """
        self.model = model
        self.cache = cache
        self.memory_manager = memory_manager
        self.console = Console()
        
        # 技能目录路径
        self.skills_root = Path(".claude/skills")
        
        # 技能映射表: skill_name -> agent_script_path
        self._skill_map: Dict[str, Path] = {}
        
        # 发现技能
        self._discover_skills()
        
        # 旧版兼容映射 (name -> skill_folder_name)
        self._legacy_mapping = {
            "rag_knowledge": "ask-question",
            "memory_query": "memory-query",
            "preference": "preference",
            "information_query": "query-info",
            "itinerary_planning": "plan-trip",
            "event_collection": "event-collection"
        }

    def _discover_skills(self):
        """扫描 .claude/skills 目录寻找可用的 Agent"""
        if not self.skills_root.exists():
            self.console.print(f"[yellow]Warning: Skills directory {self.skills_root} not found[/yellow]")
            return

        count = 0
        for skill_dir in self.skills_root.iterdir():
            if not skill_dir.is_dir():
                continue
            
            # 查找 script/agent.py
            agent_script = skill_dir / "script" / "agent.py"
            if agent_script.exists():
                skill_name = skill_dir.name
                self._skill_map[skill_name] = agent_script
                count += 1
                
        # self.console.print(f"[dim]已发现 {count} 个技能插件[/dim]")

    def _resolve_agent_name(self, agent_name: str) -> Optional[str]:
        """解析智能体名称到技能目录名"""
        # 1. 直接匹配技能名
        if agent_name in self._skill_map:
            return agent_name
            
        # 2. 尝试遗留映射
        if agent_name in self._legacy_mapping:
            skill_name = self._legacy_mapping[agent_name]
            if skill_name in self._skill_map:
                return skill_name
                
        return None

    def __getitem__(self, agent_name: str):
        """获取智能体 (懒加载)"""
        if agent_name in self.cache:
            return self.cache[agent_name]

        skill_name = self._resolve_agent_name(agent_name)
        if not skill_name:
             raise KeyError(f"Agent '{agent_name}' not found in skills directory")

        script_path = self._skill_map[skill_name]
        
        self.console.print(f"[dim]🔄 正在加载 {agent_name} (from {skill_name})...[/dim]")
        
        try:
            # 1. 动态加载模块
            module_name = f"skills.{skill_name}.agent"
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load spec from {script_path}")
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            
            # 关键：确保模块能找到项目根目录的包 (utils, config 等)
            project_root = str(Path(__file__).parent.parent.absolute())
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
                
            spec.loader.exec_module(module)
            
            # 2. 查找 Agent 类
            agent_class = None
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, AgentBase) and obj is not AgentBase:
                    agent_class = obj
                    break
            
            if not agent_class:
                raise ValueError(f"No AgentBase subclass found in {script_path}")
                
            # 3. 实例化
            # 构造参数：name, model, memory_manager (如果需要), **kwargs
            init_params = {
                "name": agent_name, # 使用请求的名字，或者可以用 skill_name
                "model": self.model,
            }
            
            # 检查是否需要 memory_manager
            sig = inspect.signature(agent_class.__init__)
            if "memory_manager" in sig.parameters:
                init_params["memory_manager"] = self.memory_manager
                
            agent_instance = agent_class(**init_params)
            
            # 4. 缓存
            self.cache[agent_name] = agent_instance
            self.console.print(f"[dim]✓ {agent_name} 加载完成[/dim]")
            
            return agent_instance
            
        except Exception as e:
            self.console.print(f"[red]✗ 加载 {agent_name} 失败: {e}[/red]")
            import traceback
            traceback.print_exc()
            raise

    def __contains__(self, agent_name: str) -> bool:
        return self._resolve_agent_name(agent_name) is not None or agent_name in self.cache

    def get(self, agent_name: str, default=None):
        try:
            return self[agent_name]
        except KeyError:
            return default

    def keys(self):
        # 返回所有可能的 key（包括 legacy mapping 的 key，为了兼容 orchestrator）
        keys = set(self._skill_map.keys())
        for legacy_key, skill_val in self._legacy_mapping.items():
            if skill_val in self._skill_map:
                keys.add(legacy_key)
        return list(keys)

    def values(self):
        return self.cache.values()

    def items(self):
        return self.cache.items()
        
    def get_loaded_agents(self) -> list:
        return list(self.cache.keys())
