import os
import yaml
from typing import Dict, List, Optional

class SkillLoader:
    """加载 .claude/skills 下的技能描述"""
    
    def __init__(self, skills_dir: str = ".claude/skills"):
        # 获取当前文件的绝对路径，然后找到项目根目录
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path))
        self.skills_dir = os.path.join(project_root, skills_dir)
        self.skills: Dict[str, Dict] = {}
        
    def load_skills(self) -> Dict[str, Dict]:
        """
        读取所有 SKILL.md 文件并解析 frontmatter
        返回格式: { "skill-name": { "name": "...", "description": "..." } }
        """
        if not os.path.exists(self.skills_dir):
            print(f"Warning: Skills directory {self.skills_dir} not found.")
            return {}
            
        for skill_name in os.listdir(self.skills_dir):
            skill_path = os.path.join(self.skills_dir, skill_name)
            if os.path.isdir(skill_path):
                md_file = os.path.join(skill_path, "SKILL.md")
                if os.path.exists(md_file):
                    skill_info = self._parse_skill_md(md_file)
                    if skill_info:
                        self.skills[skill_info.get("name", skill_name)] = skill_info
                        
        return self.skills

    def _parse_skill_md(self, file_path: str) -> Optional[Dict]:
        """解析 markdown 文件的 yaml frontmatter"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 简单的 frontmatter 解析
            if content.startswith('---'):
                end_idx = content.find('---', 3)
                if end_idx != -1:
                    yaml_content = content[3:end_idx]
                    try:
                        data = yaml.safe_load(yaml_content)
                        return data
                    except yaml.YAMLError as e:
                        print(f"Error parsing YAML in {file_path}: {e}")
            return None
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def get_skill_prompt(self, skill_mapping: Optional[Dict[str, str]] = None) -> str:
        """
        生成用于 Prompt 的技能描述字符串
        
        Args:
            skill_mapping: 可选，将 skill name 映射为系统内部的 intent name
                          例如: {"plan-trip": "itinerary_planning"}
        """
        if not self.skills:
            self.load_skills()
            
        prompt_lines = []
        index = 1
        
        # 排序以保证确定性
        sorted_skills = sorted(self.skills.items())
        
        for name, info in sorted_skills:
            display_name = name
            if skill_mapping and name in skill_mapping:
                display_name = skill_mapping[name]
            elif skill_mapping:
                # 尝试反向查找：如果name是目录名，也试试看有没有在mapping里
                pass
                
            desc = info.get("description", "").replace("\n", " ")
            prompt_lines.append(f"{index}. {display_name} - {desc}")
            index += 1
            
        return "\n\n".join(prompt_lines)

    def get_skill_content(self, skill_name: str) -> Optional[str]:
        """
        获取指定 Skill 的完整 markdown 内容（去除 frontmatter）
        用于执行阶段的 Prompt 注入
        """
        if not self.skills:
            self.load_skills()
            
        target_path = None
        
        # 1. 优先尝试直接按目录名匹配 (最快)
        test_dir_path = os.path.join(self.skills_dir, skill_name, "SKILL.md")
        if os.path.exists(test_dir_path):
            target_path = test_dir_path
        
        # 2. 如果没找到，尝试按 metadata 中的 name 匹配 (遍历)
        if not target_path:
            for dirname in os.listdir(self.skills_dir):
                skill_dir = os.path.join(self.skills_dir, dirname)
                if not os.path.isdir(skill_dir):
                    continue
                    
                md_path = os.path.join(skill_dir, "SKILL.md")
                if os.path.exists(md_path):
                    # 简单读取 name
                    info = self._parse_skill_md(md_path)
                    if info and info.get("name") == skill_name:
                        target_path = md_path
                        break

        if not target_path:
            return None
            
        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 去除 frontmatter
            if content.startswith('---'):
                end_idx = content.find('---', 3)
                if end_idx != -1:
                    content = content[end_idx+3:].strip()
            return content
        except Exception as e:
            print(f"Error reading skill content {target_path}: {e}")
            return None

