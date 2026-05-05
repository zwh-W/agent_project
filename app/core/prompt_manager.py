# app/core/prompt_manager.py
"""
Prompt 版本管理系统

设计动机：
    现在的 System Prompt 是硬编码在 manager.py 里的字符串。
    这意味着每次改 Prompt 都要改代码、重启服务、重新部署。
    在真实业务里，Prompt 是需要高频迭代的——
    今天改一句话效果好了，明天可能又要回滚。
    把 Prompt 当代码管理是一个非常初级的错误。

解决方案：Prompt 版本管理
    1. Prompt 存储在文件系统（prompts/ 目录），与代码解耦
    2. 支持版本号，可以随时切换和回滚
    3. 支持变量插值（Jinja2 语法），Prompt 可以动态注入上下文
    4. 支持热加载，修改文件后不需要重启服务
    5. 记录每个版本的使用情况，为 A/B 测试打基础

面试价值：
    能说出"我把 Prompt 做了版本管理"的候选人，
    面试官会立刻意识到你在生产环境里真的踩过坑，
    而不是在沙盒里写 demo。

目录结构（自动创建）：
    prompts/
    ├── system/
    │   ├── v1.txt          ← 版本 1 的 system prompt
    │   ├── v2.txt          ← 版本 2
    │   └── current -> v2   ← 软链接，指向当前使用版本（或直接用 current.txt）
    ├── supervisor/
    │   └── v1.txt
    └── router/
        └── v1.txt
"""
import os
import re
import time
import threading
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, field

from app.core.logger import get_logger

logger = get_logger(__name__)

# Prompt 文件根目录（相对于项目根）
_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


@dataclass
class PromptVersion:
    """一个 Prompt 版本的完整信息"""
    name: str           # prompt 名称，如 "system"
    version: str        # 版本号，如 "v1"
    content: str        # 原始模板内容
    file_path: Path     # 文件路径
    loaded_at: float    # 加载时间戳
    use_count: int = 0  # 被调用次数（用于统计）


class PromptManager:
    """
    Prompt 版本管理器

    使用方式：
        # 获取当前版本的 system prompt（无变量）
        prompt = prompt_manager.get("system")

        # 获取带变量插值的 prompt
        prompt = prompt_manager.get("system", summary="用户上次说他叫张三")

        # 指定版本
        prompt = prompt_manager.get("system", version="v1")

        # 查看所有可用版本
        versions = prompt_manager.list_versions("system")
    """

    def __init__(self, prompts_dir: Path = _PROMPTS_DIR):
        self.prompts_dir = prompts_dir
        self._cache: Dict[str, PromptVersion] = {}  # key: "name:version"
        self._lock = threading.Lock()
        self._current_versions: Dict[str, str] = {}  # name → current version

        # 确保 prompts 目录存在，并写入默认 prompt 文件
        self._ensure_default_prompts()

    def get(
        self,
        name: str,
        version: Optional[str] = None,
        **variables
    ) -> str:
        """
        获取 Prompt 内容

        Args:
            name:      prompt 名称（对应 prompts/<name>/ 目录）
            version:   版本号（None 表示使用当前版本）
            **variables: 模板变量，用于 {variable_name} 格式的插值

        Returns:
            处理完变量插值后的 prompt 字符串
        """
        resolved_version = version or self._get_current_version(name)
        cache_key = f"{name}:{resolved_version}"

        with self._lock:
            # 尝试从缓存获取
            if cache_key in self._cache:
                pv = self._cache[cache_key]
                pv.use_count += 1
                content = pv.content
            else:
                # 从文件加载
                content = self._load_from_file(name, resolved_version)
                self._cache[cache_key] = PromptVersion(
                    name=name,
                    version=resolved_version,
                    content=content,
                    file_path=self.prompts_dir / name / f"{resolved_version}.txt",
                    loaded_at=time.time(),
                    use_count=1,
                )

        # 变量插值（在锁外执行，避免长时间持锁）
        if variables:
            content = self._interpolate(content, variables)

        return content

    def reload(self, name: str, version: Optional[str] = None):
        """
        强制热加载（清除缓存，重新从文件读取）
        不需要重启服务即可生效新 Prompt

        使用场景：运营同学改了 prompt 文件后，调用此方法立即生效
        """
        resolved_version = version or self._get_current_version(name)
        cache_key = f"{name}:{resolved_version}"

        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                logger.info(f"已热加载 Prompt: {name}/{resolved_version}")

    def list_versions(self, name: str) -> List[str]:
        """列出某个 prompt 的所有可用版本"""
        prompt_dir = self.prompts_dir / name
        if not prompt_dir.exists():
            return []
        return sorted([
            f.stem for f in prompt_dir.glob("v*.txt")
        ])

    def get_stats(self) -> Dict:
        """获取所有 Prompt 的使用统计"""
        with self._lock:
            return {
                key: {
                    "name": pv.name,
                    "version": pv.version,
                    "use_count": pv.use_count,
                    "loaded_at": pv.loaded_at,
                }
                for key, pv in self._cache.items()
            }

    def _get_current_version(self, name: str) -> str:
        """获取某个 prompt 的当前版本号"""
        if name in self._current_versions:
            return self._current_versions[name]

        # 自动选择最新版本（版本号最大的）
        versions = self.list_versions(name)
        if not versions:
            raise FileNotFoundError(
                f"找不到名为 '{name}' 的 Prompt，"
                f"请在 {self.prompts_dir / name}/ 目录下创建 v1.txt"
            )

        # 取版本号最大的（按字典序，v10 > v9 需要特殊处理）
        latest = sorted(versions, key=lambda v: int(v[1:]) if v[1:].isdigit() else 0)[-1]
        self._current_versions[name] = latest
        return latest

    def _load_from_file(self, name: str, version: str) -> str:
        """从文件系统加载 Prompt"""
        file_path = self.prompts_dir / name / f"{version}.txt"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Prompt 文件不存在: {file_path}\n"
                f"可用版本: {self.list_versions(name)}"
            )

        content = file_path.read_text(encoding="utf-8").strip()
        logger.debug(f"从文件加载 Prompt: {file_path}，长度 {len(content)} 字符")
        return content

    def _interpolate(self, template: str, variables: dict) -> str:
        """
        简单的变量插值：将 {variable_name} 替换为对应值
        比 Jinja2 更轻量，适合 Prompt 场景
        """
        result = template
        for key, value in variables.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, str(value) if value else "")
        return result

    def _ensure_default_prompts(self):
        """确保默认 Prompt 文件存在，方便开箱即用"""
        defaults = {
            "system/v1.txt": _DEFAULT_SYSTEM_PROMPT,
            "system/v2.txt": _DEFAULT_SYSTEM_PROMPT_V2,
            "supervisor/v1.txt": _DEFAULT_SUPERVISOR_PROMPT,
            "router/v1.txt": _DEFAULT_ROUTER_PROMPT,
        }

        for relative_path, content in defaults.items():
            full_path = self.prompts_dir / relative_path
            if not full_path.exists():
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content, encoding="utf-8")
                logger.debug(f"已创建默认 Prompt 文件: {full_path}")


# ──────────────────────────────────────────────
# 默认 Prompt 内容
# 写在代码里是为了方便首次运行时自动生成文件
# 真实使用时应该由产品/运营来维护这些 .txt 文件
# ──────────────────────────────────────────────

_DEFAULT_SYSTEM_PROMPT = """你是一个专业、严谨的 AI 助手。

【核心原则】
1. 对于需要实时数据、最新信息、具体计算的问题，你必须使用对应工具，绝不能凭记忆编造。
2. 如果工具返回了错误，你需要分析原因并重试，而不是直接告诉用户失败了。
3. 最终回答必须简洁、准确、有条理。

【工具使用规则】
- 需要搜索事实/新闻/实时信息 → 使用 web_search
- 需要数学计算 → 使用 calculator，不要心算

【输出格式】
- 直接给出答案，不要说"根据我的知识"或"我认为"
- 如果使用了工具，可以简短说明信息来源
- 不要重复用户的问题
{summary_section}
{long_term_section}"""

_DEFAULT_SYSTEM_PROMPT_V2 = """你是一个专业、严谨的企业级 AI 助手。

【身份定位】
你服务于企业用户，回答需要准确、专业、有据可查。

【强制规则（违反则视为错误）】
1. 涉及数字计算 → 必须调用 calculator 工具，哪怕是简单加减法
2. 涉及实时/近期信息 → 必须调用 web_search，不可凭印象回答
3. 工具调用失败 → 必须分析原因并至少重试一次，而非直接报错给用户

【回答规范】
- 结论优先：先给答案，再解释过程
- 来源透明：凡是工具返回的信息，说明"根据搜索结果"
- 拒绝模糊：不说"大约"、"可能"、"我认为"这类模糊词汇
{summary_section}
{long_term_section}"""

_DEFAULT_SUPERVISOR_PROMPT = """你是一个团队主管，管理两个员工：
- Researcher：负责联网搜索事实、新闻、实时数据
- Calculator：负责数学计算、公式求值

根据对话历史，决定下一步派给谁。任务彻底完成或不需要员工时回复 FINISH。

【输出规则】只能输出以下三个词之一：
Researcher / Calculator / FINISH

【判断示例】
"今天比特币价格" → Researcher
"123 乘以 456" → Calculator
"研究员已汇报，现在需要计算" → Calculator
"计算结果已给出，任务完成" → FINISH"""

_DEFAULT_ROUTER_PROMPT = """你是 Agent 路由专家，判断用户问题交给哪种 Agent。

可选类型：
- function_calling：单步任务，简单计算，日常问答
- langgraph：多步推理，分析类问题
- multi_agent：需要多角色协作，先搜索再计算
- mcp：访问企业内网工具，查询内部系统数据

只输出类型名称，不加解释。"""


# 全局单例
prompt_manager = PromptManager()