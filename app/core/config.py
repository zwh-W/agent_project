# app/core/config.py
"""
配置加载模块

设计原则：
1. 所有配置从 config.yaml 读取
2. 敏感信息（API Key）从环境变量读取，不写进 yaml
3. 用 dataclass 做类型约束，IDE 能自动补全
4. 单例：整个进程只加载一次
"""
import os
import yaml
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent.parent


def _load_yaml() -> dict:
    config_path = BASE_DIR / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class AppConfig:
    host: str
    port: int
    debug: bool


@dataclass
class LLMConfig:
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    streaming: bool
    # API Key 从环境变量读取
    api_key: str = field(default="")


@dataclass
class AgentConfig:
    max_iterations: int
    max_supervisor_rounds: int
    tool_timeout: int
    tool_retry: bool
    tool_retry_times: int


@dataclass
class MemoryConfig:
    short_term_max_messages: int
    chroma_path: str
    long_term_top_k: int


@dataclass
class ToolsConfig:
    search_max_results: int
    rag_api_url: str
    rag_knowledge_id: int


@dataclass
class MCPConfig:
    server_script: str
    transport: str


@dataclass
class LoggingConfig:
    level: str
    file: str


@dataclass
class Settings:
    """
    全局配置入口，使用方式：
        from app.core.config import settings
        settings.llm.model
        settings.agent.max_iterations
    """
    app: AppConfig
    llm: LLMConfig
    agent: AgentConfig
    memory: MemoryConfig
    tools: ToolsConfig
    mcp: MCPConfig
    logging: LoggingConfig
    base_dir: Path


def _build_settings() -> Settings:
    raw = _load_yaml()

    # API Key 必须从环境变量读取
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        warnings.warn(
            "环境变量 DASHSCOPE_API_KEY 未设置，LLM 相关功能将无法使用。\n"
            "请在 .env 文件中配置：DASHSCOPE_API_KEY=sk-your-key",
            stacklevel=2
        )

    return Settings(
        app=AppConfig(**raw["app"]),
        llm=LLMConfig(
            **raw["llm"],
            api_key=api_key,
        ),
        agent=AgentConfig(**raw["agent"]),
        memory=MemoryConfig(**raw["memory"]),
        tools=ToolsConfig(**raw["tools"]),
        mcp=MCPConfig(**raw["mcp"]),
        logging=LoggingConfig(**raw["logging"]),
        base_dir=BASE_DIR,
    )


# 单例
settings = _build_settings()