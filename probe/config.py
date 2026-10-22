"""统一配置：环境变量 → 命令行 → 显式赋值，优先级递增。

环境变量命名与 ``.env.example`` 保持一致，便于 shell 脚本和 Python 工具
共享同一份配置。所有字段都有默认值，默认端口是 **8100**（刻意避开生产
实例的 8000），避免误打到正在做 A/B 实验的服务上。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8100"
DEFAULT_MODEL = "qwen3.8-27b"


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def mask_secret(value: str, *, keep: int = 4) -> str:
    """把 API key 之类的敏感串打码：保留前 keep 位 + 末 keep 位。

    产物（http_summary.json 等）会原样落盘配置，不能把明文 key 写进去；
    打码后仍能人工核对"是不是同一把 key"。
    """
    text = str(value or "")
    if text in ("", "EMPTY"):
        return text
    if len(text) <= keep * 2:
        return "***"
    return f"{text[:keep]}***{text[-keep:]}"


@dataclass
class ProbeConfig:
    """一次探测运行的全部上下文。"""

    # ---- 服务端 ----
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = "EMPTY"
    timeout: float = 900.0

    # ---- 产物目录 ----
    outdir: str = "runs/latest"

    # ---- 请求负载默认值 ----
    prompt_tokens: int = 512
    max_tokens: int = 128
    concurrency: int = 1
    repeat: int = 1
    endpoint: str = "chat"  # chat | completions
    stream: bool = True
    shared_prefix_tokens: int = 0
    temperature: float = 0.0

    # ---- 容器/引擎定位（shell 脚本用，Python 侧只做记录）----
    container: str = "vllm-probe"
    # 引擎组件 id（engines/<id>.sh ↔ probe/engines.py；探针按能力位协商行为）
    engine: str = "vllm"
    # 采集引擎专有指标（F3 T3.8，默认关闭；PROBE_ENGINE_EXTRAS=1 或 --engine-extras 开启）
    engine_extras: bool = False
    port: int = 8100
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "ProbeConfig":
        return cls(
            base_url=env_str("VLLM_BASE_URL", DEFAULT_BASE_URL),
            model=env_str("SERVED_NAME", DEFAULT_MODEL),
            api_key=env_str("VLLM_API_KEY", "EMPTY"),
            timeout=env_float("PROBE_TIMEOUT", 900.0),
            outdir=env_str("PROBE_OUTDIR", "runs/latest"),
            prompt_tokens=env_int("PROBE_PROMPT_TOKENS", 512),
            max_tokens=env_int("PROBE_MAX_TOKENS", 128),
            concurrency=env_int("PROBE_CONCURRENCY", 1),
            repeat=env_int("PROBE_REPEAT", 1),
            endpoint=env_str("PROBE_ENDPOINT", "chat"),
            stream=env_str("PROBE_STREAM", "1") not in ("0", "false", "False"),
            shared_prefix_tokens=env_int("PROBE_SHARED_PREFIX_TOKENS", 0),
            temperature=env_float("PROBE_TEMPERATURE", 0.0),
            container=env_str("CONTAINER", "vllm-probe"),
            engine=env_str("ENGINE", "vllm"),
            engine_extras=env_str("PROBE_ENGINE_EXTRAS", "0") in ("1", "true", "True"),
            port=env_int("PORT", 8100),
        )

    # ------------------------------------------------------------------
    def apply_namespace(self, ns: Any) -> "ProbeConfig":
        """用 argparse 的 Namespace 覆盖非 None 字段。"""
        for key, value in vars(ns).items():
            if value is None:
                continue
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/v1/chat/completions"

    @property
    def completions_url(self) -> str:
        return self.base_url.rstrip("/") + "/v1/completions"

    @property
    def metrics_url(self) -> str:
        return self.base_url.rstrip("/") + "/metrics"

    def request_url(self) -> str:
        return self.chat_url if self.endpoint == "chat" else self.completions_url

    # ------------------------------------------------------------------
    def out_path(self, *parts: str) -> Path:
        path = Path(self.outdir).joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_outdir(self) -> Path:
        path = Path(self.outdir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def dump(self, path: str | Path | None = None) -> dict[str, Any]:
        data = asdict(self)
        # 安全：产物里不能出现明文 API key（http_summary/suite 各场景会落这份配置）
        data["api_key"] = mask_secret(self.api_key)
        data["url"] = self.request_url()
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return data

    # ------------------------------------------------------------------
    def auth_headers(self) -> dict[str, str]:
        """探测请求用的鉴权头（未配置 key 时返回空字典）。"""
        key = str(self.api_key or "")
        if key in ("", "EMPTY"):
            return {}
        return {"Authorization": f"Bearer {key}"}

    # ------------------------------------------------------------------
    def describe(self) -> str:
        return (
            f"base_url={self.base_url} model={self.model} "
            f"prompt_tokens={self.prompt_tokens} max_tokens={self.max_tokens} "
            f"concurrency={self.concurrency} repeat={self.repeat} "
            f"endpoint={self.endpoint} stream={self.stream} engine={self.engine} "
            f"outdir={self.outdir}"
        )
