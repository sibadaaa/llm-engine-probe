"""vLLM stat logger 插件：把每个 engine step 的结构化统计写成 JSONL。

为什么需要它
------------
``/metrics`` 是**聚合并降采样**后的数字（直方图 + 计数器），日志是**给人看的**
文本。而 vLLM 引擎内部本来就有一个 ``SchedulerStats`` / ``IterationStats``
对象，里面同时带着：

* 逐 step 批次构成与耗时（``iteration_details``）
* 调度状态（running / waiting / skipped waiting / kv_cache_usage / wave）
* 前缀缓存命中与抢占（``prefix_cache_stats``）
* 投机解码逐步接受分布（``spec_decoding_stats``，含按位置数组）
* KV block 驱逐事件（``kv_cache_eviction_events``）
* 图调度统计（``cudagraph_stats``）、MFU 基础量（``perf_stats``）
* **每个已完成请求的完整生命周期**（``FinishedRequestStats``：queue/prefill/decode/
  e2e/TPOT/cached tokens/抢占次数）

本插件通过 vLLM 官方插件机制（entry point 组 ``vllm.stat_logger_plugins``）
在 ``record()`` 里把这些对象原样落到 JSONL —— 不解析日志、不抓 Prometheus，
拿到的是引擎里最原始、最精确的那一份数据。

接口一致性（本机 vLLM 0.18.1+das 实测）
--------------------------------------
* 组名：``vllm.stat_logger_plugins``（``vllm/plugins/__init__.py::STAT_LOGGER_PLUGINS_GROUP``）
* entry point 的值必须是 ``StatLoggerBase`` 的**子类**（不是函数），否则
  ``load_stat_logger_plugin_factories()`` 抛 ``TypeError``
* 每个 engine 一个实例，构造方式为 ``factory(vllm_config, engine_index)``（两个位置参数）
* ``record(scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0)``
  每个 engine step 调用一次

环境变量
--------
``VLLM_PROBE_LOG``      输出文件（默认 ``/logs/probe_steps.jsonl``）
``VLLM_PROBE_DISABLE``  置 1 关闭（便于 A/B 对比插件自身开销）

⚠️ 不要把 ``VLLM_PLUGINS`` 设成只含本插件：该变量是**全局白名单**，
   设了之后 ``vllm.general_plugins`` 里的内置插件（如 LoRA filesystem/hf_hub
   resolver）就不会加载。留空即可让所有插件正常加载。
"""

from __future__ import annotations

import atexit
import json
import os
import threading
from pathlib import Path
from typing import Any

from vllm.v1.metrics.loggers import StatLoggerBase

from .records import SCHEMA_VERSION, build_record

__all__ = ["ProbeStatLogger"]

DEFAULT_LOG_PATH = "/logs/probe_steps.jsonl"


def _logger():
    try:
        from vllm.logger import init_logger

        return init_logger(__name__)
    except Exception:  # noqa: BLE001
        import logging

        return logging.getLogger(__name__)


def _vllm_version() -> str | None:
    """尽力取 vLLM 版本号：官方 ``__version__``，再尝试各家 fork 的版本属性。"""
    try:
        from vllm import version

        for attr in (
            "__version__",
            "__hcu_version__",       # Hygon DCU fork
            "__ascend_version__",    # vllm-ascend fork
            "__rocm_version__",
        ):
            value = getattr(version, attr, None)
            if value:
                return str(value)
        return None
    except Exception:  # noqa: BLE001
        return None


class ProbeStatLogger(StatLoggerBase):
    """把每个 engine step 写成一行 JSON 的 stat logger。

    任何异常都被吞掉并计数（``self.failures``），**绝不影响推理主循环**。
    """

    def __init__(self, vllm_config: Any, engine_index: int = 0) -> None:
        self.vllm_config = vllm_config
        self.engine_index = int(engine_index)
        self.step = 0
        self.failures = 0
        self.records_written = 0
        self.path = Path(os.environ.get("VLLM_PROBE_LOG") or DEFAULT_LOG_PATH)
        self.disabled = os.environ.get("VLLM_PROBE_DISABLE", "0") == "1"
        self._lock = threading.Lock()
        self._fh = None
        self._log = _logger()
        if not self.disabled:
            self._open()
        atexit.register(self.close)

    # ------------------------------------------------------------------
    def _open(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")
            self._write(
                {
                    "schema": SCHEMA_VERSION,
                    "kind": "header",
                    "engine": self.engine_index,
                    "pid": os.getpid(),
                    "vllm_version": _vllm_version(),
                    "path": str(self.path),
                }
            )
            self._log.info(
                "[vllm-probe] step 级 JSONL 已启用：%s（engine=%d）", self.path, self.engine_index
            )
        except OSError as exc:
            self._fh = None
            self.disabled = True
            self._log.warning("[vllm-probe] 无法写入 %s（%s），插件已禁用", self.path, exc)

    def _write(self, payload: dict[str, Any]) -> None:
        if self._fh is None:
            return
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    # ------------------------------------------------------------------
    # StatLoggerBase 接口
    # ------------------------------------------------------------------
    def record(
        self,
        scheduler_stats: Any,
        iteration_stats: Any,
        mm_cache_stats: Any = None,
        engine_idx: int = 0,
    ) -> None:
        if self.disabled:
            return
        try:
            self.step += 1
            record = build_record(
                scheduler_stats,
                iteration_stats,
                engine_index=int(engine_idx or self.engine_index),
                step=self.step,
            )
            self._write(record)
            self.records_written += 1
        except Exception as exc:  # noqa: BLE001 - 统计失败不能影响引擎
            self.failures += 1
            if self.failures == 1:
                self._log.warning("[vllm-probe] 记录 step 失败（后续静默计数）：%r", exc)

    def log_engine_initialized(self) -> None:  # noqa: D102
        self._write(
            {
                "schema": SCHEMA_VERSION,
                "kind": "engine_initialized",
                "engine": self.engine_index,
            }
        )

    def log(self) -> None:  # noqa: D102
        # 每条记录都 flush，这里无需额外动作；保留以兼容接口。
        return None

    def record_sleep_state(self, is_awake: int, level: int) -> None:  # noqa: D102
        self._write(
            {
                "schema": SCHEMA_VERSION,
                "kind": "sleep_state",
                "engine": self.engine_index,
                "is_awake": int(is_awake),
                "level": int(level),
            }
        )

    # ------------------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.flush()
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None
