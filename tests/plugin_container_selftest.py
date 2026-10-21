"""容器内自检：验证 stat logger 插件能被 vLLM 官方加载器发现，并正确序列化真实统计对象。

必须在**装了 vLLM 的容器内**运行（宿主机没有 vLLM）。通用写法（任何厂商的
vLLM 镜像都适用；hygon 需要在启动前 source DTK 环境脚本）：

    docker run --rm \\
      -v /home/lyh/vllm-test/llm-engine-probe:/opt/vllm-probe \\
      <厂商设备参数...> <你的 vLLM 镜像> bash -lc '
        # hygon: . /opt/dtk/env.sh >/dev/null 2>&1
        export PYTHONPATH=/opt/vllm-probe
        pip install -q --no-deps --no-build-isolation -e /opt/vllm-probe/plugin_probe
        python3 /opt/vllm-probe/tests/plugin_container_selftest.py'

容器内也可以直接用工具包自带的启动逻辑生成完整命令：

    VENDOR=hygon bash scripts/start_vllm.sh --dry-run   # 看设备参数/环境脚本长什么样

检查四件事：
1. entry point 组 ``vllm.stat_logger_plugins`` 能被 ``load_stat_logger_plugin_factories()`` 发现，
   且返回的类是 ``StatLoggerBase`` 子类（vLLM 要求）；
2. 用真实的 ``SchedulerStats`` / ``IterationStats`` / ``SpecDecodingStats`` /
   ``FinishedRequestStats`` 对象调用 ``record()`` 不抛异常；
3. 落盘的 JSONL 里带上了关键字段（批次构成、前缀缓存、投机解码按位置、逐请求生命周期）；
4. 与当前 vLLM 版本的字段名一致（字段改名会在这里暴露，而不是在跑实验时静默丢数据）。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "✅" if condition else "❌"
    print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    print("== 1) vLLM 插件加载器发现 ==")
    from vllm.plugins import STAT_LOGGER_PLUGINS_GROUP
    from vllm.v1.metrics.loggers import StatLoggerBase, load_stat_logger_plugin_factories

    print(f"  组名: {STAT_LOGGER_PLUGINS_GROUP}")
    factories = load_stat_logger_plugin_factories()
    check("entry point 被发现", bool(factories), str(factories))
    check("返回值是 StatLoggerBase 子类", all(issubclass(f, StatLoggerBase) for f in factories))

    print("\n== 2) 真实统计对象序列化 ==")
    from vllm.v1.metrics.stats import (
        FinishedRequestStats,
        IterationStats,
        PrefixCacheStats,
        PromptTokenStats,
        SchedulerIterationDetails,
        SchedulerStats,
    )
    from vllm.v1.spec_decode.metrics import SpecDecodingStats

    tmpdir = tempfile.mkdtemp(prefix="probe-plugin-")
    log_path = Path(tmpdir) / "probe_steps.jsonl"
    os.environ["VLLM_PROBE_LOG"] = str(log_path)

    from vllm_probe_stat_logger import ProbeStatLogger

    scheduler_stats = SchedulerStats(
        num_running_reqs=3,
        num_waiting_reqs=2,
        num_skipped_waiting_reqs=1,
        kv_cache_usage=0.42,
        step_counter=99,
        current_wave=4,
        iteration_details=SchedulerIterationDetails(
            iteration_index=7,
            num_ctx_requests=1,
            num_ctx_tokens=512,
            num_generation_requests=2,
            num_generation_tokens=16,
            elapsed_ms=123.4,
        ),
        prefix_cache_stats=PrefixCacheStats(requests=5, queries=5120, hits=2560),
        spec_decoding_stats=SpecDecodingStats(
            num_spec_tokens=7,
            num_drafts=2,
            num_draft_tokens=14,
            num_accepted_tokens=6,
            num_accepted_tokens_per_pos=[3, 2, 1],
        ),
    )

    iteration_stats = IterationStats()
    iteration_stats.num_generation_tokens = 16
    iteration_stats.prompt_token_stats = PromptTokenStats(
        computed=256, local_cache_hit=256, external_kv_transfer=0, cached_tokens=256, total=512
    )
    iteration_stats.time_to_first_tokens_iter = [0.31, 0.29]
    iteration_stats.inter_token_latencies_iter = [0.052, 0.049, 0.061]
    iteration_stats.finished_requests = [
        FinishedRequestStats(
            finish_reason="stop",
            request_id="req-selftest-1",
            e2e_latency=1.85,
            num_prompt_tokens=512,
            num_preemptions=0,
            num_generation_tokens=24,
            max_tokens_param=128,
            queued_time=0.004,
            prefill_time=0.31,
            inference_time=1.54,
            decode_time=1.23,
            mean_time_per_output_token=0.052,
            is_corrupted=False,
            num_cached_tokens=256,
        )
    ]

    logger = ProbeStatLogger(vllm_config=None, engine_index=0)
    logger.log_engine_initialized()
    for _ in range(3):
        logger.record(scheduler_stats, iteration_stats)
    logger.close()

    check("record() 未产生失败", logger.failures == 0, f"failures={logger.failures}")
    check("已写出记录", logger.records_written == 3, f"written={logger.records_written}")

    print("\n== 3) JSONL 内容校验 ==")
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    check("文件非空", bool(lines))
    records = [row for row in lines if row.get("kind") is None]
    check("记录条数正确", len(records) == 3, f"{len(records)}")
    last = records[-1] if records else {}
    check("header 记录存在", any(row.get("kind") == "header" for row in lines))
    check("engine_initialized 记录存在", any(row.get("kind") == "engine_initialized" for row in lines))
    check("逐 step 批次构成", (last.get("iter") or {}).get("ctx_tokens") == 512, json.dumps(last.get("iter")))
    check("调度状态", (last.get("sched") or {}).get("running") == 3)
    check("KV 使用率", abs((last.get("sched") or {}).get("kv_cache_usage", 0) - 0.42) < 1e-9)
    check("前缀缓存", (last.get("prefix") or {}).get("hits") == 2560)
    spec = last.get("spec") or {}
    check("投机解码按位置", spec.get("num_accepted_tokens_per_pos") == [3, 2, 1], json.dumps(spec))
    check("投机解码接受率", abs((spec.get("draft_acceptance_rate") or 0) - 6 / 14) < 1e-9)
    reqs = last.get("reqs") or []
    check("逐请求生命周期", bool(reqs) and reqs[0].get("e2e_s") == 1.85, json.dumps(reqs[:1])[:200])
    check("逐请求缓存 token", bool(reqs) and reqs[0].get("cached_tokens") == 256)
    check("逐请求阶段拆分", bool(reqs) and reqs[0].get("prefill_s") == 0.31 and reqs[0].get("decode_s") == 1.23)
    samples = last.get("samples") or {}
    check("TTFT/ITL 样本", len(samples.get("ttft_s") or []) == 2 and len(samples.get("itl_s") or []) == 3)
    check("prompt token 来源", (last.get("prompt_tokens") or {}).get("local_cache_hit") == 256)

    print("\n== 4) 与当前 vLLM 版本字段一致性 ==")
    from vllm.v1.metrics.stats import SchedulerStats as _S

    verified = {
        "num_running_reqs", "num_waiting_reqs", "num_skipped_waiting_reqs", "kv_cache_usage",
        "step_counter", "current_wave", "iteration_details", "prefix_cache_stats",
        "spec_decoding_stats", "kv_cache_eviction_events", "cudagraph_stats", "perf_stats",
    }
    import dataclasses

    present = {f.name for f in dataclasses.fields(_S)}
    missing = sorted(verified - present)
    check("SchedulerStats 关键字段齐全", not missing, f"缺失={missing}")

    print()
    if FAILURES:
        print(f"❌ 自检失败 {len(FAILURES)} 项: {FAILURES}")
        return 1
    print("✅ 插件自检全部通过：entry point 可发现、真实统计对象可序列化、字段与当前版本一致")
    print(f"（样例产物：{log_path}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
