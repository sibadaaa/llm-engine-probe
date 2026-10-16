"""把 vLLM 的 ``SchedulerStats`` / ``IterationStats`` 抽成可离线分析的扁平记录。

设计原则（很重要，vLLM 官方明确说这两个类**不是稳定接口**）：

1. **纯标准库**：本模块不 import vllm，因此可以在没有 GPU 的宿主机上直接单测
   （见仓库 ``tests/test_plugin_records.py``）。
2. **全部 getattr + 默认值**：任何字段缺失/改名都只导致该项为 ``None``，
   绝不抛异常、绝不影响引擎主循环。
3. **有界输出**：逐步数组（TTFT / ITL / 按位置接受数）截断到 ``MAX_SAMPLES``，
   避免高并发下 JSONL 无限膨胀。

字段名来自本机实测（vLLM 0.18.1+das，容器内 introspection）：
``SchedulerStats`` / ``SchedulerIterationDetails`` / ``PrefixCacheStats`` /
``SpecDecodingStats`` / ``KVCacheEvictionEvent`` / ``CUDAGraphStat`` / ``PerfStats`` /
``IterationStats`` / ``PromptTokenStats`` / ``FinishedRequestStats``。
"""

from __future__ import annotations

import time
from typing import Any, Iterable

SCHEMA_VERSION = 1

#: 逐步数组的最大保留长度（超出截断，只影响样本，不影响计数）
MAX_SAMPLES = 256

#: 每条记录里保留的"已完成请求"明细上限
MAX_FINISHED_REQUESTS = 512


# --------------------------------------------------------------------------
# 取值助手：任何异常都收敛成 None
# --------------------------------------------------------------------------


def _g(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    try:
        value = getattr(obj, name, default)
    except Exception:  # noqa: BLE001 - 统计代码绝不能抛
        return default
    return default if value is None else value


def _i(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _f(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _seq(value: Any) -> list[Any] | None:
    if value is None:
        return None
    try:
        return list(value)
    except TypeError:
        return None


def _floats(value: Any, limit: int = MAX_SAMPLES) -> list[float] | None:
    items = _seq(value)
    if items is None:
        return None
    out = [v for v in (_f(x) for x in items[:limit]) if v is not None]
    return out or None


def _ints(value: Any, limit: int = MAX_SAMPLES) -> list[int] | None:
    items = _seq(value)
    if items is None:
        return None
    out = [v for v in (_i(x) for x in items[:limit]) if v is not None]
    return out or None


def _drop_none(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}


# --------------------------------------------------------------------------
# 各分组抽取
# --------------------------------------------------------------------------


def scheduler_record(scheduler_stats: Any) -> dict[str, Any] | None:
    """引擎/调度器瞬时状态（每个 step 一份）。"""
    if scheduler_stats is None:
        return None
    data = {
        "running": _i(_g(scheduler_stats, "num_running_reqs")),
        "waiting": _i(_g(scheduler_stats, "num_waiting_reqs")),
        "waiting_skipped": _i(_g(scheduler_stats, "num_skipped_waiting_reqs")),
        "kv_cache_usage": _f(_g(scheduler_stats, "kv_cache_usage")),
        "step_counter": _i(_g(scheduler_stats, "step_counter")),
        "wave": _i(_g(scheduler_stats, "current_wave")),
    }
    waiting_lora = _g(scheduler_stats, "waiting_lora_adapters")
    running_lora = _g(scheduler_stats, "running_lora_adapters")
    if waiting_lora:
        data["waiting_lora_adapters"] = dict(waiting_lora)
    if running_lora:
        data["running_lora_adapters"] = dict(running_lora)
    return _drop_none(data)


def iteration_details_record(scheduler_stats: Any) -> dict[str, Any] | None:
    """逐 step 批次构成与耗时（等价于日志里的 ``Iteration(n): ...`` 行）。

    注意：``SchedulerStats.iteration_details`` 只有在
    ``--enable-logging-iteration-details`` 且 log stats 开启时才非空。
    """
    details = _g(scheduler_stats, "iteration_details")
    if details is None:
        return None
    return _drop_none(
        {
            "iteration_index": _i(_g(details, "iteration_index")),
            "ctx_requests": _i(_g(details, "num_ctx_requests")),
            "ctx_tokens": _i(_g(details, "num_ctx_tokens")),
            "gen_requests": _i(_g(details, "num_generation_requests")),
            "gen_tokens": _i(_g(details, "num_generation_tokens")),
            "elapsed_ms": _f(_g(details, "elapsed_ms")),
            "encoder_inputs": _i(_g(details, "num_encoder_inputs")),
            "encoder_output_tokens": _i(_g(details, "num_encoder_output_tokens")),
            "is_dummy": bool(_g(details, "is_dummy", False)) or None,
        }
    )


def _cache_record(stats: Any) -> dict[str, Any] | None:
    if stats is None:
        return None
    data = {
        "requests": _i(_g(stats, "requests")),
        "queries": _i(_g(stats, "queries")),
        "hits": _i(_g(stats, "hits")),
        "preempted_requests": _i(_g(stats, "preempted_requests")),
        "preempted_queries": _i(_g(stats, "preempted_queries")),
        "preempted_hits": _i(_g(stats, "preempted_hits")),
    }
    data = _drop_none(data)
    return data or None


def prefix_cache_record(scheduler_stats: Any) -> dict[str, Any] | None:
    return _cache_record(_g(scheduler_stats, "prefix_cache_stats"))


def connector_prefix_cache_record(scheduler_stats: Any) -> dict[str, Any] | None:
    return _cache_record(_g(scheduler_stats, "connector_prefix_cache_stats"))


def spec_decode_record(scheduler_stats: Any) -> dict[str, Any] | None:
    """投机解码：本区间累计 + 按草稿位置的接受分布。"""
    stats = _g(scheduler_stats, "spec_decoding_stats")
    if stats is None:
        return None
    accepted = _i(_g(stats, "num_accepted_tokens"))
    drafted = _i(_g(stats, "num_draft_tokens"))
    drafts = _i(_g(stats, "num_drafts"))
    data: dict[str, Any] = {
        "num_spec_tokens": _i(_g(stats, "num_spec_tokens")),
        "num_drafts": drafts,
        "num_draft_tokens": drafted,
        "num_accepted_tokens": accepted,
        "num_accepted_tokens_per_pos": _ints(_g(stats, "num_accepted_tokens_per_pos")),
        "num_draft_tokens_per_pos": _ints(_g(stats, "num_draft_tokens_per_pos")),
    }
    if drafted:
        data["draft_acceptance_rate"] = accepted / drafted if accepted is not None else None
    if drafts:
        data["mean_acceptance_length"] = 1.0 + (accepted / drafts if accepted is not None else 0.0)
    return _drop_none(data)


def kv_evictions_record(scheduler_stats: Any) -> list[dict[str, Any]]:
    """KV block 驱逐事件（需要 ``--kv-cache-metrics``）。"""
    events = _seq(_g(scheduler_stats, "kv_cache_eviction_events"))
    if not events:
        return []
    out: list[dict[str, Any]] = []
    for event in events[:MAX_SAMPLES]:
        gaps = _floats(_g(event, "reuse_gaps_seconds"))
        out.append(
            _drop_none(
                {
                    "lifetime_s": _f(_g(event, "lifetime_seconds")),
                    "idle_s": _f(_g(event, "idle_seconds")),
                    "reuse_gaps_s": gaps,
                }
            )
        )
    return out


def cudagraph_record(scheduler_stats: Any) -> dict[str, Any] | None:
    """图调度统计（需要 ``--cudagraph-metrics``）。"""
    stats = _g(scheduler_stats, "cudagraph_stats")
    if stats is None:
        return None
    return _drop_none(
        {
            "unpadded_tokens": _i(_g(stats, "num_unpadded_tokens")),
            "padded_tokens": _i(_g(stats, "num_padded_tokens")),
            "num_paddings": _i(_g(stats, "num_paddings")),
            "runtime_mode": _g(stats, "runtime_mode"),
        }
    )


def perf_record(scheduler_stats: Any) -> dict[str, Any] | None:
    """MFU 基础量（需要 ``--enable-mfu-metrics``）。"""
    stats = _g(scheduler_stats, "perf_stats")
    if stats is None:
        return None
    return _drop_none(
        {
            "flops_per_gpu": _f(_g(stats, "num_flops_per_gpu")),
            "read_bytes_per_gpu": _f(_g(stats, "num_read_bytes_per_gpu")),
            "write_bytes_per_gpu": _f(_g(stats, "num_write_bytes_per_gpu")),
        }
    )


def prompt_token_record(iteration_stats: Any) -> dict[str, Any] | None:
    stats = _g(iteration_stats, "prompt_token_stats")
    if stats is None:
        return None
    return _drop_none(
        {
            "computed": _i(_g(stats, "computed")),
            "local_cache_hit": _i(_g(stats, "local_cache_hit")),
            "external_kv_transfer": _i(_g(stats, "external_kv_transfer")),
            "cached_tokens": _i(_g(stats, "cached_tokens")),
            "total": _i(_g(stats, "total")),
        }
    )


def finished_requests_record(iteration_stats: Any) -> list[dict[str, Any]]:
    """本 step 内**完成**的请求：服务端视角的完整生命周期明细。

    这是本插件相对"解析日志"最大的增量：每个请求都有精确的
    queue / prefill / decode / e2e / TPOT，以及缓存命中 token 与抢占次数。
    """
    requests = _seq(_g(iteration_stats, "finished_requests"))
    if not requests:
        return []
    out: list[dict[str, Any]] = []
    for req in requests[:MAX_FINISHED_REQUESTS]:
        out.append(
            _drop_none(
                {
                    "id": _g(req, "request_id"),
                    "finish_reason": _g(req, "finish_reason"),
                    "prompt_tokens": _i(_g(req, "num_prompt_tokens")),
                    "cached_tokens": _i(_g(req, "num_cached_tokens")),
                    "generation_tokens": _i(_g(req, "num_generation_tokens")),
                    "preemptions": _i(_g(req, "num_preemptions")),
                    "queue_s": _f(_g(req, "queued_time")),
                    "prefill_s": _f(_g(req, "prefill_time")),
                    "decode_s": _f(_g(req, "decode_time")),
                    "inference_s": _f(_g(req, "inference_time")),
                    "e2e_s": _f(_g(req, "e2e_latency")),
                    "tpot_s": _f(_g(req, "mean_time_per_output_token")),
                    "max_tokens_param": _i(_g(req, "max_tokens_param")),
                    "is_corrupted": True if _g(req, "is_corrupted", False) else None,
                }
            )
        )
    return out


def iteration_samples_record(iteration_stats: Any) -> dict[str, Any] | None:
    """本 step 观测到的 TTFT / ITL 样本（有界）。"""
    if iteration_stats is None:
        return None
    data = {
        "ttft_s": _floats(_g(iteration_stats, "time_to_first_tokens_iter")),
        "itl_s": _floats(_g(iteration_stats, "inter_token_latencies_iter")),
        "generation_tokens": _i(_g(iteration_stats, "num_generation_tokens")),
        "preempted_reqs": _i(_g(iteration_stats, "num_preempted_reqs")),
        "max_generation_tokens_iter": _i(_g(iteration_stats, "max_num_generation_tokens_iter")),
        "corrupted_reqs": _i(_g(iteration_stats, "num_corrupted_reqs")),
    }
    return _drop_none(data)


# --------------------------------------------------------------------------
# 顶层记录
# --------------------------------------------------------------------------


def build_record(
    scheduler_stats: Any,
    iteration_stats: Any,
    engine_index: int = 0,
    step: int = 0,
    ts: float | None = None,
) -> dict[str, Any]:
    """合成一条 step 记录（这是写进 JSONL 的内容）。"""
    record: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "engine": int(engine_index),
        "step": int(step),
        "ts": time.time() if ts is None else float(ts),
    }
    if iteration_stats is not None:
        record["ts_iter"] = _f(_g(iteration_stats, "iteration_timestamp"))
    for key, value in (
        ("sched", scheduler_record(scheduler_stats)),
        ("iter", iteration_details_record(scheduler_stats)),
        ("prefix", prefix_cache_record(scheduler_stats)),
        ("connector_prefix", connector_prefix_cache_record(scheduler_stats)),
        ("spec", spec_decode_record(scheduler_stats)),
        ("graph", cudagraph_record(scheduler_stats)),
        ("perf", perf_record(scheduler_stats)),
        ("prompt_tokens", prompt_token_record(iteration_stats)),
        ("samples", iteration_samples_record(iteration_stats)),
    ):
        if value:
            record[key] = value
    evictions = kv_evictions_record(scheduler_stats)
    if evictions:
        record["evict"] = evictions
    reqs = finished_requests_record(iteration_stats)
    if reqs:
        record["reqs"] = reqs
    return record


# --------------------------------------------------------------------------
# 轻量汇总（容器内自检用；主工具箱有更完整的 probe/plugin_records.py）
# --------------------------------------------------------------------------


def _stats(values: Iterable[float]) -> dict[str, Any]:
    data = sorted(v for v in values if v is not None)
    if not data:
        return {"count": 0}
    def pct(q: float) -> float:
        if len(data) == 1:
            return data[0]
        pos = (len(data) - 1) * q
        low = int(pos)
        high = min(low + 1, len(data) - 1)
        return data[low] + (data[high] - data[low]) * (pos - low)
    return {
        "count": len(data),
        "mean": sum(data) / len(data),
        "p50": pct(0.5),
        "p90": pct(0.9),
        "p99": pct(0.99),
        "max": data[-1],
    }


def summarise(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """把记录流汇总成一份便于人读的字典（容器内 ``python3 -m ... --summary``）。"""
    steps = 0
    elapsed: list[float] = []
    ctx_tokens = gen_tokens = idle = 0
    kv_usage: list[float] = []
    running: list[int] = []
    waiting: list[int] = []
    spec_drafts = spec_drafted = spec_accepted = 0
    per_pos: dict[int, int] = {}
    graph: dict[str, int] = {}
    evictions = 0
    lifetime: list[float] = []
    reqs: list[dict[str, Any]] = []
    ttft: list[float] = []
    itl: list[float] = []
    flops = 0.0
    span_start = span_end = None

    for record in records:
        steps += 1
        ts = record.get("ts")
        if ts is not None:
            span_start = ts if span_start is None else min(span_start, ts)
            span_end = ts if span_end is None else max(span_end, ts)
        iteration = record.get("iter") or {}
        if iteration:
            if iteration.get("elapsed_ms") is not None:
                elapsed.append(iteration["elapsed_ms"])
            ctx = iteration.get("ctx_tokens") or 0
            gen = iteration.get("gen_tokens") or 0
            ctx_tokens += ctx
            gen_tokens += gen
            if ctx == 0 and gen == 0:
                idle += 1
        sched = record.get("sched") or {}
        if sched.get("kv_cache_usage") is not None:
            kv_usage.append(sched["kv_cache_usage"])
        if sched.get("running") is not None:
            running.append(sched["running"])
        if sched.get("waiting") is not None:
            waiting.append(sched["waiting"])
        spec = record.get("spec") or {}
        spec_drafts += spec.get("num_drafts") or 0
        spec_drafted += spec.get("num_draft_tokens") or 0
        spec_accepted += spec.get("num_accepted_tokens") or 0
        for index, value in enumerate(spec.get("num_accepted_tokens_per_pos") or []):
            per_pos[index] = per_pos.get(index, 0) + value
        graph_row = record.get("graph") or {}
        if graph_row.get("runtime_mode"):
            graph[graph_row["runtime_mode"]] = graph.get(graph_row["runtime_mode"], 0) + 1
        for event in record.get("evict") or []:
            evictions += 1
            if event.get("lifetime_s") is not None:
                lifetime.append(event["lifetime_s"])
        reqs.extend(record.get("reqs") or [])
        samples = record.get("samples") or {}
        ttft.extend(samples.get("ttft_s") or [])
        itl.extend(samples.get("itl_s") or [])
        perf = record.get("perf") or {}
        flops += perf.get("flops_per_gpu") or 0.0

    wall = (span_end - span_start) if (span_start and span_end) else None
    return {
        "schema": SCHEMA_VERSION,
        "steps": steps,
        "wall_s": wall,
        "step_ms": _stats(elapsed),
        "context_tokens": ctx_tokens,
        "generation_tokens": gen_tokens,
        "idle_steps": idle,
        "kv_cache_usage": _stats(kv_usage),
        "running": _stats(running),
        "waiting": _stats(waiting),
        "spec_decode": {
            "drafts": spec_drafts,
            "draft_tokens": spec_drafted,
            "accepted_tokens": spec_accepted,
            "draft_acceptance_rate": (spec_accepted / spec_drafted) if spec_drafted else None,
            "mean_acceptance_length": (1 + spec_accepted / spec_drafts) if spec_drafts else None,
            "accepted_per_pos": per_pos,
        },
        "cudagraph_modes": graph,
        "kv_evictions": evictions,
        "kv_block_lifetime_s": _stats(lifetime),
        "finished_requests": len(reqs),
        "finished_finish_reasons": _count_by(reqs, "finish_reason"),
        "request_queue_s": _stats([r.get("queue_s") for r in reqs]),
        "request_prefill_s": _stats([r.get("prefill_s") for r in reqs]),
        "request_decode_s": _stats([r.get("decode_s") for r in reqs]),
        "request_e2e_s": _stats([r.get("e2e_s") for r in reqs]),
        "request_tpot_s": _stats([r.get("tpot_s") for r in reqs]),
        "request_cached_tokens": sum(r.get("cached_tokens") or 0 for r in reqs),
        "request_prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in reqs),
        "request_generation_tokens": sum(r.get("generation_tokens") or 0 for r in reqs),
        "request_preemptions": sum(r.get("preemptions") or 0 for r in reqs),
        "ttft_s": _stats(ttft),
        "itl_s": _stats(itl),
        "flops_per_gpu_total": flops or None,
    }


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        name = str(row.get(key) or "unknown")
        out[name] = out.get(name, 0) + 1
    return out
