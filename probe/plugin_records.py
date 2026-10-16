"""读取/汇总 stat logger 插件产出的 ``probe_steps.jsonl``。

插件（``plugin_probe/``，通过 entry point 组 ``vllm.stat_logger_plugins``）把
每个 engine step 的 ``SchedulerStats`` / ``IterationStats`` 原样落成 JSONL；
本模块把它汇总成报告用的结构。

相对"解析日志"的增量：
* 逐请求的**服务端精确生命周期**（queue / prefill / decode / e2e / TPOT / 缓存命中 token / 抢占）
* KV block 驱逐事件、图调度统计、MFU 基础量（这些在日志里只有聚合后的文本）
* 逐步数组（TTFT / ITL / 按草稿位置的接受数）

用法::

    python3 -m probe plugin --file runs/x/probe_steps.jsonl --summary runs/x/probe_steps_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

# 复用日志层同一套分位数实现，保证报告里两个来源的数字口径一致
from .log_probe import _stats as stats_of


def load_records(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """读取 JSONL，返回 (step 记录列表, header 元信息)。文件不存在时返回 ([], {})。"""
    p = Path(path)
    if not p.exists():
        return [], {}
    records: list[dict[str, Any]] = []
    header: dict[str, Any] = {}
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if row.get("kind") == "header":
                header = row
            elif row.get("kind"):
                header.setdefault("events", []).append(row)  # type: ignore[union-attr]
            elif "step" in row:
                records.append(row)
    return records, header


def _ms(stats: dict[str, Any] | None) -> dict[str, Any]:
    """把"秒"统计换算成毫秒（报告统一用 ms）。"""
    if not stats or not stats.get("count"):
        return {"count": 0}
    out = {"count": stats["count"]}
    for key in ("mean", "p50", "p90", "p99", "min", "max"):
        value = stats.get(key)
        if value is not None:
            out[f"{key}_ms"] = value * 1000.0
    return out


def summarize(records: Iterable[dict[str, Any]], header: dict[str, Any] | None = None) -> dict[str, Any]:
    """把插件记录汇总成报告用的结构。"""
    records = list(records)
    elapsed: list[float] = []
    ctx_tokens = gen_tokens = idle = 0
    kv_usage: list[float] = []
    running: list[int] = []
    waiting: list[int] = []
    prefix_queries = prefix_hits = 0
    drafts = drafted = accepted = 0
    per_pos: dict[int, int] = {}
    graph_modes: dict[str, int] = {}
    unpadded = padded = paddings = 0
    evictions = 0
    lifetime: list[float] = []
    idle_before_evict: list[float] = []
    reuse_gaps: list[float] = []
    reqs: list[dict[str, Any]] = []
    ttft: list[float] = []
    itl: list[float] = []
    flops = 0.0
    read_bytes = write_bytes = 0.0
    span_start = span_end = None
    engines: set[int] = set()

    for record in records:
        ts = record.get("ts")
        if isinstance(ts, (int, float)):
            span_start = ts if span_start is None else min(span_start, ts)
            span_end = ts if span_end is None else max(span_end, ts)
        engine = record.get("engine")
        if isinstance(engine, int):
            engines.add(engine)

        iteration = record.get("iter") or {}
        if iteration:
            if iteration.get("elapsed_ms") is not None:
                elapsed.append(float(iteration["elapsed_ms"]))
            ctx = int(iteration.get("ctx_tokens") or 0)
            gen = int(iteration.get("gen_tokens") or 0)
            ctx_tokens += ctx
            gen_tokens += gen
            if ctx == 0 and gen == 0:
                idle += 1

        sched = record.get("sched") or {}
        if sched.get("kv_cache_usage") is not None:
            kv_usage.append(float(sched["kv_cache_usage"]))
        if sched.get("running") is not None:
            running.append(int(sched["running"]))
        if sched.get("waiting") is not None:
            waiting.append(int(sched["waiting"]))

        prefix = record.get("prefix") or {}
        prefix_queries += int(prefix.get("queries") or 0)
        prefix_hits += int(prefix.get("hits") or 0)

        spec = record.get("spec") or {}
        drafts += int(spec.get("num_drafts") or 0)
        drafted += int(spec.get("num_draft_tokens") or 0)
        accepted += int(spec.get("num_accepted_tokens") or 0)
        for index, value in enumerate(spec.get("num_accepted_tokens_per_pos") or []):
            per_pos[index] = per_pos.get(index, 0) + int(value)

        graph = record.get("graph") or {}
        if graph.get("runtime_mode"):
            mode = str(graph["runtime_mode"])
            count = graph_modes.get(mode, 0)
            graph_modes[mode] = count + 1
            unpadded += int(graph.get("unpadded_tokens") or 0)
            padded += int(graph.get("padded_tokens") or 0)
            paddings += int(graph.get("num_paddings") or 0)

        for event in record.get("evict") or []:
            evictions += 1
            if event.get("lifetime_s") is not None:
                lifetime.append(float(event["lifetime_s"]))
            if event.get("idle_s") is not None:
                idle_before_evict.append(float(event["idle_s"]))
            for gap in event.get("reuse_gaps_s") or []:
                reuse_gaps.append(float(gap))

        reqs.extend(record.get("reqs") or [])

        samples = record.get("samples") or {}
        ttft.extend(float(v) for v in (samples.get("ttft_s") or []))
        itl.extend(float(v) for v in (samples.get("itl_s") or []))

        perf = record.get("perf") or {}
        flops += float(perf.get("flops_per_gpu") or 0.0)
        read_bytes += float(perf.get("read_bytes_per_gpu") or 0.0)
        write_bytes += float(perf.get("write_bytes_per_gpu") or 0.0)

    wall = (span_end - span_start) if (span_start is not None and span_end is not None) else None
    reasons: dict[str, int] = {}
    for row in reqs:
        reason = str(row.get("finish_reason") or "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1

    summary: dict[str, Any] = {
        "source": "stat_logger_plugin",
        "header": header or {},
        "engines": sorted(engines),
        "steps": len(records),
        "wall_s": wall,
        "step_ms": stats_of(elapsed),
        "idle_steps": idle,
        "idle_ratio": (idle / len(records)) if records else None,
        "context_tokens": ctx_tokens,
        "generation_tokens": gen_tokens,
        "kv_cache_usage_perc": {
            "count": len(kv_usage),
            "mean": (sum(kv_usage) / len(kv_usage) * 100.0) if kv_usage else None,
            "max": (max(kv_usage) * 100.0) if kv_usage else None,
        },
        "running": stats_of(running),
        "waiting": stats_of(waiting),
        "prefix_cache_queries": prefix_queries,
        "prefix_cache_hits": prefix_hits,
        "prefix_cache_hit_rate": (prefix_hits / prefix_queries) if prefix_queries else None,
        "spec_decode": {
            "drafts": drafts,
            "draft_tokens": drafted,
            "accepted_tokens": accepted,
            "draft_acceptance_rate": (accepted / drafted) if drafted else None,
            "mean_acceptance_length": (1.0 + accepted / drafts) if drafts else None,
            "accepted_per_pos": dict(sorted(per_pos.items())),
        },
        "cudagraph": {
            "modes": graph_modes,
            "unpadded_tokens": unpadded,
            "padded_tokens": padded,
            "num_paddings": paddings,
            "padding_ratio": (padded / unpadded) if unpadded else None,
        },
        "kv_evictions": evictions,
        "kv_block_lifetime_ms": _ms(stats_of(lifetime)),
        "kv_block_idle_before_evict_ms": _ms(stats_of(idle_before_evict)),
        "kv_block_reuse_gap_ms": _ms(stats_of(reuse_gaps)),
        "finished_requests": len(reqs),
        "finish_reasons": reasons,
        "request_queue_ms": _ms(stats_of([r.get("queue_s") for r in reqs])),
        "request_prefill_ms": _ms(stats_of([r.get("prefill_s") for r in reqs])),
        "request_decode_ms": _ms(stats_of([r.get("decode_s") for r in reqs])),
        "request_inference_ms": _ms(stats_of([r.get("inference_s") for r in reqs])),
        "request_e2e_ms": _ms(stats_of([r.get("e2e_s") for r in reqs])),
        "request_tpot_ms": _ms(stats_of([r.get("tpot_s") for r in reqs])),
        "request_prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in reqs),
        "request_cached_tokens": sum(int(r.get("cached_tokens") or 0) for r in reqs),
        "request_generation_tokens": sum(int(r.get("generation_tokens") or 0) for r in reqs),
        "request_preemptions": sum(int(r.get("preemptions") or 0) for r in reqs),
        "request_corrupted": sum(1 for r in reqs if r.get("is_corrupted")),
        "ttft_ms": _ms(stats_of(ttft)),
        "itl_ms": _ms(stats_of(itl)),
        "mfu": {
            "flops_per_gpu_total": flops or None,
            "read_bytes_per_gpu_total": read_bytes or None,
            "write_bytes_per_gpu_total": write_bytes or None,
            "tflops_per_device": (flops / wall / 1e12) if (flops and wall) else None,
            "gbps_per_device": ((read_bytes + write_bytes) / wall / 1e9) if (wall and (read_bytes or write_bytes)) else None,
        },
    }
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    """人读摘要（CLI 打印）。"""
    def fmt(value: Any, digits: int = 1, unit: str = "") -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.{digits}f}{unit}"
        return f"{value}{unit}"

    lines = [
        f"来源: stat logger 插件（engine={summary.get('engines')}）",
        f"step 数={summary.get('steps')} 窗口={fmt(summary.get('wall_s'))}s "
        f"空转={summary.get('idle_steps')}（{fmt((summary.get('idle_ratio') or 0) * 100)}%）",
    ]
    step_ms = summary.get("step_ms") or {}
    if step_ms.get("count"):
        lines.append(
            f"step 耗时 ms: 均值={fmt(step_ms.get('mean'))} p90={fmt(step_ms.get('p90'))} "
            f"p99={fmt(step_ms.get('p99'))} max={fmt(step_ms.get('max'))}"
        )
    kv = summary.get("kv_cache_usage_perc") or {}
    lines.append(
        f"KV 使用率 %: 均值={fmt(kv.get('mean'))} max={fmt(kv.get('max'))}"
        f"  前缀缓存命中率={fmt((summary.get('prefix_cache_hit_rate') or 0) * 100, 1, '%')}"
    )
    spec = summary.get("spec_decode") or {}
    if spec.get("drafts"):
        lines.append(
            f"投机解码: 平均接受长度={fmt(spec.get('mean_acceptance_length'), 2)} "
            f"草稿接受率={fmt((spec.get('draft_acceptance_rate') or 0) * 100)}% "
            f"按位置={spec.get('accepted_per_pos')}"
        )
    if summary.get("finished_requests"):
        lines.append(
            f"逐请求服务端生命周期（{summary['finished_requests']} 个完成请求，"
            f"结束原因 {summary.get('finish_reasons')}）:"
        )
        for key, label in (
            ("request_queue_ms", "排队"),
            ("request_prefill_ms", "预填充"),
            ("request_decode_ms", "解码"),
            ("request_e2e_ms", "端到端"),
            ("request_tpot_ms", "每 token"),
        ):
            row = summary.get(key) or {}
            if row.get("count"):
                lines.append(
                    f"  {label}: p50={fmt(row.get('p50_ms'))}ms p90={fmt(row.get('p90_ms'))}ms "
                    f"均值={fmt(row.get('mean_ms'))}ms"
                )
        lines.append(
            f"  缓存命中 token={summary.get('request_cached_tokens')} / "
            f"prompt={summary.get('request_prompt_tokens')}，抢占={summary.get('request_preemptions')} 次"
        )
    graph = summary.get("cudagraph") or {}
    if graph.get("modes"):
        lines.append(
            f"图调度: {graph['modes']} padded/unpadded="
            f"{fmt(graph.get('padding_ratio'), 2)}×"
        )
    if summary.get("kv_evictions"):
        lines.append(
            f"KV block 驱逐={summary['kv_evictions']} 次，生命周期 "
            f"p50={fmt((summary.get('kv_block_lifetime_ms') or {}).get('p50_ms'))}ms"
        )
    mfu = summary.get("mfu") or {}
    if mfu.get("tflops_per_device"):
        lines.append(
            f"MFU: {fmt(mfu['tflops_per_device'], 2)} TF/s/device "
            f"{fmt(mfu.get('gbps_per_device'), 1)} GB/s/device"
        )
    return "\n".join(lines)
