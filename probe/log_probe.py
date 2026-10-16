"""引擎 step 层探测：把 vLLM 的日志翻译成"逐个 iteration 的结构化事件"。

这个模块回答的问题：**"引擎每一次 step 在干什么？调度器/缓存/投机解码的状态如何变化？"**

它解析 vLLM 0.18.x 在以下开关下产生的日志（本仓库实测的真实格式）：

``--enable-logging-iteration-details``  每个 engine step 一行::

    Engine 000: Iteration(752): 0 context requests, 0 context tokens,
    1 generation requests, 8 generation tokens, iteration elapsed time: 184.79 ms,
    GPU KV cache usage: 3.5%

周期统计（默认开启，受 ``VLLM_LOG_STATS_INTERVAL`` 控制）::

    Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 8.0 tokens/s,
    Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 27.0%

投机解码::

    SpecDecoding metrics: Mean acceptance length: 1.69, ... Per-position acceptance rate: 0.462, ...

MFU（``--enable-mfu-metrics``）::

    Engine 000: MFU: 0.6 TF/s/device 32.4 GB/s/device

CUDA/HIP Graph 统计（``--cudagraph-metrics``）以 Markdown 表格形式打印，
解析成 ``cudagraph_row`` 事件。

只依赖标准库。用法::

    python3 -m probe logs --file serve.log --live          # 实时看 step 流
    python3 -m probe logs --file serve.log --jsonl steps.jsonl --summary steps_summary.json
    docker logs -f vllm-probe | python3 -m probe logs --stdin --live
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

# --------------------------------------------------------------------------
# 行前缀解析
# --------------------------------------------------------------------------

_PREFIX_RE = re.compile(
    r"^(?:\((?P<proc>[^)]*)\)\s*)?"
    r"(?P<level>[A-Z]+)\s+"
    r"(?P<ts>\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[.,]\d+)?|\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)\s+"
    r"\[(?P<loc>[^\]]*)\]\s+"
    r"(?P<msg>.*)$"
)
_UVICORN_RE = re.compile(r"^(?P<level>INFO|WARNING|ERROR):\s+(?P<msg>.*)$")

_ITER_RE = re.compile(
    r"Iteration\((?P<iteration>\d+)\):\s*"
    r"(?P<ctx_reqs>\d+) context requests,\s*"
    r"(?P<ctx_tokens>\d+) context tokens,\s*"
    r"(?P<gen_reqs>\d+) generation requests,\s*"
    r"(?P<gen_tokens>\d+) generation tokens,\s*"
    r"iteration elapsed time:\s*(?P<elapsed_ms>[\d.]+)\s*ms"
    r"(?P<tail>.*)$"
)
_KV_USAGE_RE = re.compile(r"([A-Za-z_ ]+?) KV cache usage:\s*([\d.]+)%")
_ENCODER_RE = re.compile(
    r"encoder inputs:\s*(\d+),\s*encoder output embeddings:\s*(\d+)"
)

_STATS_RE = re.compile(
    r"Avg prompt throughput:\s*(?P<prompt_tps>[\d.]+)\s*tokens/s,\s*"
    r"Avg generation throughput:\s*(?P<generation_tps>[\d.]+)\s*tokens/s,\s*"
    r"Running:\s*(?P<running>\d+)\s*reqs,\s*"
    r"Waiting:\s*(?P<waiting>\d+)\s*reqs,\s*"
    r".*?KV cache usage:\s*(?P<kv_usage>[\d.]+)%"
    r"(?P<tail>.*)$"
)
_HIT_RATE_RE = re.compile(r"Prefix cache hit rate:\s*([\d.]+)%")
_EXT_HIT_RATE_RE = re.compile(r"External prefix cache hit rate:\s*([\d.]+)%")
_MM_HIT_RATE_RE = re.compile(r"MM cache hit rate:\s*([\d.]+)%")
_NANS_RE = re.compile(r"Corrupted:\s*(\d+)\s*reqs")

_SPEC_RE = re.compile(
    r"SpecDecoding metrics:\s*"
    r"Mean acceptance length:\s*(?P<mean_acceptance>[\d.]+),\s*"
    r"Accepted throughput:\s*(?P<accepted_tps>[\d.]+)\s*tokens/s,\s*"
    r"Drafted throughput:\s*(?P<drafted_tps>[\d.]+)\s*tokens/s,\s*"
    r"Accepted:\s*(?P<accepted_tokens>\d+)\s*tokens,\s*"
    r"Drafted:\s*(?P<drafted_tokens>\d+)\s*tokens,\s*"
    r"Per-position acceptance rate:\s*(?P<per_pos>[0-9.,\s]+?),\s*"
    r"Avg Draft acceptance rate:\s*(?P<avg_acceptance>[\d.]+)%"
)

_MFU_RE = re.compile(
    r"MFU:\s*(?P<tflops>[\d.]+)\s*TF/s/device\s*(?P<gbps>[\d.]+)\s*GB/s/device"
)

_CG_ROW_RE = re.compile(
    r"^\|\s*(?P<unpadded>\d+|Unpadded Tokens)\s*\|\s*(?P<padded>\d+|Padded Tokens)\s*\|"
    r"\s*(?P<paddings>\d+|Num Paddings)\s*\|\s*(?P<mode>[A-Z_]+|Runtime Mode)\s*\|"
    r"\s*(?P<count>\d+|Count)\s*\|$"
)
_CG_MODE_RE = re.compile(r"^-\s*Mode:\s*(\S+)")
_CG_CAPTURE_RE = re.compile(r"^-\s*Capture sizes:\s*(\[.*\])")

_REQUEST_INFO_RE = re.compile(
    r"(Received request[^:]*:\s*(?P<rid>\S+)|Added request\s*(?P<rid2>\S+)|"
    r"Request (?P<rid3>[0-9a-fA-F-]{8,}) details)"
)


# --------------------------------------------------------------------------
# 事件模型
# --------------------------------------------------------------------------


@dataclass
class LogEvent:
    kind: str
    raw: str
    ts: float | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "ts": self.ts, "data": self.data, "raw": self.raw}


def _parse_ts(raw: str | None) -> float | None:
    if not raw:
        return None
    text = raw.replace(",", ".")
    for fmt in ("%m-%d %H:%M:%S.%f", "%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = time.strptime(text, fmt)
            if parsed.tm_year == 1900:
                parsed = time.struct_time(
                    (time.localtime().tm_year,) + tuple(parsed)[1:]
                )
            return time.mktime(parsed)
        except ValueError:
            continue
    return None


def parse_line(line: str, cudagraph_active: bool = False) -> LogEvent | None:
    """解析一行日志，返回事件或 ``None``（无信息量的行）。"""
    line = line.rstrip("\n")
    if not line.strip():
        return None
    m = _PREFIX_RE.match(line)
    if m:
        msg = m.group("msg")
        ts = _parse_ts(m.group("ts"))
        loc = m.group("loc") or ""
    else:
        m2 = _UVICORN_RE.match(line)
        if m2:
            msg, ts, loc = m2.group("msg"), None, "uvicorn"
        else:
            msg, ts, loc = line, None, ""

    # 1) 逐 iteration 明细
    m = _ITER_RE.search(msg)
    if m:
        tail = m.group("tail") or ""
        data: dict[str, Any] = {
            "iteration": int(m.group("iteration")),
            "ctx_reqs": int(m.group("ctx_reqs")),
            "ctx_tokens": int(m.group("ctx_tokens")),
            "gen_reqs": int(m.group("gen_reqs")),
            "gen_tokens": int(m.group("gen_tokens")),
            "elapsed_ms": float(m.group("elapsed_ms")),
        }
        kv = _KV_USAGE_RE.search(tail)
        if kv:
            data["kv_usage_perc"] = float(kv.group(2))
        enc = _ENCODER_RE.search(tail)
        if enc:
            data["encoder_inputs"] = int(enc.group(1))
            data["encoder_output_tokens"] = int(enc.group(2))
        return LogEvent("iteration", line, ts, data)

    # 2) 周期统计
    m = _STATS_RE.search(msg)
    if m and "Avg prompt throughput" in msg:
        tail = m.group("tail") or ""
        data = {
            "prompt_tps": float(m.group("prompt_tps")),
            "generation_tps": float(m.group("generation_tps")),
            "running": int(m.group("running")),
            "waiting": int(m.group("waiting")),
            "kv_usage_perc": float(m.group("kv_usage")),
        }
        hit = _HIT_RATE_RE.search(tail)
        if hit:
            data["prefix_cache_hit_rate"] = float(hit.group(1))
        ext = _EXT_HIT_RATE_RE.search(tail)
        if ext:
            data["external_prefix_cache_hit_rate"] = float(ext.group(1))
        mm = _MM_HIT_RATE_RE.search(tail)
        if mm:
            data["mm_cache_hit_rate"] = float(mm.group(1))
        nans = _NANS_RE.search(tail)
        if nans:
            data["corrupted_reqs"] = int(nans.group(1))
        return LogEvent("stats", line, ts, data)

    # 3) 投机解码
    m = _SPEC_RE.search(msg)
    if m:
        per_pos = [
            float(x)
            for x in re.split(r"[,\s]+", m.group("per_pos").strip())
            if x
        ]
        return LogEvent(
            "spec_decode",
            line,
            ts,
            {
                "mean_acceptance_length": float(m.group("mean_acceptance")),
                "accepted_tps": float(m.group("accepted_tps")),
                "drafted_tps": float(m.group("drafted_tps")),
                "accepted_tokens": int(m.group("accepted_tokens")),
                "drafted_tokens": int(m.group("drafted_tokens")),
                "per_position_acceptance": per_pos,
                "avg_draft_acceptance_rate": float(m.group("avg_acceptance")),
            },
        )

    # 4) MFU
    m = _MFU_RE.search(msg)
    if m:
        engine = _engine_of(msg)
        return LogEvent(
            "mfu",
            line,
            ts,
            {"engine": engine, "tflops_per_device": float(m.group("tflops")),
             "gbps_per_device": float(m.group("gbps"))},
        )

    # 5) CUDA/HIP Graph 统计表
    core = msg.strip()
    if "CUDAGraph Stats" in core:
        return LogEvent("cudagraph_header", line, ts, {"section": "stats"})
    if "CUDAGraph Config Settings" in core:
        return LogEvent("cudagraph_header", line, ts, {"section": "config"})
    m = _CG_ROW_RE.match(core)
    if m and m.group("unpadded").isdigit():
        return LogEvent(
            "cudagraph_row",
            line,
            ts,
            {
                "unpadded_tokens": int(m.group("unpadded")),
                "padded_tokens": int(m.group("padded")),
                "num_paddings": int(m.group("paddings")),
                "runtime_mode": m.group("mode"),
                "count": int(m.group("count")),
            },
        )
    m = _CG_MODE_RE.match(core)
    if m:
        return LogEvent("cudagraph_config", line, ts, {"mode": m.group(1)})
    m = _CG_CAPTURE_RE.match(core)
    if m:
        return LogEvent("cudagraph_config", line, ts, {"capture_sizes": m.group(1)})

    # 6) 请求日志（--enable-log-requests）
    m = _REQUEST_INFO_RE.search(msg)
    if m:
        rid = m.group("rid") or m.group("rid2") or m.group("rid3")
        return LogEvent("request", line, ts, {"request_id": rid})

    if core.startswith('"GET ') or core.startswith('"POST '):
        return LogEvent("access", line, ts, {})

    return None


def _engine_of(msg: str) -> int | None:
    m = re.match(r"Engine (\d+):", msg)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------


def iter_events(lines: Iterable[str]) -> Iterator[LogEvent]:
    for line in lines:
        event = parse_line(line)
        if event is not None:
            yield event


def read_events(path: str | Path) -> list[LogEvent]:
    with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
        return list(iter_events(fh))


def follow_file(path: str | Path, poll_s: float = 0.5) -> Iterator[str]:
    """类似 ``tail -f``：先读完已存在内容，再持续跟随新行。"""
    p = Path(path)
    while not p.exists():
        time.sleep(poll_s)
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            line = fh.readline()
            if line:
                yield line
                continue
            time.sleep(poll_s)


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": _pct(ordered, 50),
        "p90": _pct(ordered, 90),
        "p99": _pct(ordered, 99),
        "min": ordered[0],
        "max": ordered[-1],
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _pct(ordered: list[float], q: float) -> float:
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def summarize(events: list[LogEvent], slow_factor: float = 1.5) -> dict[str, Any]:
    """把事件流汇总成报告需要的结构。"""
    iters = [e for e in events if e.kind == "iteration"]
    stats_rows = [e for e in events if e.kind == "stats"]
    spec_rows = [e for e in events if e.kind == "spec_decode"]
    mfu_rows = [e for e in events if e.kind == "mfu"]
    cg_rows = [e for e in events if e.kind == "cudagraph_row"]

    # ---- step 时间与批次构成 -------------------------------------------
    pure_prefill = [
        e for e in iters
        if e.data["ctx_tokens"] > 0 and e.data["gen_tokens"] == 0
    ]
    decode_steps = [
        e for e in iters
        if e.data["ctx_tokens"] == 0 and e.data["gen_tokens"] > 0
    ]
    mixed_steps = [e for e in iters if e.data["ctx_tokens"] > 0 and e.data["gen_tokens"] > 0]
    prefill_steps = pure_prefill + mixed_steps
    idle_steps = [e for e in iters if e.data["ctx_tokens"] == 0 and e.data["gen_tokens"] == 0]

    def step_stats(subset: list[LogEvent]) -> dict[str, Any]:
        out = _stats([e.data["elapsed_ms"] for e in subset])
        tokens = [e.data["ctx_tokens"] + e.data["gen_tokens"] for e in subset]
        out["tokens_per_step"] = statistics.fmean(tokens) if tokens else None
        out["gen_reqs_per_step"] = (
            statistics.fmean([e.data["gen_reqs"] for e in subset]) if subset else None
        )
        return out

    elapsed = [e.data["elapsed_ms"] for e in iters]
    regime = _regime(elapsed, slow_factor)
    ctx_tokens_all = sum(e.data["ctx_tokens"] for e in iters)
    gen_tokens_all = sum(e.data["gen_tokens"] for e in iters)

    # ---- 预填充效率 -----------------------------------------------------
    # 只用"纯预填充 step"计算 ms/1k token；混合 step（chunked prefill 与 decode 同批）
    # 的时间无法干净拆分，单独列出，避免把解码开销算进预填充。
    prefill_pool = pure_prefill or prefill_steps
    ctx_tokens_pool = sum(e.data["ctx_tokens"] for e in prefill_pool)
    prefill_ms = sum(e.data["elapsed_ms"] for e in prefill_pool)
    prefill_ms_per_1k = (
        prefill_ms / ctx_tokens_pool * 1000.0 if ctx_tokens_pool > 0 else None
    )
    decode_ms_per_step = statistics.fmean(
        [e.data["elapsed_ms"] for e in decode_steps]
    ) if decode_steps else None
    decode_tokens = sum(e.data["gen_tokens"] for e in decode_steps)

    summary: dict[str, Any] = {
        "events": len(events),
        "counts": {
            "iteration": len(iters),
            "stats": len(stats_rows),
            "spec_decode": len(spec_rows),
            "mfu": len(mfu_rows),
            "cudagraph_row": len(cg_rows),
            "request": sum(1 for e in events if e.kind == "request"),
            "access": sum(1 for e in events if e.kind == "access"),
        },
        "window": _window(iters),
        "steps": {
            "total": len(iters),
            "prefill_only": len(pure_prefill),
            "decode_only": len(decode_steps),
            "mixed": len(mixed_steps),
            "idle": len(idle_steps),
            "idle_ratio": (len(idle_steps) / len(iters)) if iters else None,
        },
        "step_ms": _stats(elapsed),
        "step_ms_by_kind": {
            "prefill": step_stats(prefill_pool),
            "decode": step_stats(decode_steps),
            "mixed": step_stats(mixed_steps),
            "idle": step_stats(idle_steps),
        },
        "efficiency": {
            "context_tokens_total": ctx_tokens_all,
            "generation_tokens_total": gen_tokens_all,
            "generation_tokens_pure_decode": decode_tokens,
            "prefill_basis": "pure_prefill" if pure_prefill else "all_context_steps",
            "prefill_ms_per_1k_context_tokens": prefill_ms_per_1k,
            "decode_ms_per_step": decode_ms_per_step,
            "decode_ms_per_token": (
                decode_ms_per_step / (decode_tokens / len(decode_steps))
                if decode_steps and decode_tokens else None
            ),
            "tokens_per_step_mean": (
                (ctx_tokens_all + gen_tokens_all) / len(iters) if iters else None
            ),
        },
        "regimes": regime,
        "kv_usage_perc": _stats(
            [e.data["kv_usage_perc"] for e in iters if "kv_usage_perc" in e.data]
        ),
        "stats_rows": [
            {"ts": e.ts, **e.data} for e in stats_rows[-200:]
        ],
        "stats_aggregate": _stats_aggregate(stats_rows),
        "spec_decode": _spec_aggregate(spec_rows),
        "mfu": _mfu_aggregate(mfu_rows),
        "cudagraph": _cudagraph_aggregate(cg_rows),
        "timeline": _timeline(iters),
    }
    return summary


def _window(iters: list[LogEvent]) -> dict[str, Any]:
    ts = [e.ts for e in iters if e.ts]
    if not ts:
        return {}
    return {"start": min(ts), "end": max(ts), "duration_s": max(ts) - min(ts)}


def _regime(elapsed: list[float], slow_factor: float) -> dict[str, Any]:
    """双/多稳态检测：把 step 时间分成"快/慢"两态。

    这套部署历史上出现过 192ms vs 373ms 的双稳态，因此把稳态占比作为一等
    输出：如果慢态占比高且与批次大小无关，问题通常在算子/通信而不是调度。
    """
    if len(elapsed) < 10:
        return {"available": False, "reason": "样本太少(<10 step)"}
    ordered = sorted(elapsed)
    median = _pct(ordered, 50)
    fast = [v for v in elapsed if v <= median * slow_factor]
    slow = [v for v in elapsed if v > median * slow_factor]
    out: dict[str, Any] = {
        "available": True,
        "median_ms": median,
        "slow_threshold_ms": median * slow_factor,
        "slow_ratio": len(slow) / len(elapsed),
        "slow_stats": _stats(slow),
        "fast_stats": _stats(fast),
        "hint": "慢态占比 >20% 时优先查：批次构成突变 / KV 换出 / 通信等待 / 时钟降频",
    }
    third = max(1, len(elapsed) // 3)
    out["drift"] = {
        "first_third_mean_ms": statistics.fmean(elapsed[:third]),
        "last_third_mean_ms": statistics.fmean(elapsed[-third:]),
    }
    return out


def _stats_aggregate(rows: list[LogEvent]) -> dict[str, Any]:
    if not rows:
        return {"available": False}
    return {
        "available": True,
        "samples": len(rows),
        "prompt_tps": _stats([e.data["prompt_tps"] for e in rows]),
        "generation_tps": _stats([e.data["generation_tps"] for e in rows]),
        "running": _stats([e.data["running"] for e in rows]),
        "waiting": _stats([e.data["waiting"] for e in rows]),
        "kv_usage_perc": _stats([e.data["kv_usage_perc"] for e in rows]),
        "prefix_cache_hit_rate": _stats(
            [e.data["prefix_cache_hit_rate"] for e in rows if "prefix_cache_hit_rate" in e.data]
        ),
    }


def _spec_aggregate(rows: list[LogEvent]) -> dict[str, Any]:
    if not rows:
        return {"available": False}
    last = rows[-1].data
    return {
        "available": True,
        "samples": len(rows),
        "mean_acceptance_length": _stats([e.data["mean_acceptance_length"] for e in rows]),
        "avg_draft_acceptance_rate": _stats(
            [e.data["avg_draft_acceptance_rate"] for e in rows]
        ),
        "accepted_tps": _stats([e.data["accepted_tps"] for e in rows]),
        "drafted_tps": _stats([e.data["drafted_tps"] for e in rows]),
        "last_per_position_acceptance": last.get("per_position_acceptance", []),
        "total_accepted_tokens": sum(e.data["accepted_tokens"] for e in rows),
        "total_drafted_tokens": sum(e.data["drafted_tokens"] for e in rows),
    }


def _mfu_aggregate(rows: list[LogEvent]) -> dict[str, Any]:
    if not rows:
        return {"available": False, "hint": "需要 --enable-mfu-metrics"}
    active = [e for e in rows if e.data["tflops_per_device"] > 0]
    return {
        "available": True,
        "samples": len(rows),
        "tflops_per_device": _stats([e.data["tflops_per_device"] for e in rows]),
        "tflops_per_device_active": _stats([e.data["tflops_per_device"] for e in active]),
        "gbps_per_device": _stats([e.data["gbps_per_device"] for e in rows]),
        "gbps_per_device_active": _stats([e.data["gbps_per_device"] for e in active]),
    }


def _cudagraph_aggregate(rows: list[LogEvent]) -> dict[str, Any]:
    if not rows:
        return {"available": False, "hint": "需要 --cudagraph-metrics（且开了 HIP/CUDA Graph）"}
    by_mode: dict[str, int] = {}
    unpadded = padded = paddings = 0
    for row in rows:
        by_mode[row.data["runtime_mode"]] = by_mode.get(row.data["runtime_mode"], 0) + row.data["count"]
        unpadded += row.data["unpadded_tokens"] * row.data["count"]
        padded += row.data["padded_tokens"] * row.data["count"]
        paddings += row.data["num_paddings"] * row.data["count"]
    return {
        "available": True,
        "dispatch_by_mode": by_mode,
        "unpadded_tokens": unpadded,
        "padded_tokens": padded,
        "num_paddings": paddings,
        "padding_ratio": (padded / unpadded) if unpadded else None,
    }


def _timeline(iters: list[LogEvent], max_points: int = 600) -> dict[str, Any]:
    """降采样后的时间序列，供报告画 ASCII 图或前端绘图。"""
    if not iters:
        return {}
    step = max(1, len(iters) // max_points)
    sampled = iters[::step]
    return {
        "iteration": [e.data["iteration"] for e in sampled],
        "ts": [e.ts for e in sampled],
        "elapsed_ms": [e.data["elapsed_ms"] for e in sampled],
        "ctx_tokens": [e.data["ctx_tokens"] for e in sampled],
        "gen_tokens": [e.data["gen_tokens"] for e in sampled],
        "kv_usage_perc": [e.data.get("kv_usage_perc") for e in sampled],
        "downsample": step,
    }


# --------------------------------------------------------------------------
# 实时打印
# --------------------------------------------------------------------------


def format_live(event: LogEvent) -> str | None:
    """``--live`` 模式下每条事件的单行渲染。"""
    d = event.data
    if event.kind == "iteration":
        kv = "" if "kv_usage_perc" not in d else f" kv={d['kv_usage_perc']:.1f}%"
        kind = "PREFILL" if d["ctx_tokens"] and not d["gen_tokens"] else (
            "MIXED" if d["ctx_tokens"] and d["gen_tokens"] else (
                "DECODE" if d["gen_tokens"] else "IDLE"
            )
        )
        return (
            f"step #{d['iteration']:<7} {kind:<8} "
            f"ctx={d['ctx_reqs']}req/{d['ctx_tokens']}tok "
            f"gen={d['gen_reqs']}req/{d['gen_tokens']}tok "
            f"{d['elapsed_ms']:8.2f} ms{kv}"
        )
    if event.kind == "stats":
        hit = d.get("prefix_cache_hit_rate")
        return (
            f"[stats] running={d['running']} waiting={d['waiting']} "
            f"kv={d['kv_usage_perc']:.1f}% prompt={d['prompt_tps']:.0f}tok/s "
            f"gen={d['generation_tps']:.0f}tok/s"
            + (f" prefix_hit={hit:.1f}%" if hit is not None else "")
        )
    if event.kind == "spec_decode":
        return (
            f"[spec] accept_len={d['mean_acceptance_length']:.2f} "
            f"draft_acc={d['avg_draft_acceptance_rate']:.1f}% "
            f"per_pos={d['per_position_acceptance']}"
        )
    if event.kind == "mfu":
        return f"[mfu] {d['tflops_per_device']:.1f} TF/s/device {d['gbps_per_device']:.1f} GB/s/device"
    if event.kind == "cudagraph_row":
        return (
            f"[graph] mode={d['runtime_mode']} count={d['count']} "
            f"unpadded={d['unpadded_tokens']} padded={d['padded_tokens']} pad={d['num_paddings']}"
        )
    return None


def summarize_text(summary: dict[str, Any]) -> str:
    """人读摘要（CLI 打印）。"""
    lines: list[str] = []
    counts = summary.get("counts", {})
    lines.append(
        "事件: " + " ".join(f"{k}={v}" for k, v in counts.items() if v)
    )
    steps = summary.get("steps", {})
    if steps:
        lines.append(
            f"step 总数={steps.get('total')} 纯预填充={steps.get('prefill_only')} "
            f"纯解码={steps.get('decode_only')} 混合={steps.get('mixed')} "
            f"空转={steps.get('idle')} (空转占比="
            f"{0 if steps.get('idle_ratio') is None else steps['idle_ratio'] * 100:.1f}%)"
        )
    step_ms = summary.get("step_ms", {})
    if step_ms.get("count"):
        lines.append(
            f"step 耗时 ms: 均值={step_ms['mean']:.1f} p50={step_ms['p50']:.1f} "
            f"p90={step_ms['p90']:.1f} p99={step_ms['p99']:.1f} max={step_ms['max']:.1f}"
        )
    eff = summary.get("efficiency", {})
    if eff.get("prefill_ms_per_1k_context_tokens") is not None:
        lines.append(
            f"预填充 {eff['prefill_ms_per_1k_context_tokens']:.1f} ms / 1k context token"
        )
    if eff.get("decode_ms_per_token") is not None:
        lines.append(f"解码 {eff['decode_ms_per_token']:.2f} ms / token")
    regime = summary.get("regimes", {})
    if regime.get("available"):
        slow_mean = (regime.get("slow_stats") or {}).get("mean")
        lines.append(
            f"稳态: 中位数={regime['median_ms']:.1f}ms 慢态占比={regime['slow_ratio'] * 100:.1f}%"
            + (f" (慢态均值={slow_mean:.1f}ms)" if slow_mean else "")
        )
    spec = summary.get("spec_decode", {})
    if spec.get("available"):
        lines.append(
            f"投机解码: 平均接受长度={spec['mean_acceptance_length']['mean']:.2f} "
            f"接受率={spec['avg_draft_acceptance_rate']['mean']:.1f}% "
            f"位置分布={spec['last_per_position_acceptance']}"
        )
    mfu = summary.get("mfu", {})
    if mfu.get("available") and mfu["tflops_per_device_active"].get("count"):
        lines.append(
            f"MFU: 有负载时 {mfu['tflops_per_device_active']['mean']:.1f} TF/s/device "
            f"{mfu['gbps_per_device_active']['mean']:.1f} GB/s/device"
        )
    cg = summary.get("cudagraph", {})
    if cg.get("available"):
        ratio = cg.get("padding_ratio")
        waste = None if ratio is None else (ratio - 1.0) * 100.0
        lines.append(
            f"图调度: {cg['dispatch_by_mode']} padded/unpadded="
            f"{'—' if ratio is None else f'{ratio:.2f}×'} "
            f"补位浪费={'—' if waste is None else f'{waste:.1f}%'}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from .cli import main as cli_main

    return cli_main(["logs"] + (argv if argv is not None else sys.argv[1:]))


if __name__ == "__main__":  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
