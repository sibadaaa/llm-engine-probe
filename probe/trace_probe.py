"""算子/kernel 层探测：torch profiler trace（Chrome Trace JSON）分析。

这个模块回答的问题：**"GPU 时间到底花在哪些 kernel / 算子上？CPU 在等谁？"**

数据来源是 vLLM 的 ``--profiler-config '{"profiler":"torch", ...}'`` +
``POST /start_profile`` / ``POST /stop_profile`` 产生的
``rank<N>.<ts>.pt.trace.json[.gz]``（本部署为 **131 MB/rank** 的 Chrome trace）。

为了不把 131MB JSON 一次性读进内存，这里实现了一个**流式** ``traceEvents``
解析器（增量 ``raw_decode``），内存占用与单条事件同阶。

输出内容：

* kernel 分类占比（GEMM / Attention / 通信 / Norm / 逐元素 / MoE / GDN / 采样 …）
* Top-N kernel（总耗时、次数、平均耗时）
* GPU 忙闲时间线（busy / idle、利用率、>1ms 的空洞数量）
* 逐 step（``ProfilerStep#N``）的 wall / GPU busy / 空闲
* ``execute_context_X(Y)_generation_Z(W)`` 注解 → **预填充 step vs 解码 step
  的 GPU 时间与 token 吞吐归因**（本 fork 的逐 iteration 注解）
* 可选：解析 torch 自带的 ``profiler_out_*.txt`` key-averages 表

只依赖标准库（支持 .gz）。
"""

from __future__ import annotations

import gzip
import json
import os
import re
import statistics
import sys
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# --------------------------------------------------------------------------
# 流式解析
# --------------------------------------------------------------------------

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation", "gpu_op"}
# 叶子 GPU 事件（真正的 kernel/拷贝），用于分类占比的分母
GPU_LEAF_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_op"}
# 区间型 GPU 事件（NVTX/annotation 范围），它们**包含**叶子事件，不能直接相加
GPU_RANGE_CATS = {"gpu_user_annotation"}
CPU_CATS = {"cpu_op", "cuda_runtime", "user_annotation", "python_function", "memory"}

_ANNOTATION_RE = re.compile(
    r"execute_context_(?P<ctx_reqs>\d+)\((?P<ctx_tokens>\d+)\)"
    r"_generation_(?P<gen_reqs>\d+)\((?P<gen_tokens>\d+)\)"
)


def open_maybe_gz(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def iter_trace_events(path: str | Path, chunk: int = 1 << 20) -> Iterator[dict[str, Any]]:
    """流式产出 ``traceEvents`` 里的每一个事件（不整体载入内存）。"""
    decoder = json.JSONDecoder()
    with open_maybe_gz(path) as fh:
        buf = fh.read(chunk)
        key = buf.find('"traceEvents"')
        while key == -1:
            more = fh.read(chunk)
            if not more:
                return
            buf = buf[-32:] + more
            key = buf.find('"traceEvents"')
        start = buf.find("[", key)
        if start == -1:
            more = fh.read(chunk)
            if not more:
                return
            buf += more
            start = buf.find("[", key)
        buf = buf[start:]
        pos = 1
        while True:
            while pos >= len(buf):
                more = fh.read(chunk)
                if not more:
                    return
                buf = buf[pos:] + more
                pos = 0
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if pos >= len(buf):
                continue
            if buf[pos] == "]":
                return
            try:
                obj, end = decoder.raw_decode(buf, pos)
            except ValueError:
                more = fh.read(chunk)
                if not more:
                    return
                buf = buf[pos:] + more
                pos = 0
                continue
            pos = end
            if isinstance(obj, dict):
                yield obj


# --------------------------------------------------------------------------
# kernel 分类（基础规则 + 厂商规则包）
# --------------------------------------------------------------------------

# 基础规则：跨厂商通用的 kernel 命名（规则顺序=匹配优先级，先匹配者胜）
BASE_RULES: list[tuple[str, str]] = [
    ("通信/集合通信", r"nccl|rccl|all_reduce|all_gather|reduce_scatter|allreduce|allgather|"
                   r"broadcast|sendrecv|record_param_comms|c10d::|vllm::all_reduce|"
                   r"deepep|mooncake|nixl|custom_all_reduce"),
    ("MoE", r"\bmoe\b|moe_|fused_moe|grouped_gemm|expert|topk_softmax|moe_align"),
    ("量化/GEMM", r"gemm|scaled_mm|w4a|w8a|int8|fp8|sdot|dot4|Cijk_|aten::mm\b|aten::addmm|"
                r"cutlass|linear|matmul|qgemm"),
    ("Attention", r"attention|flash|fmha|paged|decode_attention|prefill_attention|mla|"
                  r"unified_attention|hisparse"),
    ("GDN/Mamba", r"gdn|gated_delta|mamba|ssm|chunk_gated|causal_conv|replayssm"),
    ("Norm/激活", r"norm|rms|layernorm|silu|gelu|swiglu|act_and_mul|rope|rotary"),
    ("采样/logits", r"sampl|logits|argmax|topk|top_p|penalt|softmax|logprob"),
    ("KV/搬运", r"reshape_and_cache|copy_|memcpy|gather|scatter|cat_|concat|transpose|"
               r"contiguous|index_select|to_copy|elementwise"),
    ("图/框架", r"graph|ProfilerStep|cudaLaunch|hipLaunch|LaunchKernel|"
               r"hipEventSynchronize|hipStreamSynchronize|synchronize"),
]

# 厂商规则包：追加到同名分类的基础正则之后（不改变基础规则的优先级）；
# 包中若出现基础规则没有的分类名，则追加为一个新分类（优先级最低）。
CATEGORY_RULE_PACKS: dict[str, dict[str, str]] = {
    "nvidia": {
        "量化/GEMM": r"cublas|cublasLt|ampere_|sm\d+_xmma|wgmma|implicit_gemm|nvjet",
        "Attention": r"fmha_v2|flash_fwd|flash_bwd|mha_fwd|mha_bwd",
        "通信/集合通信": r"\bnvls",
        "图/框架": r"cudaLaunchKernel|cudaGraphLaunch|cuLaunchKernel",
    },
    "rocm": {
        "量化/GEMM": r"hipblas|rocblas|rocblaslt|tensile|ck_tile|ck::",
        "通信/集合通信": r"rcclDevKernel",
        "图/框架": r"hipGraph|hipLaunchKernel|hipModuleLaunchKernel",
    },
    "hygon": {
        "量化/GEMM": r"fp8_emu|emu_gemm|gdn_mtp_update",
        "GDN/Mamba": r"gdn|replayssm|chunk_gated",
    },
    "ascend": {
        "量化/GEMM": r"aclnn|aclop|matmul_v\d|gemm_v\d|npu_gemm",
        "通信/集合通信": r"hccl|hcom",
        "Attention": r"flash_attention_score|paged_attention|prompt_flash|fused_attention",
        "Norm/激活": r"add_rms|npu_rms",
        "采样/logits": r"sampling|top_k_top_p",
        "图/框架": r"aclrt|acl\.",
    },
    "intel": {
        "量化/GEMM": r"onednn|dnnl|xetla|gemm_kernel|bmk_",
        "通信/集合通信": r"\boneccl|\bccl_",
        "图/框架": r"zeCommandList|zeKernelLaunch",
    },
}


def _compile_rule_set(pack: dict[str, str] | None) -> list[tuple[str, re.Pattern[str]]]:
    """基础规则 + 一个厂商包 → 编译后的规则列表。"""
    pack = pack or {}
    merged: list[tuple[str, re.Pattern[str]]] = []
    base_cats = {cat for cat, _ in BASE_RULES}
    for category, pattern in BASE_RULES:
        extra = pack.get(category, "")
        full = f"{pattern}|{extra}" if extra else pattern
        merged.append((category, re.compile(full, re.IGNORECASE)))
    for category, extra in pack.items():
        if category not in base_cats and extra:
            merged.append((category, re.compile(extra, re.IGNORECASE)))
    return merged


# pack=None 表示"未识别厂商"：合并全部厂商包（超集，跨厂商 kernel 名互不冲突）
_ALL_PACKS: dict[str, str] = {}
for _pack in CATEGORY_RULE_PACKS.values():
    for _cat, _pat in _pack.items():
        _ALL_PACKS[_cat] = f"{_ALL_PACKS[_cat]}|{_pat}" if _cat in _ALL_PACKS else _pat

_RULE_CACHE: dict[str | None, list[tuple[str, re.Pattern[str]]]] = {}


def rules_for(vendor: str | None) -> list[tuple[str, re.Pattern[str]]]:
    if vendor in _RULE_CACHE:
        return _RULE_CACHE[vendor]
    if vendor is None or vendor == "generic":
        pack = _ALL_PACKS
    else:
        pack = CATEGORY_RULE_PACKS.get(vendor, _ALL_PACKS)
    compiled = _compile_rule_set(pack)
    _RULE_CACHE[vendor] = compiled
    return compiled


def classify(name: str, vendor: str | None = None) -> str:
    for category, pattern in rules_for(vendor):
        if pattern.search(name):
            return category
    return "其它"


# --------------------------------------------------------------------------
# 从 trace 元数据识别厂商（deviceProperties 的名称，torch profiler 头部）
# --------------------------------------------------------------------------

_TRACE_VENDOR_PATTERNS: list[tuple[str, str]] = [
    ("ascend", r"Ascend|910B|910A|CANN|Atlas"),
    ("nvidia", r"NVIDIA|Tesla|GeForce|Quadro|H100|A100|V100|RTX"),
    ("hygon", r"Hygon|DCU|Z100|Z200|gfx9\d\d"),
    ("amd", r"AMD|Radeon|Instinct|MI\d{2,3}|gfx9\d\d\d"),
    ("intel", r"Intel|Arc |XPU|Ponte|Data Center GPU"),
]


def detect_trace_vendor(path: str | Path, probe_bytes: int = 1 << 20) -> str | None:
    """从 trace 头部（deviceProperties 等元数据）嗅探厂商；识别不了返回 None。"""
    try:
        with open_maybe_gz(path) as fh:
            head = fh.read(probe_bytes)
    except OSError:
        return None
    for vendor, pattern in _TRACE_VENDOR_PATTERNS:
        if re.search(pattern, head):
            return vendor
    return None


def resolve_trace_vendor(path: str | Path, requested: str | None = None) -> str | None:
    """厂商解析：显式参数 > trace 元数据嗅探 > 环境变量 VENDOR > None（超集规则）。"""
    if requested and requested != "auto":
        return requested
    detected = detect_trace_vendor(path)
    if detected:
        return detected
    env = os.environ.get("VENDOR", "auto").strip() or "auto"
    return None if env == "auto" else env


# --------------------------------------------------------------------------
# 聚合
# --------------------------------------------------------------------------


@dataclass
class Agg:
    count: int = 0
    total_us: float = 0.0

    def add(self, dur: float) -> None:
        self.count += 1
        self.total_us += dur


@dataclass
class StepInfo:
    name: str
    start_us: float
    dur_us: float
    gpu_busy_us: float = 0.0
    kernel_count: int = 0


@dataclass
class TraceSummary:
    path: str = ""
    vendor: str | None = None
    events: int = 0
    wall_us: float = 0.0
    gpu_busy_us: float = 0.0
    gpu_leaf_us: float = 0.0
    gpu_range_us: float = 0.0
    cpu_total_us: float = 0.0
    gpu_intervals: int = 0
    idle_gaps_over_1ms: int = 0
    longest_gap_ms: float = 0.0
    categories: dict[str, dict[str, float]] = field(default_factory=dict)
    gpu_by_name: dict[str, Agg] = field(default_factory=dict)
    gpu_ranges: dict[str, Agg] = field(default_factory=dict)
    cpu_by_name: dict[str, Agg] = field(default_factory=dict)
    steps: list[StepInfo] = field(default_factory=list)
    step_annotations: dict[str, dict[str, float]] = field(default_factory=dict)
    device_names: list[str] = field(default_factory=list)
    peak_memory_bytes: float | None = None
    threads: int = 0
    processes: int = 0


def analyze_trace(path: str | Path, limit: int | None = None,
                  vendor: str | None = None) -> TraceSummary:
    """解析一个 trace 文件并聚合。``limit`` 只取前 N 条事件（调试用）。

    ``vendor`` 决定 kernel 分类规则包（None=自动：trace 元数据 → 环境变量 →
    跨厂商超集规则）。
    """
    summary = TraceSummary(path=str(path))
    summary.vendor = resolve_trace_vendor(path, vendor)
    gpu_intervals: list[tuple[float, float]] = []
    threads: set[int] = set()
    processes: set[int] = set()
    step_keys: set[tuple[str, float]] = set()
    step_starts: list[float] = []
    step_ends: list[float] = []
    step_names: list[str] = []

    for index, event in enumerate(iter_trace_events(path)):
        if limit is not None and index >= limit:
            break
        summary.events += 1
        cat = str(event.get("cat", ""))
        name = str(event.get("name", ""))
        pid = event.get("pid")
        tid = event.get("tid")
        if isinstance(tid, int):
            threads.add(tid)
        if isinstance(pid, int):
            processes.add(pid)
        ph = event.get("ph")
        dur = event.get("dur")
        ts = event.get("ts")
        args = event.get("args") or {}

        if ph != "X" or not isinstance(dur, (int, float)) or not isinstance(ts, (int, float)):
            # 只统计完整事件（ph == "X"）
            allocated = None
            if isinstance(args, dict):
                allocated = args.get("Total Allocated")
            if isinstance(allocated, (int, float)):
                summary.peak_memory_bytes = max(summary.peak_memory_bytes or 0, float(allocated))
            continue

        is_device = pid == 0
        is_range = cat in GPU_RANGE_CATS
        is_leaf_gpu = cat in GPU_LEAF_CATS or (is_device and not is_range)
        if is_leaf_gpu:
            summary.gpu_by_name.setdefault(name, Agg()).add(float(dur))
            gpu_intervals.append((float(ts), float(ts) + float(dur)))
        elif is_range:
            summary.gpu_ranges.setdefault(name, Agg()).add(float(dur))
        elif not is_device:
            summary.cpu_by_name.setdefault(name, Agg()).add(float(dur))

        if isinstance(args, dict):
            allocated = args.get("Total Allocated")
            if isinstance(allocated, (int, float)):
                summary.peak_memory_bytes = max(summary.peak_memory_bytes or 0, float(allocated))

        # 逐 step 标记（只认 CPU 侧 user_annotation，避免设备侧副本重复计数）
        if name.startswith("ProfilerStep") and not is_device and not is_range:
            key = (name, float(ts))
            if key not in step_keys:
                step_keys.add(key)
                step_starts.append(float(ts))
                step_ends.append(float(ts) + float(dur))
                step_names.append(name)

        # 逐 iteration 注解（本 fork：execute_context_A(B)_generation_C(D)）
        m = _ANNOTATION_RE.search(name)
        if m:
            kind = _annotation_kind(int(m.group("ctx_tokens")), int(m.group("gen_tokens")))
            bucket = summary.step_annotations.setdefault(
                kind, {"count": 0.0, "time_us": 0.0, "ctx_tokens": 0.0, "gen_tokens": 0.0,
                       "gpu_time_us": 0.0, "gpu_events": 0.0}
            )
            bucket["ctx_tokens"] += int(m.group("ctx_tokens"))
            bucket["gen_tokens"] += int(m.group("gen_tokens"))
            if is_leaf_gpu or is_range:
                bucket["gpu_time_us"] += float(dur)
                bucket["gpu_events"] += 1
            else:
                bucket["count"] += 1
                bucket["time_us"] += float(dur)

        if isinstance(args, dict) and "device" in args and isinstance(args["device"], int):
            device = str(args["device"])
            if device not in summary.device_names:
                summary.device_names.append(device)

    # ---- GPU 忙闲 ----
    if gpu_intervals:
        merged = _merge_intervals(gpu_intervals)
        summary.gpu_intervals = len(merged)
        summary.gpu_busy_us = sum(end - start for start, end in merged)
        cover_start = merged[0][0]
        cover_end = merged[-1][1]
        gaps = [
            (merged[i + 1][0] - merged[i][1], merged[i][1])
            for i in range(len(merged) - 1)
        ]
        big = [gap for gap in gaps if gap[0] > 1000.0]
        summary.idle_gaps_over_1ms = len(big)
        summary.longest_gap_ms = (max((gap[0] for gap in gaps), default=0.0)) / 1000.0
        summary.wall_us = cover_end - cover_start

        # ---- 逐 step 的 GPU busy ----
        starts = [start for start, _ in merged]
        for name, start, end in zip(step_names, step_starts, step_ends):
            busy = 0.0
            i = max(0, bisect_right(starts, start) - 1)
            while i < len(merged) and merged[i][0] < end:
                left = max(merged[i][0], start)
                right = min(merged[i][1], end)
                if right > left:
                    busy += right - left
                i += 1
            summary.steps.append(
                StepInfo(name=name, start_us=start, dur_us=end - start, gpu_busy_us=busy)
            )

    # ---- 分类占比（分母只用叶子 GPU 事件，区间型注解单独统计，避免重复计数）----
    categories: dict[str, dict[str, float]] = {}
    summary.gpu_leaf_us = sum(a.total_us for a in summary.gpu_by_name.values())
    summary.gpu_range_us = sum(a.total_us for a in summary.gpu_ranges.values())
    summary.cpu_total_us = sum(a.total_us for a in summary.cpu_by_name.values())
    total_gpu = summary.gpu_leaf_us
    total_cpu = summary.cpu_total_us
    for name, agg in summary.gpu_by_name.items():
        category = classify(name, summary.vendor)
        row = categories.setdefault(category, {"gpu_us": 0.0, "count": 0.0, "cpu_us": 0.0})
        row["gpu_us"] += agg.total_us
        row["count"] += agg.count
    for name, agg in summary.cpu_by_name.items():
        category = classify(name, summary.vendor)
        row = categories.setdefault(category, {"gpu_us": 0.0, "count": 0.0, "cpu_us": 0.0})
        row["cpu_us"] += agg.total_us
    for row in categories.values():
        row["gpu_share"] = (row["gpu_us"] / total_gpu) if total_gpu else 0.0
        row["cpu_share"] = (row["cpu_us"] / total_cpu) if total_cpu else 0.0
    summary.categories = dict(
        sorted(categories.items(), key=lambda kv: kv[1]["gpu_us"], reverse=True)
    )
    summary.threads = len(threads)
    summary.processes = len(processes)
    return summary


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """把重叠区间合并成不相交的区间列表（输入无需预排序）。"""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _annotation_kind(ctx_tokens: int, gen_tokens: int) -> str:
    if ctx_tokens and gen_tokens:
        return "混合 step（预填充+解码）"
    if ctx_tokens:
        return "预填充 step（context）"
    if gen_tokens:
        return "解码 step（generation）"
    return "空转 step（idle）"


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------


def top_table(summary: TraceSummary, top: int = 25, kind: str = "gpu") -> list[dict[str, Any]]:
    source = summary.gpu_by_name if kind == "gpu" else summary.cpu_by_name
    rows = [
        {
            "name": name,
            "category": classify(name, summary.vendor),
            "count": agg.count,
            "total_ms": agg.total_us / 1000.0,
            "avg_us": agg.total_us / agg.count if agg.count else 0.0,
        }
        for name, agg in source.items()
    ]
    rows.sort(key=lambda r: r["total_ms"], reverse=True)
    return rows[:top]


def to_dict(summary: TraceSummary, top: int = 25) -> dict[str, Any]:
    return {
        "path": summary.path,
        "vendor": summary.vendor,
        "events": summary.events,
        "threads": summary.threads,
        "processes": summary.processes,
        "wall_ms": summary.wall_us / 1000.0,
        "gpu_busy_ms": summary.gpu_busy_us / 1000.0,
        "gpu_leaf_ms": summary.gpu_leaf_us / 1000.0,
        "gpu_range_ms": summary.gpu_range_us / 1000.0,
        "cpu_total_ms": summary.cpu_total_us / 1000.0,
        "gpu_utilization": (
            summary.gpu_leaf_us / summary.wall_us if summary.wall_us else None
        ),
        "gpu_intervals": summary.gpu_intervals,
        "idle_gaps_over_1ms": summary.idle_gaps_over_1ms,
        "longest_gap_ms": summary.longest_gap_ms,
        "peak_memory_bytes": summary.peak_memory_bytes,
        "categories": summary.categories,
        "steps": [
            {
                "name": step.name,
                "wall_ms": step.dur_us / 1000.0,
                "gpu_busy_ms": step.gpu_busy_us / 1000.0,
                "gpu_utilization": (step.gpu_busy_us / step.dur_us) if step.dur_us else None,
            }
            for step in summary.steps
        ],
        "step_annotations": summary.step_annotations,
        "top_gpu_ranges": [
            {"name": name, "count": agg.count, "total_ms": agg.total_us / 1000.0}
            for name, agg in sorted(
                summary.gpu_ranges.items(), key=lambda kv: kv[1].total_us, reverse=True
            )[:top]
        ],
        "top_gpu": top_table(summary, top, "gpu"),
        "top_cpu": top_table(summary, top, "cpu"),
    }


def format_summary(summary: TraceSummary, top: int = 15) -> str:
    lines: list[str] = []
    lines.append(
        f"trace: {summary.path}\n事件数={summary.events} 线程数={summary.threads} "
        f"进程数={summary.processes}"
    )
    if summary.wall_us:
        lines.append(
            f"时间窗口={summary.wall_us / 1000:.1f}ms GPU busy={summary.gpu_busy_us / 1000:.1f}ms "
            f"利用率={summary.gpu_leaf_us / summary.wall_us * 100:.1f}% "
            f">1ms 空洞={summary.idle_gaps_over_1ms} 最长空洞={summary.longest_gap_ms:.1f}ms"
        )
    lines.append(
        f"GPU 叶子事件（kernel/memcpy）合计={summary.gpu_leaf_us / 1000:.1f}ms "
        f"注解区间合计={summary.gpu_range_us / 1000:.1f}ms "
        f"CPU 事件合计={summary.cpu_total_us / 1000:.1f}ms"
    )
    if summary.categories:
        lines.append("\nGPU 时间分类占比（分母=叶子 GPU 事件，故相加约为 100%）:")
        for category, row in summary.categories.items():
            if row["gpu_us"] <= 0:
                continue
            lines.append(
                f"  {category:<14} {row['gpu_us'] / 1000:10.1f} ms  {row['gpu_share'] * 100:5.1f}%  "
                f"events={int(row['count'])}"
            )
    if summary.step_annotations:
        lines.append("\n逐 iteration 注解归因（execute_context_X(Y)_generation_Z(W)）:")
        for kind, row in summary.step_annotations.items():
            lines.append(
                f"  {kind:<24} steps={int(row['count']):<5} GPU={row['gpu_time_us'] / 1000:9.1f} ms "
                f"ctx_tokens={int(row['ctx_tokens'])} gen_tokens={int(row['gen_tokens'])}"
            )
    if summary.steps:
        lines.append("\n逐 ProfilerStep:")
        for step in summary.steps[:12]:
            lines.append(
                f"  {step.name:<18} wall={step.dur_us / 1000:8.1f}ms "
                f"gpu={step.gpu_busy_us / 1000:8.1f}ms "
                f"util={step.gpu_busy_us / step.dur_us * 100 if step.dur_us else 0:5.1f}%"
            )
    lines.append(f"\nTop{top} GPU kernel:")
    for row in top_table(summary, top, "gpu"):
        lines.append(
            f"  {row['total_ms']:10.1f} ms  n={row['count']:<7} avg={row['avg_us']:9.1f}us  "
            f"[{row['category']}] {row['name'][:80]}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# torch 自带 key-averages 表（profiler_out_*.txt）
# --------------------------------------------------------------------------

_UNIT = {"s": 1e6, "ms": 1e3, "us": 1.0, "µs": 1.0, "ns": 1e-3}


def _parse_duration(text: str) -> float | None:
    text = text.strip()
    m = re.match(r"^([\d.]+)\s*(s|ms|us|µs|ns)$", text)
    if not m:
        return None
    return float(m.group(1)) * _UNIT[m.group(2)]


def parse_key_averages(path: str | Path) -> list[dict[str, Any]]:
    """解析 torch profiler 的 key-averages 文本表（``profiler_out_*.txt``）。"""
    rows: list[dict[str, Any]] = []
    header: list[str] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or set(line.strip()) <= {"-"}:
            continue
        cells = [c.strip() for c in re.split(r"\s{2,}", line.strip()) if c.strip()]
        if not cells:
            continue
        # 表头识别：torch 不同后端会打印 CUDA / Device 两种列名（ROCm 复用 CUDA 列名）
        if "Name" in cells[0] and any(
            "CPU" in c or "CUDA" in c or "Device" in c for c in cells
        ):
            header = cells
            continue
        if len(cells) < 2:
            continue
        row: dict[str, Any] = {"name": cells[0]}
        if header and len(cells) == len(header):
            for key, value in zip(header[1:], cells[1:]):
                row[key] = value
        else:
            row["cells"] = cells[1:]
        # 归一化常用列
        for key, value in list(row.items()):
            if not isinstance(value, str):
                continue
            if value.endswith("%"):
                try:
                    row[key.replace(" %", "_pct")] = float(value[:-1])
                except ValueError:
                    pass
            else:
                parsed = _parse_duration(value)
                if parsed is not None:
                    row[key + "_us"] = parsed
                else:
                    try:
                        row[key] = int(value)
                    except ValueError:
                        pass
        rows.append(row)
    return rows


def _row_num(row: dict[str, Any], *keys: str) -> float:
    """按优先级取行内某个数值列（兼容 CUDA / Device 两种列名及旧格式）。"""
    for key in keys:
        if key in row:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return 0.0


# 设备侧耗时列的别名族（NVIDIA/ROCm torch 打印 Self CUDA；部分后端打印 Self Device）
_SELF_DEVICE_KEYS = ("Self CUDA_us", "Self CUDA (us)_us", "Self CUDA_ (us)", "Self Device_us")
_DEVICE_TOTAL_KEYS = ("CUDA total_us", "Device total_us")


def summarize_key_averages(rows: list[dict[str, Any]], top: int = 20) -> dict[str, Any]:
    def total_ms(key: str) -> float:
        return sum(float(r.get(key, 0.0)) for r in rows) / 1000.0

    devices = [r for r in rows if _row_num(r, *_SELF_DEVICE_KEYS) > 0]
    ordered = sorted(rows, key=lambda r: _row_num(r, *_SELF_DEVICE_KEYS), reverse=True)
    device_ms_total = sum(_row_num(r, *_SELF_DEVICE_KEYS) for r in rows) / 1000.0
    return {
        "rows": len(rows),
        "self_cpu_ms_total": total_ms("Self CPU_us"),
        "self_cuda_ms_total": device_ms_total,
        "self_device_ms_total": device_ms_total,
        "cuda_total_ms_total": total_ms("CUDA total_us"),
        "device_total_ms_total": sum(_row_num(r, *_DEVICE_TOTAL_KEYS) for r in rows) / 1000.0,
        "top": [
            {
                "name": r.get("name", ""),
                "self_cpu_ms": float(r.get("Self CPU_us", 0.0)) / 1000.0,
                "self_cuda_ms": _row_num(r, *_SELF_DEVICE_KEYS) / 1000.0,
                "self_device_ms": _row_num(r, *_SELF_DEVICE_KEYS) / 1000.0,
                "cuda_total_ms": _row_num(r, *_DEVICE_TOTAL_KEYS) / 1000.0,
                "calls": r.get("# of Calls"),
            }
            for r in ordered[:top]
        ],
        "device_rows": len(devices),
    }


def main(argv: list[str] | None = None) -> int:
    from .cli import main as cli_main

    return cli_main(["trace"] + (argv if argv is not None else sys.argv[1:]))


if __name__ == "__main__":  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
