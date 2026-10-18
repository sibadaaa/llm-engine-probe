"""汇总层：把各层探测产物合成一份中文 Markdown 报告。

报告结构固定为"从请求到算子"的推理链路顺序，并在最后给出**自动归因**：
每条结论都带触发它的证据数字与可执行的调参建议。

用法::

    python3 -m probe report --outdir runs/20261016-101500
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RUN_META = "run_meta.json"
SERVE_LOG = "serve.log"


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------


def load_json(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def fmt(value: Any, digits: int = 1, unit: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        # 只有极大（>=1e6）或极小（<1e-3）才用科学计数法：
        # 延迟/耗时这类数字用科学计数法会非常难读（例如 12500ms 写成 1.25e+04）
        if isinstance(value, float) and (abs(value) >= 1e6 or (value != 0 and abs(value) < 1e-3)):
            text = f"{value:.3g}"
        else:
            text = f"{value:.{digits}f}"
        return f"{text}{unit}"
    return str(value)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_（无数据）_\n"
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return "\n".join(lines) + "\n"


def sparkline(values: list[float], width: int = 60) -> str:
    """把时间序列压成一行 Unicode 波形，用于报告里"看趋势"。"""
    blocks = "▁▂▃▄▅▆▇█"
    data = [v for v in values if isinstance(v, (int, float))]
    if not data:
        return ""
    if len(data) > width:
        step = len(data) / width
        data = [data[int(i * step)] for i in range(width)]
    low, high = min(data), max(data)
    if high <= low:
        return blocks[0] * len(data)
    return "".join(blocks[min(7, int((v - low) / (high - low) * 7.999))] for v in data)


@dataclass
class Artifacts:
    """一次运行目录里所有可用产物。"""

    run_dir: Path
    meta: dict[str, Any] = field(default_factory=dict)
    metrics_before: dict[str, Any] | None = None
    metrics_after: dict[str, Any] | None = None
    metrics_diff: dict[str, Any] | None = None
    http_summary: dict[str, Any] | None = None
    http_requests: list[dict[str, Any]] = field(default_factory=list)
    steps_summary: dict[str, Any] | None = None
    plugin_summary: dict[str, Any] | None = None
    trace_summary: dict[str, Any] | None = None
    key_averages: dict[str, Any] | None = None
    gpu_telem_rows: list[dict[str, Any]] = field(default_factory=list)
    serve_log: Path | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def scan(cls, run_dir: str | Path) -> "Artifacts":
        d = Path(run_dir)
        art = cls(run_dir=d)
        art.meta = load_json(d / RUN_META) or {}
        art.metrics_before = load_json(d / "metrics_before.json")
        art.metrics_after = load_json(d / "metrics_after.json")
        art.metrics_diff = load_json(d / "metrics_diff.json")
        art.http_summary = load_json(d / "http_summary.json")
        art.http_requests = load_jsonl(d / "http_requests.jsonl")
        art.steps_summary = load_json(d / "steps_summary.json")
        art.plugin_summary = load_json(d / "probe_steps_summary.json")
        art.trace_summary = load_json(d / "trace_summary.json")
        art.key_averages = load_json(d / "key_averages.json")
        art.gpu_telem_rows = load_jsonl(d / "gpu_telem.jsonl")
        log = d / SERVE_LOG
        art.serve_log = log if log.exists() else None
        for extra in ("env_check.json", "profiler_files.json", "suite_results.json", "vendor_info.json"):
            data = load_json(d / extra)
            if data is not None:
                art.extras[extra] = data
        return art


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------

TITLE = "vLLM 推理全流程探测报告"


def build_report(run_dir: str | Path, title: str = TITLE) -> str:
    art = Artifacts.scan(run_dir)
    parts: list[str] = [f"# {title}\n"]
    parts.append(_section_overview(art))
    parts.append(_section_env(art))
    parts.append(_section_http(art))
    parts.append(_section_steps(art))
    parts.append(_section_schedule(art))
    parts.append(_section_spec(art))
    parts.append(_section_memory(art))
    parts.append(_section_kernels(art))
    parts.append(_section_diagnosis(art))
    parts.append(_section_artifacts(art))
    return "\n".join(part for part in parts if part)


def _artifact_vendor(art: Artifacts) -> str:
    """优先 run_meta 里的厂商，其次 trace 元数据嗅探结果。"""
    meta_vendor = str(art.meta.get("vendor") or "").strip()
    if meta_vendor:
        return meta_vendor
    return str((art.trace_summary or {}).get("vendor") or "").strip()


def _vendor_label(art: Artifacts) -> str:
    label = str(art.meta.get("vendor_label") or "").strip()
    vendor = _artifact_vendor(art)
    if label:
        return label
    if vendor:
        try:
            from .vendors import get_vendor

            return get_vendor(vendor).label
        except KeyError:
            return vendor
    return "—"


def _smi_hint(art: Artifacts) -> str:
    """遥测行里记录的实际 SMI 命令（如 hy-smi）。"""
    for row in art.gpu_telem_rows:
        smi = row.get("smi")
        if smi:
            return str(smi)
    vendor = _artifact_vendor(art)
    if vendor:
        try:
            from .vendors import get_vendor

            profile = get_vendor(vendor)
            return profile.smi_cmd or "GPU SMI"
        except KeyError:
            pass
    return "GPU SMI"


def _runtime_label(meta: dict) -> str:
    """运行方式：docker 容器 / 物理机（host）。旧 run_meta 没有 runtime 时按 docker 处理。"""
    return "物理机（host）" if str(meta.get("runtime") or "docker").lower() == "host" else "容器（docker）"


def _runtime_rows(meta: dict) -> list[list[object]]:
    """概览里的运行方式相关行：host 展示宿主环境，docker 展示镜像/容器。"""
    if str(meta.get("runtime") or "docker").lower() == "host":
        host = meta.get("host") or {}
        host_knl = " / ".join(
            part for part in (str(host.get("hostname") or ""), str(host.get("kernel") or "")) if part
        )
        return [
            ["运行方式", _runtime_label(meta)],
            ["Host Python", host.get("python") or "—"],
            ["宿主 Python 版本", host.get("python_version") or "—"],
            ["宿主 vLLM", host.get("vllm_version") or "—"],
            ["宿主 Torch", host.get("torch_version") or "—"],
            ["主机 / 内核", host_knl or "—"],
        ]
    return [
        ["运行方式", _runtime_label(meta)],
        ["镜像", meta.get("image") or "—"],
        ["容器", meta.get("container") or "—"],
    ]


def _section_overview(art: Artifacts) -> str:
    meta = art.meta
    rows = [
        ["运行目录", str(art.run_dir)],
        ["时间", meta.get("started_at", "—")],
        ["厂商", _vendor_label(art) + (f"（{_artifact_vendor(art)}）" if _artifact_vendor(art) else "")],
        *_runtime_rows(meta),
        ["模型", meta.get("model", "—")],
        ["服务名", meta.get("served_name", "—")],
        ["端口", meta.get("port", "—")],
        ["观测档位", meta.get("probe_level", "—")],
        ["张量并行 TP", meta.get("tensor_parallel_size", "—")],
        ["max-model-len", meta.get("max_model_len", "—")],
        ["max-num-batched-tokens", meta.get("max_num_batched_tokens", "—")],
        ["enforce-eager", meta.get("enforce_eager", "—")],
    ]
    present = [
        name
        for name, ok in (
            ("指标快照", art.metrics_before is not None or art.metrics_after is not None),
            ("请求级探测", art.http_summary is not None),
            ("step 级日志", art.steps_summary is not None),
            ("算子级 trace", art.trace_summary is not None),
            ("torch key-averages", art.key_averages is not None),
            (f"{_smi_hint(art)} 采样", bool(art.gpu_telem_rows)),
            ("引擎日志", art.serve_log is not None),
        )
        if ok
    ]
    parts = ["## 0. 运行概览\n", table(["项目", "值"], rows)]
    parts.append(f"**已采集的产物**：{'、'.join(present) if present else '（无）'}\n")
    return "\n".join(parts)


def _section_env(art: Artifacts) -> str:
    env_check = art.extras.get("env_check.json")
    if not env_check:
        return ""
    rows = [
        [item.get("name"), "✅" if item.get("ok") else "❌", item.get("detail", "")]
        for item in env_check.get("checks", [])
    ]
    return "## 1. 探测开关自检\n\n" + table(["检查项", "状态", "证据"], rows)


def _section_http(art: Artifacts) -> str:
    summary = art.http_summary
    if not summary:
        return ""
    cfg = summary.get("config", {})
    lines = ["## 2. 请求级生命周期（客户端视角）\n"]
    lines.append(
        f"负载：prompt≈{cfg.get('prompt_tokens')} token，max_tokens={cfg.get('max_tokens')}，"
        f"并发={cfg.get('concurrency')}，重复={cfg.get('repeat')}，"
        f"流式={cfg.get('stream')}，端点={cfg.get('endpoint')}\n"
    )
    rows = [
        ["成功率", f"{summary.get('ok')}/{summary.get('requests')}"],
        [
            "TTFT ms",
            f"p50={fmt(summary['ttft_ms']['p50'])} p90={fmt(summary['ttft_ms']['p90'])} "
            f"p99={fmt(summary['ttft_ms']['p99'])}",
        ],
        [
            "E2E ms",
            f"p50={fmt(summary['e2e_ms']['p50'])} p90={fmt(summary['e2e_ms']['p90'])} "
            f"p99={fmt(summary['e2e_ms']['p99'])}",
        ],
        [
            "TPOT ms",
            f"p50={fmt(summary['tpot_ms']['p50'])} p90={fmt(summary['tpot_ms']['p90'])}",
        ],
        [
            "ITL ms",
            f"均值={fmt(summary['itl_ms']['mean_ms'])} p50={fmt(summary['itl_ms']['p50_ms'])} "
            f"p99={fmt(summary['itl_ms']['p99_ms'])} max={fmt(summary['itl_ms']['max_ms'])}",
        ],
        [
            "HTTP 首字节 ms",
            f"p50={fmt(summary['headers_ms']['p50'])} p90={fmt(summary['headers_ms']['p90'])}",
        ],
        [
            "吞吐",
            f"输出={fmt(summary['throughput'].get('output_tokens_per_s'))} tok/s，"
            f"总={fmt(summary['throughput'].get('total_tokens_per_s'))} tok/s，"
            f"请求={fmt(summary['throughput'].get('requests_per_s'))} req/s",
        ],
    ]
    lines.append(table(["指标", "值"], rows))
    server = summary.get("server_side") or {}
    if server.get("available"):
        rows = [
            [
                key,
                f"p50={fmt((server.get(key) or {}).get('p50'))} "
                f"p90={fmt((server.get(key) or {}).get('p90'))} "
                f"均值={fmt((server.get(key) or {}).get('mean'))}",
            ]
            for key in (
                "queue_time_ms",
                "time_to_first_token_ms",
                "generation_time_ms",
                "mean_itl_ms",
                "tokens_per_second",
            )
            if server.get(key)
        ]
        lines.append("\n**服务端逐请求指标**（`--enable-per-request-metrics`）：\n")
        lines.append(table(["字段", "分位"], rows))
        lines.append(
            "> 对比要点：服务端 `queue_time_ms` ≈ 排队时间；"
            "`time_to_first_token_ms` 含预填充；客户端 TTFT 还额外包含 HTTP/序列化开销。\n"
        )
    cached = summary.get("cached_tokens") or []
    if cached:
        lines.append(f"\n**前缀缓存命中 token 数（逐个请求）**：{cached}\n")
    if art.http_requests:
        rows = []
        for item in art.http_requests[:20]:
            rows.append(
                [
                    item.get("index"),
                    "✅" if item.get("ok") else "❌",
                    fmt(item.get("prompt_tokens"), 0),
                    fmt(item.get("cached_tokens"), 0),
                    fmt(item.get("ttft_ms")),
                    fmt(item.get("e2e_ms")),
                    fmt(item.get("tpot_ms"), 2),
                    fmt(item.get("completion_tokens"), 0),
                    item.get("finish_reason") or "",
                ]
            )
        lines.append("\n**逐请求明细**（最多 20 行）：\n")
        lines.append(
            table(
                ["#", "OK", "prompt_tok", "cached_tok", "TTFT ms", "E2E ms", "TPOT ms", "out_tok", "结束原因"],
                rows,
            )
        )
    return "\n".join(lines)


def _plugin_block(plugin: dict[str, Any]) -> str:
    """stat logger 插件的结构化数据（比日志解析更精确、字段更多）。"""
    lines = ["\n### 3.x stat logger 插件（结构化 step + 逐请求服务端生命周期）\n"]
    header = plugin.get("header") or {}
    lines.append(
        f"来源文件：`probe_steps.jsonl`（engine={plugin.get('engines')}，"
        f"vLLM={header.get('vllm_version') or '未知'}），共 {fmt(plugin.get('steps'), 0)} 个 step，"
        f"窗口 {fmt(plugin.get('wall_s'))} s，空转 {fmt(plugin.get('idle_steps'), 0)}"
        f"（{fmt((plugin.get('idle_ratio') or 0) * 100)}%）。\n"
    )
    step_ms = plugin.get("step_ms") or {}
    if step_ms.get("count"):
        lines.append(
            table(
                ["step 耗时 ms", "均值", "p50", "p90", "p99", "max"],
                [[
                    fmt(step_ms.get("count"), 0), fmt(step_ms.get("mean")), fmt(step_ms.get("p50")),
                    fmt(step_ms.get("p90")), fmt(step_ms.get("p99")), fmt(step_ms.get("max")),
                ]],
            )
        )
    kv = plugin.get("kv_cache_usage_perc") or {}
    lines.append(
        table(
            ["状态", "值"],
            [
                ["KV 使用率（均值 / 峰值）", f"{fmt(kv.get('mean'))}% / {fmt(kv.get('max'))}%"],
                ["前缀缓存命中率",
                 f"{fmt((plugin.get('prefix_cache_hit_rate') or 0) * 100)}%"
                 f"（{fmt(plugin.get('prefix_cache_hits'), 0)}/{fmt(plugin.get('prefix_cache_queries'), 0)} token）"],
                ["运行 / 等待请求（均值）",
                 f"{fmt((plugin.get('running') or {}).get('mean'))} / {fmt((plugin.get('waiting') or {}).get('mean'))}"],
                ["KV block 驱逐次数", fmt(plugin.get("kv_evictions"), 0)],
            ],
        )
    )
    lifetime = plugin.get("kv_block_lifetime_ms") or {}
    if lifetime.get("count"):
        lines.append(
            f"\nKV block 生命周期（ms）：p50={fmt(lifetime.get('p50_ms'))} "
            f"p90={fmt(lifetime.get('p90_ms'))} 均值={fmt(lifetime.get('mean_ms'))}；"
            f"驱逐前空闲 p50={fmt((plugin.get('kv_block_idle_before_evict_ms') or {}).get('p50_ms'))}\n"
        )
    spec = plugin.get("spec_decode") or {}
    if spec.get("drafts"):
        lines.append(
            f"\n**投机解码（插件口径）**：平均接受长度 {fmt(spec.get('mean_acceptance_length'), 2)}，"
            f"草稿接受率 {fmt((spec.get('draft_acceptance_rate') or 0) * 100)}%，"
            f"按位置接受数 {spec.get('accepted_per_pos')}\n"
        )
    graph = plugin.get("cudagraph") or {}
    if graph.get("modes"):
        lines.append(
            f"\n**图调度（插件口径）**：{graph['modes']}，padded/unpadded="
            f"{fmt(graph.get('padding_ratio'), 2)}×\n"
        )
    if plugin.get("finished_requests"):
        lines.append(
            f"\n**逐请求服务端生命周期**（{fmt(plugin.get('finished_requests'), 0)} 个完成请求，"
            f"结束原因 {plugin.get('finish_reasons')}）\n"
        )
        rows = []
        for key, label in (
            ("request_queue_ms", "排队"),
            ("request_prefill_ms", "预填充"),
            ("request_decode_ms", "解码"),
            ("request_inference_ms", "运行合计"),
            ("request_e2e_ms", "端到端"),
            ("request_tpot_ms", "每 token"),
        ):
            row = plugin.get(key) or {}
            if row.get("count"):
                rows.append([
                    label, fmt(row.get("count"), 0), fmt(row.get("mean_ms")),
                    fmt(row.get("p50_ms")), fmt(row.get("p90_ms")), fmt(row.get("p99_ms")),
                ])
        if rows:
            lines.append(table(["阶段", "样本", "均值 ms", "p50 ms", "p90 ms", "p99 ms"], rows))
        lines.append(
            f"缓存命中 token={fmt(plugin.get('request_cached_tokens'), 0)} / "
            f"prompt token={fmt(plugin.get('request_prompt_tokens'), 0)}，"
            f"生成 token={fmt(plugin.get('request_generation_tokens'), 0)}，"
            f"抢占={fmt(plugin.get('request_preemptions'), 0)} 次\n"
        )
    mfu = plugin.get("mfu") or {}
    if mfu.get("tflops_per_device"):
        lines.append(
            f"\n**MFU（插件口径）**：{fmt(mfu.get('tflops_per_device'), 2)} TF/s/device，"
            f"{fmt(mfu.get('gbps_per_device'))} GB/s/device\n"
        )
    return "\n".join(lines)


def _section_steps(art: Artifacts) -> str:
    s = art.steps_summary
    plugin = art.plugin_summary
    if not s and not plugin:
        return ""
    if not s:
        return (
            "## 3. 引擎 step 级（`--enable-logging-iteration-details`）\n\n"
            "_本次没有解析到日志形式的 step 明细；下面是 stat logger 插件产出的结构化数据。_\n"
            + _plugin_block(plugin or {})
        )
    lines = ["## 3. 引擎 step 级（`--enable-logging-iteration-details`）\n"]
    counts = s.get("counts", {})
    lines.append(
        f"解析到 {counts.get('iteration', 0)} 条 iteration 记录、"
        f"{counts.get('stats', 0)} 条周期统计、{counts.get('spec_decode', 0)} 条投机解码、"
        f"{counts.get('mfu', 0)} 条 MFU。\n"
    )
    window = s.get("window") or {}
    if window:
        lines.append(
            f"时间窗口：{window.get('duration_s', 0):.1f} s（时间戳 "
            f"{window.get('start')} → {window.get('end')}）\n"
        )
    steps = s.get("steps", {})
    rows = [
        ["step 总数", fmt(steps.get("total"), 0)],
        ["纯预填充 step", fmt(steps.get("prefill_only"), 0)],
        ["纯解码 step", fmt(steps.get("decode_only"), 0)],
        ["混合 step", fmt(steps.get("mixed"), 0)],
        [
            "空转 step",
            f"{fmt(steps.get('idle'), 0)}（{fmt((steps.get('idle_ratio') or 0) * 100)}%）",
        ],
    ]
    lines.append(table(["批次构成", "数量"], rows))

    step_ms = s.get("step_ms", {})
    if step_ms.get("count"):
        lines.append(
            "\n**step 耗时分布（ms）**\n\n"
            + table(
                ["样本", "均值", "p50", "p90", "p99", "min", "max", "标准差"],
                [
                    [
                        fmt(step_ms.get("count"), 0),
                        fmt(step_ms.get("mean")),
                        fmt(step_ms.get("p50")),
                        fmt(step_ms.get("p90")),
                        fmt(step_ms.get("p99")),
                        fmt(step_ms.get("min")),
                        fmt(step_ms.get("max")),
                        fmt(step_ms.get("stdev")),
                    ]
                ],
            )
        )
    by_kind = s.get("step_ms_by_kind") or {}
    rows = []
    for label, key in (("预填充", "prefill"), ("解码", "decode"), ("混合", "mixed"), ("空转", "idle")):
        row = by_kind.get(key) or {}
        if not row.get("count"):
            continue
        rows.append(
            [
                label,
                fmt(row.get("count"), 0),
                fmt(row.get("mean")),
                fmt(row.get("p50")),
                fmt(row.get("p90")),
                fmt(row.get("tokens_per_step"), 1),
            ]
        )
    if rows:
        lines.append("\n**分类型 step**\n")
        lines.append(table(["类型", "step 数", "均值 ms", "p50 ms", "p90 ms", "token/step"], rows))

    eff = s.get("efficiency") or {}
    lines.append(
        "\n**效率口径**\n\n"
        + table(
            ["口径", "值"],
            [
                ["预填充耗时", f"{fmt(eff.get('prefill_ms_per_1k_context_tokens'))} ms / 1k context token"],
                ["解码耗时", f"{fmt(eff.get('decode_ms_per_token'), 2)} ms / token"],
                ["解码 step 耗时", f"{fmt(eff.get('decode_ms_per_step'))} ms / step"],
                ["平均 token/step", fmt(eff.get("tokens_per_step_mean"))],
                ["context token 总数", fmt(eff.get("context_tokens_total"), 0)],
                ["generation token 总数", fmt(eff.get("generation_tokens_total"), 0)],
            ],
        )
    )
    timeline = s.get("timeline") or {}
    if timeline.get("elapsed_ms"):
        lines.append("\n**step 耗时趋势**（左旧右新，降采样 %s×）\n" % timeline.get("downsample", 1))
        lines.append("```")
        lines.append(f"elapsed_ms : {sparkline(timeline['elapsed_ms'])}")
        lines.append(f"ctx_tokens : {sparkline(timeline.get('ctx_tokens') or [])}")
        lines.append(f"gen_tokens : {sparkline(timeline.get('gen_tokens') or [])}")
        lines.append(f"kv_usage%  : {sparkline(timeline.get('kv_usage_perc') or [])}")
        lines.append("```\n")
    regime = s.get("regimes") or {}
    if regime.get("available"):
        lines.append(
            "\n**多稳态检查**：中位数 %s ms，慢态阈值 %s ms，慢态占比 %s%%；"
            "慢态均值 %s ms，快态均值 %s ms；首/末三分之一步均值 %s / %s ms\n"
            % (
                fmt(regime.get("median_ms")),
                fmt(regime.get("slow_threshold_ms")),
                fmt((regime.get("slow_ratio") or 0) * 100),
                fmt((regime.get("slow_stats") or {}).get("mean")),
                fmt((regime.get("fast_stats") or {}).get("mean")),
                fmt((regime.get("drift") or {}).get("first_third_mean_ms")),
                fmt((regime.get("drift") or {}).get("last_third_mean_ms")),
            )
        )
    if plugin:
        lines.append(_plugin_block(plugin))
    return "\n".join(lines)


def _section_schedule(art: Artifacts) -> str:
    after = art.metrics_after or art.metrics_before
    if not after:
        return ""
    families = after.get("families", {})

    def total(name: str) -> float | None:
        fam = families.get(name)
        if not fam:
            return None
        return float(sum(float(s["value"]) for s in fam.get("series", [])))

    def gauge(name: str) -> float | None:
        fam = families.get(name)
        if not fam or not fam.get("series"):
            return None
        return float(fam["series"][-1]["value"])

    hits, queries = total("vllm:prefix_cache_hits_total"), total("vllm:prefix_cache_queries_total")
    hit_rate = (hits / queries * 100.0) if hits is not None and queries else None
    lines = ["## 4. 调度 / 队列 / KV 缓存\n"]
    rows = [
        ["运行中请求", fmt(gauge("vllm:num_requests_running"))],
        ["等待中请求", fmt(gauge("vllm:num_requests_waiting"))],
        ["KV 缓存使用率", f"{fmt((gauge('vllm:kv_cache_usage_perc') or 0) * 100)}%"],
        ["前缀缓存命中率", f"{fmt(hit_rate)}%" if hit_rate is not None else "—"],
        ["抢占次数", fmt(total("vllm:num_preemptions_total"), 0)],
        ["prompt token 总数", fmt(total("vllm:prompt_tokens_total"), 0)],
        ["其中命中缓存", fmt(total("vllm:prompt_tokens_cached_total"), 0)],
        ["generation token 总数", fmt(total("vllm:generation_tokens_total"), 0)],
    ]
    waiting_by_reason = [
        s for s in (families.get("vllm:num_requests_waiting_by_reason") or {}).get("series", [])
    ]
    if waiting_by_reason:
        rows.append(
            [
                "等待原因分解",
                "，".join(
                    f"{s['labels'].get('reason')}={fmt(s['value'])}" for s in waiting_by_reason
                ),
            ]
        )
    lines.append(table(["指标", "值"], rows))
    diff = art.metrics_diff
    if diff and diff.get("rates"):
        interesting = [
            "vllm:prompt_tokens_total",
            "vllm:generation_tokens_total",
            "vllm:request_success_total",
            "vllm:num_preemptions_total",
            "vllm:prefix_cache_hits_total",
            "vllm:prefix_cache_queries_total",
            "vllm:spec_decode_num_draft_tokens_total",
            "vllm:spec_decode_num_accepted_tokens_total",
        ]
        rows = [
            [name, fmt(diff["rates"].get(name), 2, "/s"), fmt(diff["counters"].get(name), 0)]
            for name in interesting
            if name in diff["rates"]
        ]
        if rows:
            lines.append(
                f"\n**探测期间增量速率**（窗口 {fmt(diff.get('elapsed_s'))} s）\n\n"
                + table(["计数器", "速率", "增量"], rows)
            )
    if diff and diff.get("histograms"):
        rows = [
            [name, fmt(value.get("count"), 0), fmt(value.get("mean_ms"))]
            for name, value in diff["histograms"].items()
        ]
        lines.append("\n**探测窗口内的直方图增量（均值）**\n\n" + table(["指标", "样本", "均值 ms"], rows))
    return "\n".join(lines)


def _section_spec(art: Artifacts) -> str:
    steps = art.steps_summary or {}
    spec = steps.get("spec_decode") or {}
    http_spec = ((art.http_summary or {}).get("spec_decode") or {})
    if not spec.get("available") and not http_spec.get("available"):
        return ""
    lines = ["## 5. 投机解码（草稿模型 / MTP）\n"]
    rows: list[list[Any]] = []
    if spec.get("available"):
        rows.append(
            [
                "引擎日志（周期）",
                f"平均接受长度 {fmt((spec.get('mean_acceptance_length') or {}).get('mean'), 2)}，"
                f"草稿接受率 {fmt((spec.get('avg_draft_acceptance_rate') or {}).get('mean'))}%，"
                f"样本 {fmt(spec.get('samples'), 0)}",
            ]
        )
        rows.append(
            [
                "累计 accepted/drafted token",
                f"{fmt(spec.get('total_accepted_tokens'), 0)} / {fmt(spec.get('total_drafted_tokens'), 0)}",
            ]
        )
    if http_spec.get("available"):
        rows.append(
            [
                "逐请求（响应体）",
                f"平均接受长度 {fmt(http_spec.get('mean_acceptance_length'), 2)}，"
                f"草稿接受率 {fmt((http_spec.get('draft_acceptance_rate') or 0) * 100)}%，"
                f"spec step 数 {fmt(http_spec.get('num_spec_steps'), 0)}",
            ]
        )
    lines.append(table(["来源", "结果"], rows))
    per_pos = spec.get("last_per_position_acceptance") or []
    hist = http_spec.get("acceptance_histogram") or []
    if per_pos:
        lines.append("\n**按草稿位置的接受率**（第 i 个草稿 token 被接受的比例）\n")
        lines.append(
            table(
                [f"position {i}" for i in range(len(per_pos))],
                [[fmt(v, 3) for v in per_pos]],
            )
        )
    if hist:
        lines.append(
            f"\n**每 step 接受数量直方图**（index=j 表示一步接受 j 个草稿 token）：{hist}\n"
        )
    per_step = http_spec.get("per_step_accepted") or []
    if per_step:
        lines.append("\n**逐 verify step 接受数（前 60 步）**\n\n```\n" + sparkline(per_step, 60) + "\n```\n")
    return "\n".join(lines)


def _section_memory(art: Artifacts) -> str:
    rows: list[list[Any]] = []
    after = art.metrics_after or art.metrics_before
    if after:
        families = after.get("families", {})
        for key, name in (
            ("KV block 生命周期", "vllm:kv_block_lifetime_seconds"),
            ("KV block 空闲至驱逐", "vllm:kv_block_idle_before_evict_seconds"),
            ("KV block 复用间隔", "vllm:kv_block_reuse_gap_seconds"),
        ):
            fam = families.get(name)
            if not fam:
                continue
            counts = [float(s["value"]) for s in fam["series"] if s["name"].endswith("_count")]
            sums = [float(s["value"]) for s in fam["series"] if s["name"].endswith("_sum")]
            if counts and counts[0] > 0:
                rows.append([key, fmt(counts[0], 0), fmt(sums[0] / counts[0] * 1000.0)])
    trace = art.trace_summary
    if trace and trace.get("peak_memory_bytes"):
        rows.append(
            [
                "trace 峰值显存（torch profiler）",
                "—",
                fmt(trace["peak_memory_bytes"] / (1024 ** 3), 2, " GiB"),
            ]
        )
    if art.gpu_telem_rows:
        # gpu_telem.jsonl：probe/telemetry.py 归一化后的多厂商 schema
        #   {ts, vendor, smi, ok, cards:[{vram_perc,power_w,temp_c,util_perc}], vram_perc_sum,...}
        smi = _smi_hint(art)
        vram: list[float] = []
        power: list[float] = []
        temp: list[float] = []
        util: list[float] = []
        samples = 0
        for record in art.gpu_telem_rows:
            if not record.get("ok"):
                continue
            samples += 1
            for card in record.get("cards") or []:
                for values, key in ((vram, "vram_perc"), (power, "power_w"),
                                    (temp, "temp_c"), (util, "util_perc")):
                    value = card.get(key)
                    if isinstance(value, (int, float)):
                        values.append(float(value))
        if vram:
            rows.append([f"{smi} 显存占用（逐卡采样）", fmt(len(vram), 0),
                         f"max={fmt(max(vram))}% 均值={fmt(statistics.fmean(vram))}%"])
        if power:
            rows.append([f"{smi} 功耗（逐卡采样）", fmt(len(power), 0),
                         f"max={fmt(max(power))}W 均值={fmt(statistics.fmean(power))}W"])
        if temp:
            rows.append([f"{smi} 温度（逐卡采样）", fmt(len(temp), 0),
                         f"max={fmt(max(temp))}C 均值={fmt(statistics.fmean(temp))}C"])
        if util:
            rows.append([f"{smi} 利用率（逐卡采样）", fmt(len(util), 0),
                         f"max={fmt(max(util))}% 均值={fmt(statistics.fmean(util))}%"])
        if not samples and art.gpu_telem_rows:
            rows.append([f"{smi} 采样", fmt(len(art.gpu_telem_rows), 0),
                         "（无有效数据：本机 SMI 命令不可用或解析为空）"])
    if not rows:
        return ""
    return "## 6. 显存 / KV 缓存驻留 / 功耗\n\n" + table(["项目", "样本", "值"], rows)


def _section_kernels(art: Artifacts) -> str:
    trace = art.trace_summary
    lines: list[str] = []
    if trace:
        lines.append("## 7. 算子 / kernel 级（torch profiler）\n")
        lines.append(
            table(
                ["项目", "值"],
                [
                    ["trace 文件", trace.get("path", "—")],
                    ["事件数", fmt(trace.get("events"), 0)],
                    ["时间窗口", fmt(trace.get("wall_ms"), 0, " ms")],
                    ["GPU busy", fmt(trace.get("gpu_busy_ms"), 0, " ms")],
                    ["GPU 利用率", f"{fmt((trace.get('gpu_utilization') or 0) * 100)}%"],
                    [">1ms 的空洞数", fmt(trace.get("idle_gaps_over_1ms"), 0)],
                    ["最长空洞", fmt(trace.get("longest_gap_ms"), 2, " ms")],
                ],
            )
        )
        categories = trace.get("categories") or {}
        rows = [
            [name, fmt(row.get("gpu_share", 0) * 100, 1, "%"), fmt(row.get("gpu_us", 0) / 1000.0, 1), fmt(row.get("count"), 0)]
            for name, row in categories.items()
            if row.get("gpu_us")
        ]
        if rows:
            lines.append("\n**GPU 时间分类占比**\n")
            lines.append(table(["分类", "占比", "总耗时 ms", "事件数"], rows))
        annotations = trace.get("step_annotations") or {}
        if annotations:
            rows = [
                [
                    kind,
                    fmt(row.get("count"), 0),
                    fmt(row.get("gpu_time_us", 0) / 1000.0, 1),
                    fmt(row.get("ctx_tokens"), 0),
                    fmt(row.get("gen_tokens"), 0),
                ]
                for kind, row in annotations.items()
            ]
            lines.append("\n**逐 iteration 注解归因**（`execute_context_X(Y)_generation_Z(W)`）\n")
            lines.append(table(["step 类型", "step 数", "GPU 时间 ms", "ctx token", "gen token"], rows))
        steps = trace.get("steps") or []
        if steps:
            rows = [
                [
                    step.get("name"),
                    fmt(step.get("wall_ms")),
                    fmt(step.get("gpu_busy_ms")),
                    f"{fmt((step.get('gpu_utilization') or 0) * 100)}%",
                ]
                for step in steps[:12]
            ]
            lines.append("\n**逐 ProfilerStep（前 12 个）**\n")
            lines.append(table(["step", "wall ms", "GPU busy ms", "GPU 利用率"], rows))
        top = trace.get("top_gpu") or []
        if top:
            lines.append("\n**Top GPU kernel**\n")
            lines.append(
                table(
                    ["总耗时 ms", "占比注", "次数", "平均 us", "分类", "kernel"],
                    [
                        [
                            fmt(row.get("total_ms")),
                            "",
                            fmt(row.get("count"), 0),
                            fmt(row.get("avg_us")),
                            row.get("category"),
                            f"`{str(row.get('name'))[:70]}`",
                        ]
                        for row in top[:20]
                    ],
                )
            )
        top_cpu = trace.get("top_cpu") or []
        if top_cpu:
            lines.append(
                "\n**Top CPU 侧范围 / 算子**"
                "（启动时带 `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` 时会包含 "
                "`schedule: allocate_slots`、`gpu_model_runner: forward/sample/...`、"
                "`llm_engine step: *` 等引擎内部子步骤，可直接看出 CPU 时间花在哪个阶段）\n"
            )
            lines.append(
                table(
                    ["总耗时 ms", "次数", "平均 us", "分类", "范围/算子"],
                    [
                        [
                            fmt(row.get("total_ms")),
                            fmt(row.get("count"), 0),
                            fmt(row.get("avg_us")),
                            row.get("category"),
                            f"`{str(row.get('name'))[:80]}`",
                        ]
                        for row in top_cpu[:20]
                    ],
                )
            )
    ka = art.key_averages
    if ka:
        if not lines:
            lines.append("## 7. 算子 / kernel 级（torch profiler）\n")
        device_ms_total = ka.get("self_device_ms_total", ka.get("self_cuda_ms_total"))
        lines.append(
            "\n**torch key-averages（`profiler_out_*.txt`）**：Self CUDA/Device 合计 "
            f"{fmt(device_ms_total)} ms，Self CPU 合计 {fmt(ka.get('self_cpu_ms_total'))} ms\n"
        )
        rows = [
            [
                fmt(row.get("self_device_ms", row.get("self_cuda_ms"))),
                fmt(row.get("self_cpu_ms")),
                fmt(row.get("calls"), 0),
                f"`{str(row.get('name'))[:70]}`",
            ]
            for row in (ka.get("top") or [])[:15]
        ]
        lines.append(table(["设备耗时 ms", "Self CPU ms", "次数", "算子"], rows))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 自动归因
# --------------------------------------------------------------------------


@dataclass
class Finding:
    level: str  # 高 / 中 / 低
    symptom: str
    evidence: str
    action: str


# --------------------------------------------------------------------------
# 诊断阈值：默认基线来自本项目的现网实测（27B / TP4），
# 厂商可在 probe/vendors.py 的 VendorProfile.diagnose_overrides 中覆盖。
# --------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict[str, float] = {
    "queue_p90_ms": 50.0,            # 服务端队列等待 p90 超过它 → 调度容量告警
    "kv_usage_ratio": 0.9,           # KV 使用率超过它 → 抢占/打满告警
    "client_ttft_freeze_ms": 2000.0,  # 客户端 TTFT p90 秒级冻结
    "client_server_ttft_ratio": 3.0,  # 客户端/服务端 TTFT 背离倍数
    "itl_p99_freeze_ms": 2000.0,     # ITL p99 秒级冻结
    "prefix_hit_low": 0.2,           # 前缀缓存命中率低
    "prefix_hit_high": 0.6,          # 前缀缓存命中率高
    "idle_ratio": 0.3,               # 空转 step 占比
    "slow_ratio": 0.2,               # 慢态 step 占比（双稳态）
    "decode_ms_per_token": 50.0,     # 解码每 token 耗时
    "prefill_ms_per_1k": 500.0,      # 预填充 每 1k context token 耗时
    "gemm_share_high": 0.45,         # GEMM/量化 GPU 占比
    "comm_share_high": 0.15,         # 集合通信 GPU 占比
    "attn_share_high": 0.30,         # Attention GPU 占比
    "gpu_util_low": 0.6,             # GPU 利用率低（配合空洞数）
    "accept_len_low": 1.5,           # 投机解码平均接受长度低
    "accept_len_high": 3.0,          # 投机解码平均接受长度高
    "padding_ratio": 1.2,            # 图捕获桶 padding 比例
}


def thresholds_for(vendor: str | None) -> dict[str, float]:
    """默认阈值 + 厂商覆盖（probe/vendors.py::VendorProfile.diagnose_overrides）。"""
    thresholds = dict(DEFAULT_THRESHOLDS)
    if vendor:
        try:
            from .vendors import get_vendor

            thresholds.update(get_vendor(vendor).diagnose_overrides)
        except KeyError:
            pass
    return thresholds


def diagnose(art: Artifacts) -> list[Finding]:
    findings: list[Finding] = []
    http = art.http_summary or {}
    steps = art.steps_summary or {}
    trace = art.trace_summary or {}
    after = art.metrics_after or art.metrics_before or {}
    families = after.get("families", {})
    vendor = _artifact_vendor(art)
    th = thresholds_for(vendor or None)
    smi_name = _smi_hint(art)
    graph_label = "HIP/CUDA Graph"
    if vendor:
        try:
            from .vendors import get_vendor

            graph_label = get_vendor(vendor).graph_label
        except KeyError:
            pass

    def total(name: str) -> float:
        fam = families.get(name)
        if not fam:
            return 0.0
        return float(sum(float(s["value"]) for s in fam.get("series", [])))

    def gauge(name: str) -> float | None:
        fam = families.get(name)
        if not fam or not fam.get("series"):
            return None
        return float(fam["series"][-1]["value"])

    # ---- 排队与调度容量 ------------------------------------------------
    queue_p90 = ((http.get("server_side") or {}).get("queue_time_ms") or {}).get("p90")
    waiting = gauge("vllm:num_requests_waiting")
    running = gauge("vllm:num_requests_running")
    if queue_p90 is not None and queue_p90 > th["queue_p90_ms"]:
        findings.append(
            Finding(
                "高",
                "队列等待偏高，请求在 WAITING 队列里被卡住",
                f"服务端 queue_time p90={queue_p90:.1f} ms，running={running}，waiting={waiting}",
                "提高 `--max-num-seqs` / `--max-num-batched-tokens`（注意 KV 余量），"
                "或降低单请求 `max_tokens`；若 waiting 的 reason=capacity 说明是调度容量限制，"
                "reason=deferred 说明是 LoRA / KV 传输等瞬时约束",
            )
        )
    preemptions = total("vllm:num_preemptions_total")
    kv_usage = gauge("vllm:kv_cache_usage_perc")
    if preemptions > 0 or (kv_usage is not None and kv_usage > th["kv_usage_ratio"]):
        findings.append(
            Finding(
                "高",
                "KV 缓存压力大，出现抢占或接近打满",
                f"抢占={preemptions:.0f} 次，KV 使用率={0 if kv_usage is None else kv_usage * 100:.1f}%",
                "降低 `--max-model-len` 或并发；提高 `--gpu-memory-utilization`；"
                "考虑 `--kv-cache-dtype fp8`；检查前缀缓存是否把长前缀长期占住",
            )
        )

    # ---- 客户端口径 vs 服务端口径背离 -----------------------------------
    # 现网实测（2026-10-15，qwen3.8-27b TP4 DFlash2-k7，c=32 各 1.45k prompt）：
    # 服务端逐请求 queue_time p90=8.3ms、TTFT p90=12.9s，而客户端实测 TTFT p90=88.8s。
    # 原因是服务端 queue_time 只覆盖"入队→首次被调度"，**不含**排在别人的预填充 chunk
    # 之后等待的调度级排队；只按 queue_time 判阈值会把最严重的卡点判成"没排队"。
    client_ttft_p90 = (http.get("ttft_ms") or {}).get("p90")
    server_ttft_p90 = ((http.get("server_side") or {}).get("time_to_first_token_ms") or {}).get("p90")
    if (
        client_ttft_p90
        and server_ttft_p90
        and client_ttft_p90 > th["client_ttft_freeze_ms"]
        and client_ttft_p90 > th["client_server_ttft_ratio"] * server_ttft_p90
    ):
        findings.append(
            Finding(
                "高",
                "客户端 TTFT 远大于服务端自报 TTFT：服务端 queue 指标漏报调度级排队",
                f"客户端 TTFT p90={client_ttft_p90:.0f} ms vs 服务端 TTFT p90={server_ttft_p90:.0f} ms"
                f"（服务端 queue p90={queue_p90 if queue_p90 is None else round(queue_p90, 1)} ms），"
                f"比值 {client_ttft_p90 / max(1.0, server_ttft_p90):.1f}×",
                "容量评估以**客户端/网关侧 TTFT** 为准；`vllm:request_queue_time_seconds` 与逐请求 "
                "`queue_time_ms` 只覆盖入队到首次调度，不包含「排在其他请求的预填充 chunk 之后」的等待。"
                "排查方向：降低 `--max-num-batched-tokens`、限制单步预填充占比、或为长 prompt 单独分流。",
            )
        )

    # ---- 解码 token 间隔的秒级冻结（step 边界阻塞）-----------------------
    itl = http.get("itl_ms") or {}
    itl_p99 = itl.get("p99_ms")
    if itl_p99 and itl_p99 > th["itl_p99_freeze_ms"]:
        findings.append(
            Finding(
                "高",
                "解码 token 间隔出现秒级冻结：单个 step 的批次构成把所有解码流整段挡住",
                f"客户端 ITL p99={itl_p99:.0f} ms，max={itl.get('max_ms')} ms，"
                f"step 耗时 p99={fmt((steps.get('step_ms') or {}).get('p99'))} ms",
                "step 是调度量子：一个 step 里塞进大块预填充 chunk 时，同一 step 内的解码请求必须等它算完。"
                "把 `--max-num-batched-tokens` 调小（例如 8192→2048）可把秒级冻结压到亚秒级；"
                "同时确认没有把长 prompt 与交互式解码混在同一实例上。",
            )
        )

    # ---- 前缀缓存 ------------------------------------------------------
    hits, queries = total("vllm:prefix_cache_hits_total"), total("vllm:prefix_cache_queries_total")
    if queries > 0:
        rate = hits / queries
        if rate < th["prefix_hit_low"]:
            findings.append(
                Finding(
                    "中",
                    "前缀缓存命中率低，重复前缀没被复用",
                    f"命中率={rate * 100:.1f}%（hits={hits:.0f}/queries={queries:.0f}）",
                    "确认启动参数含 `--enable-prefix-caching`；把 system prompt / few-shot 放在"
                    "请求最前面并保持逐字节一致；长上下文场景可配合 `--kv-cache-metrics` 观察 block 复用间隔",
                )
            )
        elif rate > th["prefix_hit_high"]:
            findings.append(
                Finding(
                    "低",
                    "前缀缓存命中率高，预填充已被显著摊薄",
                    f"命中率={rate * 100:.1f}%",
                    "继续利用：把稳定前缀做长、把变化内容后置；命中率高时 TTFT 主要剩算力瓶颈",
                )
            )

    # ---- step 级 --------------------------------------------------------
    steps_info = steps.get("steps") or {}
    idle_ratio = steps_info.get("idle_ratio")
    if idle_ratio is not None and idle_ratio > th["idle_ratio"]:
        findings.append(
            Finding(
                "中",
                "空转 step 占比高：引擎在无请求或等待中反复 step",
                f"空转 {steps_info.get('idle')}/{steps_info.get('total')}（{idle_ratio * 100:.1f}%）",
                "空转通常是低负载下的正常现象；若此时仍在压测，说明请求被调度限制，"
                "检查 waiting 队列与 `--max-num-seqs`",
            )
        )
    regime = steps.get("regimes") or {}
    if regime.get("available") and regime.get("slow_ratio", 0) > th["slow_ratio"]:
        findings.append(
            Finding(
                "高",
                "step 耗时存在双稳态（快/慢两态），不是单一稳定瓶颈",
                f"中位数={regime['median_ms']:.1f} ms，慢态占比={regime['slow_ratio'] * 100:.1f}%，"
                f"慢态均值={regime['slow_stats'].get('mean', 0):.1f} ms",
                f"对比 {smi_name} 采到的功耗/温度（降频）、KV 使用率（换出）、批次构成（大 prefill 混入）；"
                "把 step 时间与 `ctx_tokens` 对齐判断是预填充抖动还是访存/通信抖动",
            )
        )
    eff = steps.get("efficiency") or {}
    if eff.get("decode_ms_per_token"):
        decode_ms = eff["decode_ms_per_token"]
        if decode_ms > th["decode_ms_per_token"]:
            findings.append(
                Finding(
                    "中",
                    "解码每 token 耗时很高，需要看算子级占比",
                    f"{decode_ms:.1f} ms/token",
                    "结合第 7 节：若 GEMM/量化算子占比 >45%，瓶颈是权重访存（考虑更低比特/更优 GEMV）；"
                    "若通信占比高，看 TP/EP 切分与 all-reduce 拓扑；若空洞多，看 CPU 侧调度",
                )
            )
    if eff.get("prefill_ms_per_1k_context_tokens"):
        per_1k = eff["prefill_ms_per_1k_context_tokens"]
        if per_1k > th["prefill_ms_per_1k"]:
            findings.append(
                Finding(
                    "中",
                    "预填充吞吐偏低",
                    f"{per_1k:.0f} ms / 1k context token",
                    "检查 `--max-num-batched-tokens` 是否把预填充切得过碎；"
                    "确认 chunked prefill 与 attention 后端匹配；看 trace 中 Attention/GEMM 占比",
                )
            )

    # ---- 算子级 --------------------------------------------------------
    categories = trace.get("categories") or {}
    if categories:
        gemm = next((v for k, v in categories.items() if "GEMM" in k), None)
        comm = next((v for k, v in categories.items() if "通信" in k), None)
        attn = next((v for k, v in categories.items() if "Attention" in k), None)
        gpu_share = gemm.get("gpu_share") if gemm else None
        top_gpu = trace.get("top_gpu") or []
        top_names = " ".join(str(row.get("name", "")) for row in top_gpu[:10])
        emulated_fp8 = ("fp8_emu" in top_names) or ("emu_gemm" in top_names)
        if gpu_share and gpu_share > th["gemm_share_high"]:
            if emulated_fp8:
                if vendor == "hygon":
                    fp8_advice = (
                        "注意这**不是**访存受限而是仿真计算受限（gfx906 无原生 FP8，"
                        "`VLLM_DCU_USE_FP8_EMU=1` 用 fp16 反量化后再算）。"
                        "先做 A/B：bf16/fp16 权重（去掉 emu 路径）vs 现网 FP8-EMU；"
                        "或换 int8 w8a8/sdot4 量化；并确认 M=1+k 的投机验证没有 tile padding 浪费。"
                    )
                elif vendor == "amd":
                    fp8_advice = (
                        "注意这**不是**访存受限而是仿真计算受限（gfx906 等老卡无原生 FP8，"
                        "走的是 fp16 反量化后再算的模拟路径）。先做 A/B：bf16/fp16 权重（去掉 emu 路径）"
                        "vs FP8 模拟；或换 int8 w8a8 量化；并确认 M=1+k 的投机验证没有 tile padding 浪费。"
                    )
                else:
                    fp8_advice = (
                        "kernel 名命中 FP8 模拟路径（fp8_emu/emu_gemm）：量化收益可能被仿真开销吃掉。"
                        "确认该卡是否有原生 FP8 支持；若无，优先 bf16/fp16 或 int8 w8a8 量化，"
                        "并确认投机解码验证没有 tile padding 浪费。"
                    )
                findings.append(
                    Finding(
                        "高",
                        "GPU 时间以 FP8「模拟」GEMM 为主：量化收益被仿真开销吃掉",
                        f"GEMM/量化占比={gpu_share * 100:.1f}%，Top kernel 命中 `fp8_emu_gemm_block_kernel`，"
                        f"通信占比={fmt((comm or {}).get('gpu_share', 0) * 100)}%",
                        fp8_advice,
                    )
                )
            else:
                findings.append(
                    Finding(
                        "高",
                        "GPU 时间以 GEMM/量化算子为主，属于访存受限型瓶颈",
                        f"GEMM/量化占比={gpu_share * 100:.1f}%",
                        "优先优化权重访存路径：更低比特量化（w4a16 / int8 sdot4 / fp8）、"
                        "增大有效 M（batch/投机 token 数）、确认没有 M=1 的 64× padding 浪费",
                    )
                )
        if comm and comm.get("gpu_share", 0) > th["comm_share_high"]:
            findings.append(
                Finding(
                    "中",
                    "集合通信占比偏高",
                    f"通信占比={comm['gpu_share'] * 100:.1f}%",
                    "检查 TP/DP 切分是否过大、是否需要 `--disable-custom-all-reduce`、"
                    "以及是否存在细碎 all-reduce（合并小通信）",
                )
            )
        if attn and attn.get("gpu_share", 0) > th["attn_share_high"]:
            findings.append(
                Finding(
                    "中",
                    "Attention 占比偏高",
                    f"Attention 占比={attn['gpu_share'] * 100:.1f}%",
                    "长上下文下优先查 attention 后端与 KV layout（block size / dtype），"
                    "确认 chunked prefill 与 decode 的 kernel 选择",
                )
            )
        util = trace.get("gpu_utilization")
        if util is not None and util < th["gpu_util_low"] and (trace.get("idle_gaps_over_1ms") or 0) > 0:
            findings.append(
                Finding(
                    "中",
                    "GPU 时间线存在明显空洞，疑似 CPU 侧或同步等待",
                    f"GPU 利用率={util * 100:.1f}%，>1ms 空洞={trace.get('idle_gaps_over_1ms')} 个，"
                    f"最长空洞={trace.get('longest_gap_ms')} ms",
                    "用 `scripts/pyspy.sh` 抓 Python 栈看是否卡在调度/后处理/日志；"
                    "确认日志量级（iteration details 会放大 CPU 开销）；必要时关闭逐 iteration 日志复测",
                )
            )

    # ---- 投机解码 ------------------------------------------------------
    spec = (steps.get("spec_decode") or {})
    if spec.get("available"):
        mean_len = (spec.get("mean_acceptance_length") or {}).get("mean")
        if mean_len is not None and mean_len < th["accept_len_low"]:
            findings.append(
                Finding(
                    "中",
                    "投机解码接受长度低，草稿模型与目标模型分布不匹配",
                    f"平均接受长度={mean_len:.2f}（理论上限=k+1）",
                    "考虑减小 k、换 draft 模型/温度对齐、或对草稿头做蒸馏适配；"
                    "若位置 2 之后接受率≈0，说明只有第一个草稿 token 有效",
                )
            )
        elif mean_len is not None and mean_len > th["accept_len_high"]:
            findings.append(
                Finding(
                    "低",
                    "接受长度较高，可尝试增大 k 进一步摊薄解码步数",
                    f"平均接受长度={mean_len:.2f}",
                    "在 `--speculative-config` 中增大 `num_speculative_tokens` 做 A/B，"
                    "同时观察 GPU 时间是否被 verify pass 吃掉",
                )
            )

    # ---- 图模式 --------------------------------------------------------
    cg = steps.get("cudagraph") or {}
    if cg.get("available"):
        modes = cg.get("dispatch_by_mode") or {}
        if modes and not any(mode != "NONE" for mode in modes):
            graph_note = (
                "确认没有 `--enforce-eager`，且批次大小落在捕获桶内。"
            )
            if vendor == "hygon":
                graph_note += "本部署历史结论是图模式对 27B decode 收益仅 ~0.5%，优先级低于算子优化。"
            else:
                graph_note += "图模式收益与模型/批次强相关，建议与 eager 基线做 A/B 后再决定优先级。"
            findings.append(
                Finding(
                    "低",
                    f"{graph_label} 未真正 replay（Runtime Mode 全是 NONE）",
                    f"dispatch={modes}",
                    graph_note,
                )
            )
        padding_ratio = cg.get("padding_ratio")
        if padding_ratio and padding_ratio > th["padding_ratio"]:
            findings.append(
                Finding(
                    "中",
                    "图捕获桶与实际批次不匹配，padding 浪费明显",
                    f"padding 比例={padding_ratio * 100:.0f}%",
                    "调整 `--compilation-config` 的 capture sizes，让常见 batch 精确落桶",
                )
            )
    return findings


def _section_diagnosis(art: Artifacts) -> str:
    findings = diagnose(art)
    lines = ["## 8. 自动归因与优化建议\n"]
    if not findings:
        lines.append("_未触发任何规则：要么数据不足，要么各项指标都在阈值内。_\n")
        lines.append(
            "\n> 提示：至少采集请求级 + step 级 + 算子级三类数据，规则才能给出有意义的结论。\n"
        )
        return "\n".join(lines)
    rows = [[f.level, f.symptom, f.evidence, f.action] for f in findings]
    lines.append(table(["优先级", "现象", "证据", "建议动作"], rows))
    lines.append(
        "\n> 规则的默认阈值定义在 `probe/report.py` 的 `DEFAULT_THRESHOLDS`（来自本项目现网实测），"
        "厂商专属覆盖在 `probe/vendors.py` 的 `diagnose_overrides`，可按你的历史基线调整。\n"
    )
    return "\n".join(lines)


def _section_artifacts(art: Artifacts) -> str:
    lines = ["## 9. 产物清单与复现\n"]
    rows = []
    for path in sorted(art.run_dir.glob("*")):
        if path.is_file():
            rows.append([f"`{path.name}`", fmt(path.stat().st_size / 1024.0, 1, " KiB")])
    lines.append(table(["文件", "大小"], rows))
    is_host = str(art.meta.get("runtime") or "docker").lower() == "host"
    prelude = "# 物理机（host）模式：可先跑环境体检\nmake -C vllm-docker-probe host-check\n" if is_host else ""
    runtime_var = "RUNTIME=host " if is_host else ""
    lines.append(
        "\n复现命令（在项目根目录执行）：\n\n```bash\n"
        + prelude
        + "make -C vllm-docker-probe " + runtime_var + "probe RUN_DIR=" + str(art.run_dir) + "\n```\n"
    )
    cmd = art.meta.get("command")
    if cmd:
        label = "本次启动命令（物理机直接启动）" if is_host else "本次启动命令"
        lines.append("\n" + label + "：\n\n```bash\n" + str(cmd) + "\n```\n")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from .cli import main as cli_main

    return cli_main(["report"] + (argv if argv is not None else []))
