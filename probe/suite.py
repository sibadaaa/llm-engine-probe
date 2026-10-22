"""标准化对比测试套件：固定场景矩阵 → 稳定 schema 的 ``suite_results.json``。

定位：让**不同厂商/不同机器上的同一份负载**产出可直接并排对比的结果
（``python3 -m probe compare A.json B.json``）。

设计要点：

* **确定性**：temperature=0、固定 prompt 构造器（token 密度稳定）、固定矩阵；
* **稳定 schema**：每个场景一组固定字段（请求分位 + 吞吐 + 引擎侧指标差分 +
  可选投机解码/前缀缓存数据），跨厂商缺哪项就为 null，不影响其它项；
* **容错**：场景级失败（超时/报错）只记录到结果里，不中断整个套件——
  半可用的卡也能出报告；
* **客户端 + 引擎双口径**：客户端分位看体验，Prometheus 差分看引擎真实吞吐，
  两者互相校验（和 probe_all 的报告口径一致）。

默认矩阵（8 个场景，见 ``DEFAULT_SCENARIOS``）：热身 / 短对话 / 长解码 /
重预填充 / 并发 c8 / 前缀缓存 / 长上下文 / 投机解码（可选）。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__
from .config import ProbeConfig
from .http_probe import results_to_jsonl, run_matrix
from .metrics_probe import fetch_metrics
from .vendors import resolve_vendor

SUITE_SCHEMA = 1


# --------------------------------------------------------------------------
# 场景矩阵
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    id: str
    label: str
    prompt_tokens: int
    max_tokens: int
    concurrency: int = 1
    repeat: int = 1
    shared_prefix_tokens: int = 0
    warmup: bool = False       # 热身场景：不参与整体成败判定
    optional: bool = False     # 依赖特定能力（如投机解码），缺失不算失败


def default_scenarios(long_context_tokens: int | None = None) -> list[Scenario]:
    """默认 8 场景矩阵（长上下文长度可用 SUITE_LONG_CONTEXT_TOKENS 覆盖）。"""
    long_tokens = long_context_tokens or int(os.environ.get("SUITE_LONG_CONTEXT_TOKENS", "24576") or 24576)
    return [
        Scenario("s01_warmup", "热身（不计入对比）", 512, 8, 1, 1, warmup=True),
        Scenario("s02_chat_short_c1", "短对话 c1（p512/g128）", 512, 128, 1, 3),
        Scenario("s03_decode_heavy_c1", "长解码 c1（p512/g512）", 512, 512, 1, 2),
        Scenario("s04_prefill_heavy", "重预填充（p16k/g32）", 16384, 32, 1, 2),
        Scenario("s05_concurrent_c8", "并发 c8（p4k/g128）", 4096, 128, 8, 2),
        Scenario("s06_prefix_cache", "前缀缓存（共享前缀 2k）", 512, 32, 1, 4, shared_prefix_tokens=2048),
        Scenario("s07_long_context", f"长上下文（p{long_tokens // 1024}k/g64）", long_tokens, 64, 1, 1),
        Scenario("s08_spec_decode", "投机解码（p512/g252）", 512, 252, 1, 3, optional=True),
    ]


def select_scenarios(ids: str | None, matrix: list[Scenario]) -> list[Scenario]:
    """按逗号分隔的 id 子串选择场景（大小写不敏感，如 ``--scenarios s02,s05``）。"""
    if not ids:
        return matrix
    wanted = [token.strip().lower() for token in ids.split(",") if token.strip()]
    picked = [
        scenario for scenario in matrix
        if any(token in scenario.id.lower() for token in wanted)
    ]
    if not picked:
        raise KeyError(f"--scenarios {ids!r} 没有匹配任何场景（可用：{[s.id for s in matrix]}）")
    return picked


# --------------------------------------------------------------------------
# 场景执行
# --------------------------------------------------------------------------


def _scenario_config(base: ProbeConfig, scenario: Scenario) -> ProbeConfig:
    return replace(
        base,
        prompt_tokens=scenario.prompt_tokens,
        max_tokens=scenario.max_tokens,
        concurrency=scenario.concurrency,
        repeat=scenario.repeat,
        shared_prefix_tokens=scenario.shared_prefix_tokens,
        temperature=0.0,
        stream=True,
    )


def _family_total(snapshot: Any, name: str) -> float | None:
    family = snapshot.family(name)
    if family is None:
        return None
    return float(sum(series.value for series in family.series))


def _family_gauge(snapshot: Any, name: str) -> float | None:
    family = snapshot.family(name)
    if family is None or not family.series:
        return None
    return float(family.series[-1].value)


def run_scenario(
    cfg: ProbeConfig,
    scenario: Scenario,
    run_dir: Path,
    verbose: bool = False,
) -> dict[str, Any]:
    """执行一个场景：前后 /metrics 快照 + 请求矩阵，返回稳定 schema 的结果。"""
    scfg = _scenario_config(cfg, scenario)
    result: dict[str, Any] = {
        "id": scenario.id,
        "label": scenario.label,
        "warmup": scenario.warmup,
        "optional": scenario.optional,
        "config": {
            "prompt_tokens": scenario.prompt_tokens,
            "max_tokens": scenario.max_tokens,
            "concurrency": scenario.concurrency,
            "repeat": scenario.repeat,
            "shared_prefix_tokens": scenario.shared_prefix_tokens,
            "temperature": 0.0,
        },
        "ok": False,
        "error": None,
    }
    scenario_dir = run_dir / "suite" / scenario.id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    before = after = None
    try:
        before = fetch_metrics(scfg)
    except Exception as exc:  # noqa: BLE001
        result["metrics_error"] = f"{type(exc).__name__}: {exc}"

    started_at = time.time()
    try:
        results, summary = run_matrix(scfg, warmup=0, progress=verbose)
    except Exception as exc:  # noqa: BLE001 - 套件必须容错
        result["error"] = f"请求矩阵失败：{type(exc).__name__}: {exc}"
        result["duration_s"] = time.time() - started_at
        return result
    duration = time.time() - started_at

    try:
        after = fetch_metrics(scfg)
    except Exception:  # noqa: BLE001
        pass

    # 落盘原始数据（逐请求 + 客户端汇总），供事后深挖
    results_to_jsonl(results, scenario_dir / "http_requests.jsonl")
    (scenario_dir / "http_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    result["duration_s"] = duration
    result["requests"] = summary.get("requests")
    result["ok_requests"] = summary.get("ok")
    result["failed_requests"] = summary.get("failed")
    result["errors"] = summary.get("errors") or []
    result["ttft_ms"] = _pct_block(summary.get("ttft_ms"), want_mean=True, results=results, attr="ttft_ms")
    result["e2e_ms"] = _pct_block(summary.get("e2e_ms"), want_mean=True, results=results, attr="e2e_ms")
    result["tpot_ms"] = _pct_block(summary.get("tpot_ms"), want_mean=True, results=results, attr="tpot_ms")
    itl = summary.get("itl_ms") or {}
    result["itl_ms"] = {
        "mean": itl.get("mean_ms"), "p50": itl.get("p50_ms"),
        "p90": itl.get("p90_ms"), "p99": itl.get("p99_ms"), "max": itl.get("max_ms"),
    }
    result["throughput"] = summary.get("throughput") or {}
    result["prompt_tokens_actual"] = (summary.get("prompt_tokens") or [None])[0]
    result["cached_tokens_last"] = (summary.get("cached_tokens") or [None])[-1] if summary.get("cached_tokens") else None

    server = summary.get("server_side") or {}
    result["server_metrics"] = {
        key: {
            "p50": (server.get(key) or {}).get("p50"),
            "p90": (server.get(key) or {}).get("p90"),
            "mean": (server.get(key) or {}).get("mean"),
        }
        for key in ("queue_time_ms", "time_to_first_token_ms", "generation_time_ms", "mean_itl_ms")
    } if server.get("available") else None

    spec = summary.get("spec_decode") or {}
    result["spec_decode"] = {
        "mean_acceptance_length": spec.get("mean_acceptance_length"),
        "draft_acceptance_rate": spec.get("draft_acceptance_rate"),
        "num_spec_steps": spec.get("num_spec_steps"),
        "acceptance_histogram": spec.get("acceptance_histogram"),
    } if spec.get("available") else None

    result["engine"] = _engine_diff(before, after, duration)

    result["ok"] = bool(summary.get("ok")) and not result["error"]
    return result


def _pct_block(
    pct: dict[str, Any] | None,
    want_mean: bool,
    results: list[Any],
    attr: str,
) -> dict[str, Any] | None:
    if not pct:
        return None
    block = {"p50": pct.get("p50"), "p90": pct.get("p90"), "p99": pct.get("p99")}
    if want_mean:
        values = [getattr(r, attr) for r in results if r.ok and getattr(r, attr) is not None]
        block["mean"] = sum(values) / len(values) if values else None
    return block


def _engine_diff(before: Any, after: Any, duration: float) -> dict[str, Any] | None:
    """Prometheus 计数器差分 → 引擎真实吞吐 / 利用率（缺指标则为 None）。"""
    if before is None or after is None or duration <= 0:
        return None
    window = duration
    out: dict[str, Any] = {"window_s": round(window, 3)}

    def rate(name: str) -> float | None:
        b, a = _family_total(before, name), _family_total(after, name)
        if b is None or a is None:
            return None
        return max(0.0, a - b) / window

    out["prompt_tokens_per_s"] = rate("vllm:prompt_tokens_total")
    out["generation_tokens_per_s"] = rate("vllm:generation_tokens_total")
    out["requests_per_s"] = rate("vllm:request_success_total")
    out["preemptions_delta"] = (
        (_family_total(after, "vllm:num_preemptions_total") or 0)
        - (_family_total(before, "vllm:num_preemptions_total") or 0)
        if _family_total(after, "vllm:num_preemptions_total") is not None
        and _family_total(before, "vllm:num_preemptions_total") is not None
        else None
    )
    hits_b, hits_a = _family_total(before, "vllm:prefix_cache_hits_total"), _family_total(after, "vllm:prefix_cache_hits_total")
    q_b, q_a = _family_total(before, "vllm:prefix_cache_queries_total"), _family_total(after, "vllm:prefix_cache_queries_total")
    if None not in (hits_b, hits_a, q_b, q_a) and (q_a - q_b) > 0:
        out["prefix_cache_hit_rate_delta"] = (hits_a - hits_b) / (q_a - q_b)
    out["kv_usage_perc_after"] = (
        _family_gauge(after, "vllm:kv_cache_usage_perc") * 100.0
        if _family_gauge(after, "vllm:kv_cache_usage_perc") is not None else None
    )
    return out


# --------------------------------------------------------------------------
# 套件级执行
# --------------------------------------------------------------------------


def _engine_context(cfg: ProbeConfig) -> dict[str, Any]:
    """套件头部：引擎/缓存配置（能拿到多少写多少）。

    注意 ``payload["engine"]`` 本身就是这个上下文 dict（既有 schema，compare 的
    指标路径 ``("engine", ...)`` 依赖它）——引擎组件 id 记在 ``context["engine"]``
    （F3 T3.5；取值链 run_meta.json 的 engine → cfg.engine → vllm）。
    """
    context: dict[str, Any] = {"base_url": cfg.base_url, "model": cfg.model}
    context["engine"] = str(getattr(cfg, "engine", "") or "vllm")
    try:
        snapshot = fetch_metrics(cfg)
        family = snapshot.family("vllm:cache_config_info")
        if family and family.series:
            context["cache_config"] = family.series[0].labels
    except Exception:  # noqa: BLE001
        pass
    meta_path = Path(cfg.outdir) / "run_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            context["probe_level"] = meta.get("probe_level")
            context["command"] = meta.get("command")
            context["image"] = meta.get("image")
            if meta.get("engine"):
                context["engine"] = meta["engine"]
            # 运行方式：docker（默认，兼容旧 run_meta）| host（物理机直接跑）
            context["runtime"] = meta.get("runtime") or "docker"
            if isinstance(meta.get("host"), dict):
                context["host"] = meta["host"]
            if meta.get("vendor"):
                context["vendor_in_meta"] = meta.get("vendor")
        except (json.JSONDecodeError, OSError):
            pass
    return context


def _hardware_context() -> dict[str, Any]:
    """一次 GPU 遥测采样（拿不到就为空结构，不阻塞套件）。"""
    try:
        from .telemetry import sample_once

        record = sample_once()
        return {
            "vendor": record.get("vendor"),
            "smi": record.get("smi"),
            "ok": record.get("ok"),
            "cards": [c.get("name") or f"card{c.get('index')}" for c in record.get("cards", [])],
            "vram_perc_sum": record.get("vram_perc_sum"),
            "power_w_sum": record.get("power_w_sum"),
            "temp_c_max": record.get("temp_c_max"),
        }
    except Exception:  # noqa: BLE001
        return {"ok": False}


def run_suite(
    cfg: ProbeConfig,
    scenarios: list[Scenario] | None = None,
    verbose: bool = False,
    progress=None,
) -> dict[str, Any]:
    """跑整个套件，返回 suite_results（同时写入 ``<outdir>/suite_results.json``）。"""
    matrix = scenarios or default_scenarios()
    run_dir = Path(cfg.outdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    profile = resolve_vendor()
    started = time.time()

    payload: dict[str, Any] = {
        "schema": SUITE_SCHEMA,
        "kind": "vllm-probe-suite",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "probe_version": __version__,
        "vendor": {"id": profile.id, "label": profile.label},
        "hardware": _hardware_context(),
        "engine": _engine_context(cfg),
        "matrix": [asdict(s) for s in matrix],
        "scenarios": {},
    }

    for scenario in matrix:
        if progress:
            progress(f"▶ {scenario.id}（{scenario.label}）")
        result = run_scenario(cfg, scenario, run_dir, verbose=verbose)
        payload["scenarios"][scenario.id] = result
        if progress:
            ok_flag = "ok" if result.get("ok") else ("跳过" if result.get("optional") else "ERR")
            ttft = (result.get("ttft_ms") or {}).get("p50")
            progress(f"  {ok_flag}  ttft_p50={'—' if ttft is None else f'{ttft:.0f}ms'}")

    payload["duration_s"] = time.time() - started
    core = [
        scenario.id for scenario in matrix if not scenario.warmup and not scenario.optional
    ]
    payload["ok"] = all(
        payload["scenarios"].get(sid, {}).get("ok") for sid in core
    )
    out = run_dir / "suite_results.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


# --------------------------------------------------------------------------
# 跨厂商对比
# --------------------------------------------------------------------------

# 对比表展示的指标：路径 → (显示名, 单位, 越大越好)
COMPARE_METRICS: list[tuple[tuple[str, ...], str, str, bool]] = [
    (("ttft_ms", "p50"), "TTFT p50", "ms", False),
    (("ttft_ms", "p90"), "TTFT p90", "ms", False),
    (("e2e_ms", "p50"), "E2E p50", "ms", False),
    (("tpot_ms", "p50"), "TPOT p50", "ms", False),
    (("tpot_ms", "mean"), "TPOT 均值", "ms", False),
    (("itl_ms", "p50"), "ITL p50", "ms", False),
    (("itl_ms", "p99"), "ITL p99", "ms", False),
    (("throughput", "output_tokens_per_s"), "客户端吞吐", "tok/s", True),
    (("engine", "generation_tokens_per_s"), "引擎生成速率", "tok/s", True),
    (("engine", "prompt_tokens_per_s"), "引擎预填充速率", "tok/s", True),
    (("engine", "kv_usage_perc_after"), "KV 使用率(后)", "%", False),
    (("spec_decode", "mean_acceptance_length"), "接受长度", "", True),
]


def _dig(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def load_suite(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("kind") != "vllm-probe-suite":
        raise ValueError(f"{path} 不是 probe suite 结果（缺少 kind=vllm-probe-suite）")
    return data


def _fmt_value(value: Any, unit: str) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        text = f"{value:.2f}" if abs(value) < 10 else f"{value:.1f}"
        return f"{text}{unit}"
    return str(value)


def _runtime_desc(payload: dict[str, Any]) -> str:
    """套件结果里的运行方式描述（跨 Docker / 物理机对比时一眼看出差异）。"""
    engine = payload.get("engine") or {}
    if str(engine.get("runtime") or "docker").lower() == "host":
        host = engine.get("host") or {}
        parts = [
            str(part)
            for part in (host.get("hostname"), host.get("python_version") or host.get("python"))
            if part
        ]
        return "物理机（host）" + (f" · {' / '.join(parts)}" if parts else "")
    image = engine.get("image") or ""
    return "容器（docker）" + (f" · {image}" if image else "")


def compare_payloads(a: dict[str, Any], b: dict[str, Any]) -> str:
    """两份 suite_results → 并排 Markdown 对比表。"""
    lines = ["# 标准化套件对比（A vs B）\n"]
    header = [
        ["A", f"{a.get('vendor', {}).get('label', '?')} · {_dig(a, ('engine', 'model')) or '?'}"],
        ["A 运行", str(a.get("created_at", "—"))],
        ["A 运行方式", _runtime_desc(a)],
        ["B", f"{b.get('vendor', {}).get('label', '?')} · {_dig(b, ('engine', 'model')) or '?'}"],
        ["B 运行", str(b.get("created_at", "—"))],
        ["B 运行方式", _runtime_desc(b)],
        ["A 硬件", ", ".join((a.get("hardware") or {}).get("cards") or []) or "—"],
        ["B 硬件", ", ".join((b.get("hardware") or {}).get("cards") or []) or "—"],
    ]
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    for key, value in header:
        lines.append(f"| {key} | {value} |")
    lines.append("")

    scenario_ids = [sid for sid in a.get("scenarios", {}) if sid in b.get("scenarios", {})]
    labels = {s["id"]: s.get("label", s["id"]) for s in a.get("matrix", [])}
    for sid in scenario_ids:
        sa, sb = a["scenarios"][sid], b["scenarios"][sid]
        tag = ""
        if sa.get("warmup") or sb.get("warmup"):
            tag = "（热身）"
        if sa.get("optional") or sb.get("optional"):
            tag += "（可选）"
        lines.append(f"## {sid} · {labels.get(sid, sa.get('label', ''))}{tag}\n")
        if not sa.get("ok") and not sb.get("ok"):
            lines.append("_两边场景都未成功（A: %s / B: %s）_\n" % (
                sa.get("error") or "失败", sb.get("error") or "失败"))
            continue
        lines.append("| 指标 | A | B | B/A |")
        lines.append("|---|---|---|---|")
        for path, name, unit, higher_better in COMPARE_METRICS:
            va, vb = _dig(sa, path), _dig(sb, path)
            ratio = ""
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va != 0:
                ratio = f"{vb / va:.2f}×"
                # 只有明显差异（±0.1%）才标注好/差方向
                if vb > va * 1.001:
                    ratio += " ↑好" if higher_better else " ↑差"
                elif vb < va * 0.999:
                    ratio += " ↓好" if higher_better else " ↓差"
            lines.append(f"| {name} | {_fmt_value(va, unit)} | {_fmt_value(vb, unit)} | {ratio} |")
        lines.append("")
    return "\n".join(lines)
