"""``/metrics`` 层探测：Prometheus 文本解析、直方图分位数、阶段表。

这个模块回答的问题：**"服务器自己看到的推理各阶段耗时是多少？"**

它把 vLLM 暴露的直方图（queue / prefill / decode / TTFT / ITL / TPOT / e2e）
换算成均值与 p50/p90/p99，并把队列、KV 缓存、前缀缓存、抢占、投机解码、
MFU 等计数器整理成结构化 JSON，供 ``report`` 生成中文报告。

只依赖标准库。典型用法::

    python3 -m probe metrics --outdir runs/latest          # 保存快照
    python3 -m probe metrics --diff before.json after.json # 两次快照做差
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import ProbeConfig

# --------------------------------------------------------------------------
# Prometheus 文本协议解析
# --------------------------------------------------------------------------

_SAMPLE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?"
    r"[ \t]+(?P<value>[^ \t]+)"
    r"(?:[ \t]+(?P<ts>-?\d+))?[ \t]*$"
)
_LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')
_HELP_RE = re.compile(r"^#\s+HELP\s+(\S+)\s*(.*)$")
_TYPE_RE = re.compile(r"^#\s+TYPE\s+(\S+)\s*(\S+)?\s*$")
_HIST_SUFFIXES = ("_bucket", "_sum", "_count")


def _unescape(value: str) -> str:
    return value.replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "\n")


def _parse_labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    return {k: _unescape(v) for k, v in _LABEL_RE.findall(raw)}


@dataclass
class Series:
    """一条样本（一个 name + 一组 label）。"""

    name: str
    labels: dict[str, str]
    value: float

    def matches(self, **want: str) -> bool:
        return all(self.labels.get(k) == v for k, v in want.items())


@dataclass
class Family:
    """一个指标族（histogram 的 _bucket/_sum/_count 归到同一族）。"""

    name: str
    type: str = "untyped"
    help: str = ""
    series: list[Series] = field(default_factory=list)

    # -- 便捷访问 ---------------------------------------------------------
    def select(self, **labels: str) -> list[Series]:
        return [s for s in self.series if s.matches(**labels)]

    def value(self, **labels: str) -> float | None:
        got = self.select(**labels)
        return got[0].value if got else None

    def total(self, **labels: str) -> float:
        return float(sum(s.value for s in self.select(**labels)))

    @property
    def count(self) -> float:
        """histogram/summary 的观测次数；其它类型退化为样本条数。"""
        counts = [s.value for s in self.series if s.name.endswith("_count")]
        if counts:
            return float(sum(counts))
        return float(len(self.series))

    def sum(self) -> float:
        sums = [s.value for s in self.series if s.name.endswith("_sum")]
        return float(sum(sums)) if sums else 0.0

    def buckets(self, **labels: str) -> list[tuple[float, float]]:
        """返回 [(上界, 累计计数)]，按上界升序。"""
        out: list[tuple[float, float]] = []
        for s in self.series:
            if not s.name.endswith("_bucket"):
                continue
            if not s.matches(**labels):
                continue
            le = s.labels.get("le")
            if le is None:
                continue
            bound = math.inf if le in ("+Inf", "Inf", "inf") else float(le)
            out.append((bound, s.value))
        out.sort(key=lambda item: item[0])
        return out

    def mean(self) -> float | None:
        count = sum(s.value for s in self.series if s.name.endswith("_count"))
        total = sum(s.value for s in self.series if s.name.endswith("_sum"))
        if count <= 0:
            return None
        return total / count


@dataclass
class MetricsSnapshot:
    """一次 ``/metrics`` 抓取的完整内容。"""

    families: dict[str, Family] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- 访问 -------------------------------------------------------------
    def family(self, name: str) -> Family | None:
        return self.families.get(name)

    def value(self, name: str, **labels: str) -> float | None:
        fam = self.families.get(name)
        return fam.value(**labels) if fam else None

    def total(self, name: str, **labels: str) -> float:
        fam = self.families.get(name)
        return fam.total(**labels) if fam else 0.0

    def quantile(self, name: str, q: float, **labels: str) -> float | None:
        fam = self.families.get(name)
        return histogram_quantile(fam, q, **labels) if fam else None

    def names(self) -> list[str]:
        return sorted(self.families)

    # -- 序列化 -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "families": {
                name: {
                    "type": fam.type,
                    "help": fam.help,
                    "series": [
                        {"name": s.name, "labels": s.labels, "value": s.value}
                        for s in fam.series
                    ],
                }
                for name, fam in self.families.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetricsSnapshot":
        families: dict[str, Family] = {}
        for name, raw in data.get("families", {}).items():
            fam = Family(name=name, type=raw.get("type", "untyped"), help=raw.get("help", ""))
            fam.series = [
                Series(name=s["name"], labels=s.get("labels", {}), value=float(s["value"]))
                for s in raw.get("series", [])
            ]
            families[name] = fam
        return cls(families=families, meta=data.get("meta", {}))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MetricsSnapshot":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save_tsv(self, path: str | Path, only_vllm: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["name\tlabels\tvalue"]
        for fam in self.families.values():
            for s in fam.series:
                if only_vllm and not s.name.startswith("vllm:"):
                    continue
                labels = ",".join(f"{k}={v}" for k, v in sorted(s.labels.items()))
                lines.append(f"{s.name}\t{labels}\t{s.value}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def parse_metrics_text(text: str, meta: dict[str, Any] | None = None) -> MetricsSnapshot:
    """解析 Prometheus exposition 文本。"""
    snap = MetricsSnapshot(meta=dict(meta or {}))
    pending_help: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = _HELP_RE.match(line)
            if m:
                pending_help[m.group(1)] = m.group(2).strip()
                continue
            m = _TYPE_RE.match(line)
            if m:
                base, kind = m.group(1), m.group(2) or "untyped"
                fam = snap.families.setdefault(base, Family(name=base, help=pending_help.get(base, "")))
                fam.type = kind
                if pending_help.get(base):
                    fam.help = pending_help[base]
                continue
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        labels = _parse_labels(m.group("labels"))
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        base = name
        for suffix in _HIST_SUFFIXES:
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                break
        fam = snap.families.get(base)
        if fam is None:
            fam = Family(name=base, help=pending_help.get(base, ""))
            snap.families[base] = fam
        fam.series.append(Series(name=name, labels=labels, value=value))
    if meta:
        snap.meta.update(meta)
    return snap


@dataclass
class QuantileDetail:
    """分位数估计结果，附带"这个估计有多可信"。"""

    value: float | None
    upper: float | None
    coarse: bool
    observations: float = 0.0


def histogram_quantile_detail(
    fam: Family | None, q: float, **labels: str
) -> QuantileDetail:
    """按 Prometheus ``histogram_quantile`` 的线性插值法估算分位数。

    本部署的部分直方图桶非常粗（例如 ``vllm:request_queue_time_seconds`` 的首桶
    是 ``le=0.3``），此时插值会给出"看起来精确、实际不可信"的数字。因此这里
    同时返回：

    * ``value``：插值结果（与 Prometheus 一致，便于和 Grafana 对齐）；
    * ``upper``：该分位所在桶的上界（保守上界）；
    * ``coarse``：是否落在首桶 / 与前一个桶跨度 >4×，即分辨率不足。
    """
    if fam is None:
        return QuantileDetail(None, None, False, 0.0)
    buckets = fam.buckets(**labels)
    if not buckets:
        return QuantileDetail(None, None, False, 0.0)
    total = buckets[-1][1]
    if total <= 0:
        return QuantileDetail(None, None, False, float(total))
    rank = q * total
    prev_bound = 0.0
    prev_count = 0.0
    for index, (bound, cumulative) in enumerate(buckets):
        if cumulative >= rank:
            if math.isinf(bound):
                return QuantileDetail(prev_bound, prev_bound, True, float(total))
            span = cumulative - prev_count
            if span > 0:
                value = prev_bound + (bound - prev_bound) * ((rank - prev_count) / span)
            else:
                value = bound
            coarse = index == 0 or (
                prev_bound > 0 and bound / max(prev_bound, 1e-12) > 4.0
            )
            return QuantileDetail(value, bound, bool(coarse), float(total))
        prev_bound, prev_count = bound, cumulative
    return QuantileDetail(prev_bound, prev_bound, True, float(total))


def histogram_quantile(fam: Family | None, q: float, **labels: str) -> float | None:
    """分位数插值结果（对粗桶场景请改用 :func:`histogram_quantile_detail`）。"""
    return histogram_quantile_detail(fam, q, **labels).value


def fetch_metrics(cfg: ProbeConfig, timeout: float | None = None) -> MetricsSnapshot:
    """抓取并解析 ``/metrics``（在线模式下使用）。"""
    import urllib.request

    headers = {"Accept": "text/plain"}
    # 反代/网关可能要求鉴权；实例开了 --api-key 时带上
    headers.update(cfg.auth_headers())
    request = urllib.request.Request(cfg.metrics_url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout or min(cfg.timeout, 30)) as resp:
        text = resp.read().decode("utf-8", errors="replace")
        snap = parse_metrics_text(
            text, meta={"url": cfg.metrics_url, "bytes": len(text)}
        )
    return snap


# --------------------------------------------------------------------------
# 推理阶段表：stage → 指标
# --------------------------------------------------------------------------

STAGE_METRICS: list[tuple[str, str, str]] = [
    ("① 队列等待 queue", "vllm:request_queue_time_seconds", "WAITING 阶段：排队等调度"),
    ("② 预填充 prefill", "vllm:request_prefill_time_seconds", "PREFILL 阶段：算 KV"),
    ("③ 解码 decode", "vllm:request_decode_time_seconds", "DECODE 阶段：逐 token 生成"),
    ("④ 运行合计 inference", "vllm:request_inference_time_seconds", "RUNNING = prefill + decode"),
    ("⑤ 端到端 e2e", "vllm:e2e_request_latency_seconds", "进入引擎到返回结束"),
    ("⑥ 首 token TTFT", "vllm:time_to_first_token_seconds", "首次出 token 的延迟"),
    ("⑦ token 间隔 ITL", "vllm:inter_token_latency_seconds", "相邻输出 token 间隔"),
    ("⑧ 每 token 时间 TPOT", "vllm:request_time_per_output_token_seconds", "解码期平均每 token"),
]

QUANTILES = (0.5, 0.9, 0.99)


def stage_table(snap: MetricsSnapshot) -> list[dict[str, Any]]:
    """把生命周期直方图整理成表格行。"""
    rows: list[dict[str, Any]] = []
    for stage, name, desc in STAGE_METRICS:
        fam = snap.family(name)
        if fam is None:
            rows.append(
                {"stage": stage, "metric": name, "desc": desc, "count": 0, "available": False}
            )
            continue
        count = sum(s.value for s in fam.series if s.name.endswith("_count"))
        total = sum(s.value for s in fam.series if s.name.endswith("_sum"))
        row: dict[str, Any] = {
            "stage": stage,
            "metric": name,
            "desc": desc,
            "available": True,
            "count": count,
            "sum_s": total,
            "mean_ms": (total / count * 1000.0) if count else None,
        }
        for q in QUANTILES:
            detail = histogram_quantile_detail(fam, q)
            key = f"p{int(q * 100)}"
            row[f"{key}_ms"] = None if detail.value is None else detail.value * 1000.0
            row[f"{key}_upper_ms"] = None if detail.upper is None else detail.upper * 1000.0
            row[f"{key}_coarse"] = detail.coarse
        rows.append(row)
    return rows


def engine_state(snap: MetricsSnapshot) -> dict[str, Any]:
    """引擎瞬时状态：队列 / KV / 前缀缓存 / 抢占 / 吞吐计数。"""
    hits = snap.total("vllm:prefix_cache_hits_total")
    queries = snap.total("vllm:prefix_cache_queries_total")
    ext_hits = snap.total("vllm:external_prefix_cache_hits_total")
    ext_queries = snap.total("vllm:external_prefix_cache_queries_total")
    drafted = snap.total("vllm:spec_decode_num_draft_tokens_total")
    accepted = snap.total("vllm:spec_decode_num_accepted_tokens_total")
    drafts = snap.total("vllm:spec_decode_num_drafts_total")
    return {
        "num_requests_running": snap.total("vllm:num_requests_running"),
        "num_requests_waiting": snap.total("vllm:num_requests_waiting"),
        "num_requests_waiting_by_reason": {
            s.labels.get("reason", "?"): s.value
            for s in (snap.family("vllm:num_requests_waiting_by_reason").series
                      if snap.family("vllm:num_requests_waiting_by_reason") else [])
        },
        "kv_cache_usage_perc": snap.total("vllm:kv_cache_usage_perc"),
        # 注意：本版本（0.18.x）不再导出 cpu_cache_usage_perc / gpu_cache_usage_perc，
        # 用 value() 而不是 total()，缺失时返回 None 而不是误导性的 0.0。
        "cpu_cache_usage_perc": snap.value("vllm:cpu_cache_usage_perc"),
        "prefix_cache_hits": hits,
        "prefix_cache_queries": queries,
        "prefix_cache_hit_rate": (hits / queries) if queries else None,
        "external_prefix_cache_hit_rate": (ext_hits / ext_queries) if ext_queries else None,
        "num_preemptions_total": snap.total("vllm:num_preemptions_total"),
        "prompt_tokens_total": snap.total("vllm:prompt_tokens_total"),
        "generation_tokens_total": snap.total("vllm:generation_tokens_total"),
        "prompt_tokens_cached_total": snap.total("vllm:prompt_tokens_cached_total"),
        "request_success_total": snap.total("vllm:request_success_total"),
        "spec_decode": {
            "drafts": drafts,
            "draft_tokens": drafted,
            "accepted_tokens": accepted,
            "draft_acceptance_rate": (accepted / drafted) if drafted else None,
            "mean_acceptance_length": (1.0 + accepted / drafts) if drafts else None,
            "per_position_accepted": {
                s.labels.get("position", "?"): s.value
                for s in (snap.family("vllm:spec_decode_num_accepted_tokens_per_pos_total").series
                          if snap.family("vllm:spec_decode_num_accepted_tokens_per_pos_total") else [])
            },
        },
        "iteration_tokens": _iteration_tokens(snap),
    }


def _iteration_tokens(snap: MetricsSnapshot) -> dict[str, Any]:
    fam = snap.family("vllm:iteration_tokens_total")
    if fam is None:
        return {}
    count = sum(s.value for s in fam.series if s.name.endswith("_count"))
    total = sum(s.value for s in fam.series if s.name.endswith("_sum"))
    return {
        "steps": count,
        "tokens": total,
        "tokens_per_step": (total / count) if count else None,
        "p50_tokens_per_step": histogram_quantile(fam, 0.5),
        "p90_tokens_per_step": histogram_quantile(fam, 0.9),
    }


def kv_block_metrics(snap: MetricsSnapshot) -> dict[str, Any]:
    """KV block 驻留指标（需要 ``--kv-cache-metrics``）。"""
    out: dict[str, Any] = {}
    for key, name in (
        ("lifetime", "vllm:kv_block_lifetime_seconds"),
        ("idle_before_evict", "vllm:kv_block_idle_before_evict_seconds"),
        ("reuse_gap", "vllm:kv_block_reuse_gap_seconds"),
    ):
        fam = snap.family(name)
        if fam is None:
            continue
        out[key] = {
            "count": sum(s.value for s in fam.series if s.name.endswith("_count")),
            "mean_ms": None if fam.mean() is None else fam.mean() * 1000.0,
            "p50_ms": _maybe_ms(histogram_quantile(fam, 0.5)),
            "p90_ms": _maybe_ms(histogram_quantile(fam, 0.9)),
        }
    return out


def _maybe_ms(value: float | None) -> float | None:
    return None if value is None else value * 1000.0


# --------------------------------------------------------------------------
# 两次快照做差
# --------------------------------------------------------------------------

DROP_INTERVAL_METRICS = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:cpu_cache_usage_perc",
    "vllm:cache_config_info",
    "vllm:num_requests_waiting_by_reason",
}


def histogram_unit(name: str) -> str:
    """直方图的单位：只有 ``*_seconds`` 是时间（秒→毫秒），其余按原值。

    vLLM 把 token 数、字节数也导出成 histogram（例如 ``vllm:request_prompt_tokens``），
    早期实现对所有 histogram 都乘 1000，导致报告出现 "均值 709000.0 ms" 这种错误。
    """
    base = str(name or "")
    if base.endswith("_seconds"):
        return "s"
    if "tokens" in base:
        return "tokens"
    if "bytes" in base:
        return "bytes"
    return "count"


def diff_snapshots(
    before: MetricsSnapshot,
    after: MetricsSnapshot,
    elapsed_s: float,
    model: str | None = None,
) -> dict[str, Any]:
    """计算计数器增量与速率（用于一次探测前后对比）。"""
    result: dict[str, Any] = {"elapsed_s": elapsed_s, "counters": {}, "rates": {}, "histograms": {}}
    names = set(before.families) | set(after.families)
    for name in sorted(names):
        if name in DROP_INTERVAL_METRICS or name.endswith("_created"):
            continue
        fb, fa = before.family(name), after.family(name)
        if fb is None or fa is None:
            continue
        if fb.type in ("counter",) or name.endswith("_total"):
            delta = fa.total() - fb.total()
            result["counters"][name] = delta
            result["rates"][name] = delta / elapsed_s if elapsed_s > 0 else None
        elif fb.type == "histogram" and name.startswith("vllm:"):
            cb = sum(s.value for s in fb.series if s.name.endswith("_count"))
            ca = sum(s.value for s in fa.series if s.name.endswith("_count"))
            sb = sum(s.value for s in fb.series if s.name.endswith("_sum"))
            sa = sum(s.value for s in fa.series if s.name.endswith("_sum"))
            if ca - cb > 0:
                mean = (sa - sb) / (ca - cb)
                unit = histogram_unit(name)
                result["histograms"][name] = {
                    "count": ca - cb,
                    "unit": unit,
                    "mean": mean,
                    "mean_ms": mean * 1000.0 if unit == "s" else None,
                }
    model_name = model or (after.meta.get("model") if after.meta else None)
    result["model"] = model_name
    return result


# --------------------------------------------------------------------------
# 保存 / 打印
# --------------------------------------------------------------------------

def save_snapshot(cfg: ProbeConfig, tag: str = "metrics") -> tuple[Path, MetricsSnapshot]:
    import time as _time

    snap = fetch_metrics(cfg)
    snap.meta.update(
        {
            "tag": tag,
            "model": cfg.model,
            "base_url": cfg.base_url,
            "fetched_at": _time.time(),
            "fetched_at_iso": _time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    json_path = snap.save(cfg.out_path(f"{tag}.json"))
    snap.save_tsv(cfg.out_path(f"{tag}.tsv"))
    return json_path, snap


def _display_width(text: str) -> int:
    """中日韩字符按 2 列宽计算，用于对齐表格。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def _pad(text: str, width: int, align: str = "left") -> str:
    space = max(0, width - _display_width(text))
    return (text + " " * space) if align == "left" else (" " * space + text)


def format_stage_table(rows: Iterable[dict[str, Any]]) -> str:
    rows = list(rows)
    columns = [("阶段", 22, "left"), ("样本", 7, "right"), ("均值ms", 11, "right"),
               ("p50", 11, "right"), ("p90", 11, "right"), ("p99", 11, "right")]
    lines = ["".join(_pad(title, width, align) for title, width, align in columns)]
    lines.append("-" * sum(width for _, width, _ in columns))
    coarse_seen = False
    for row in rows:
        if not row.get("available"):
            lines.append(_pad(row["stage"], 22) + _pad("—", 7, "right") + _pad("（该指标未导出）", 11, "right"))
            continue

        def fmt(key: str) -> str:
            value = row.get(key)
            if value is None:
                return "—"
            base = key[: -len("_ms")] if key.endswith("_ms") else key
            if row.get(f"{base}_coarse"):
                upper = row.get(f"{base}_upper_ms")
                return f"≤{upper:.0f}" if upper is not None else f"{value:.1f}"
            return f"{value:.1f}"

        for q in QUANTILES:
            if row.get(f"p{int(q * 100)}_coarse"):
                coarse_seen = True
        lines.append(
            _pad(row["stage"], 22)
            + _pad(str(int(row["count"])), 7, "right")
            + _pad(fmt("mean_ms"), 11, "right")
            + _pad(fmt("p50_ms"), 11, "right")
            + _pad(fmt("p90_ms"), 11, "right")
            + _pad(fmt("p99_ms"), 11, "right")
        )
    if coarse_seen:
        lines.append(
            "\n注：带 ≤ 的分位数落在该直方图的首个桶/超大跨度桶内，桶阶梯分辨率不足，"
            "\n    数值只是上界；此时请以「均值」或「服务端逐请求指标」为准。"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from .cli import main as cli_main

    return cli_main(["metrics"] + (argv if argv is not None else sys.argv[1:]))


if __name__ == "__main__":  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
