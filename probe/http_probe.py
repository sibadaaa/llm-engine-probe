"""请求层探测：客户端视角的完整请求生命周期。

这个模块回答的问题：**"从客户端发出请求到最后一个 token 回来，时间花在哪？"**

它按 SSE 逐 chunk 打点，得到：

* ``t_send → t_headers``    HTTP 建连/首字节（网络 + API server 开销）
* ``t_headers → 首个内容块``  TTFT（含队列等待 + 预填充）
* 内容块之间的间隔            ITL（token 间隔），用于看解码是否稳定
* ``最后一个块``              E2E 总时长

如果服务端开了 ``--enable-per-request-metrics``，响应里还会带服务端自己算的
``metrics``（queue_time_ms / time_to_first_token_ms / generation_time_ms /
mean_itl_ms / tokens_per_second），两者可以互相校验；开了
``--per-request-spec-decode-metrics detailed`` 时还会带
``metrics.speculative_decoding``（含逐步接受数组）。

只依赖标准库。用法::

    python3 -m probe http --concurrency 8 --prompt-tokens 4096 --max-tokens 128
"""

from __future__ import annotations

import http.client
import json
import statistics
import ssl
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .config import ProbeConfig

# --------------------------------------------------------------------------
# 负载构造
# --------------------------------------------------------------------------

_WORDS = (
    "the quick brown fox jumps over the lazy dog while seventeen engineers "
    "review pipeline stage number for latency regressions and throughput"
).split()


def build_prompt(target_tokens: int, shared_prefix_tokens: int = 0, index: int = 0) -> str:
    """构造确定性长 prompt（token 密度稳定，便于跨实验对比）。

    ``shared_prefix_tokens`` 会生成一段对**所有请求都相同**的前缀，用于探测
    前缀缓存（APC）命中：第 2 个及之后的请求应当命中缓存、TTFT 明显下降。
    """
    lines: list[str] = []
    approx_per_line = 22  # 一行约 22 个 token，与 scripts/bench_e2e.py 的口径一致
    total_lines = max(1, int(round(target_tokens / approx_per_line)))
    for i in range(total_lines):
        lines.append(
            f"Line {i}: the quick brown fox jumps over the lazy dog while "
            f"seventeen engineers review pipeline stage number {i} for latency regressions."
        )
    text = "\n".join(lines)
    if shared_prefix_tokens > 0:
        head_lines: list[str] = []
        for i in range(max(1, shared_prefix_tokens // approx_per_line)):
            head_lines.append(
                f"SHARED-PREFIX block {i}: static system context that every request in this "
                f"probe reuses verbatim so the prefix cache can be observed."
            )
        prefix = "\n".join(head_lines)
        text = prefix + f"\nUNIQUE-SUFFIX-{index}\n" + text
    return text


# --------------------------------------------------------------------------
# 单次请求
# --------------------------------------------------------------------------


@dataclass
class RequestResult:
    index: int = 0
    ok: bool = False
    error: str | None = None
    status: int | None = None
    request_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    ttft_ms: float | None = None
    headers_ms: float | None = None
    e2e_ms: float | None = None
    decode_ms: float | None = None
    tpot_ms: float | None = None
    itl_ms: list[float] = field(default_factory=list)
    chunks: int = 0
    finish_reason: str | None = None
    server_metrics: dict[str, Any] | None = None
    spec_decode: dict[str, Any] | None = None
    started_at: float = 0.0

    def itl_stats(self) -> dict[str, float | None]:
        values = [v for v in self.itl_ms if v is not None]
        if not values:
            return {"count": 0, "mean_ms": None, "p50_ms": None, "p90_ms": None, "max_ms": None}
        return {
            "count": len(values),
            "mean_ms": statistics.fmean(values),
            "p50_ms": percentile(values, 50),
            "p90_ms": percentile(values, 90),
            "max_ms": max(values),
        }


def percentile(values: Iterable[float], q: float) -> float | None:
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * (q / 100.0)
    low = int(pos)
    high = min(low + 1, len(data) - 1)
    if low == high:
        return data[low]
    return data[low] + (data[high] - data[low]) * (pos - low)


def _split_url(base_url: str) -> tuple[str, str, int, bool]:
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path_prefix = parsed.path.rstrip("/")
    return host, path_prefix, port, parsed.scheme == "https"


def _open_connection(cfg: ProbeConfig) -> tuple[http.client.HTTPConnection, str]:
    host, prefix, port, use_tls = _split_url(cfg.base_url)
    if use_tls:
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(
            host, port, timeout=cfg.timeout, context=ssl.create_default_context()
        )
    else:
        conn = http.client.HTTPConnection(host, port, timeout=cfg.timeout)
    conn.connect()
    return conn, prefix


def _build_body(cfg: ProbeConfig, prompt: str, max_tokens: int, stream: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": cfg.model,
        "max_tokens": max_tokens,
        "temperature": cfg.temperature,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    if cfg.endpoint == "chat":
        body["messages"] = [{"role": "user", "content": prompt}]
        body["chat_template_kwargs"] = {"reasoning_effort": "low"}
    else:
        body["prompt"] = prompt
    return body


def request_once(
    cfg: ProbeConfig,
    index: int = 0,
    prompt: str | None = None,
    max_tokens: int | None = None,
    stream: bool | None = None,
) -> RequestResult:
    """发一次请求并逐块打点。异常一律收敛成 ``ok=False``，不抛出。"""
    max_tokens = cfg.max_tokens if max_tokens is None else max_tokens
    stream = cfg.stream if stream is None else stream
    if prompt is None:
        prompt = build_prompt(cfg.prompt_tokens, cfg.shared_prefix_tokens, index)

    result = RequestResult(index=index, started_at=time.time())
    conn = None
    try:
        conn, prefix = _open_connection(cfg)
        body = json.dumps(_build_body(cfg, prompt, max_tokens, stream)).encode("utf-8")
        path = prefix + ("/v1/chat/completions" if cfg.endpoint == "chat" else "/v1/completions")
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "User-Agent": "vllm-docker-probe/1.0",
        }
        # 只有配置了真实 key 才带 Authorization（无鉴权实例不再发占位 "Bearer EMPTY"）
        headers.update(cfg.auth_headers())
        t0 = time.perf_counter()
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        result.status = resp.status
        result.headers_ms = (time.perf_counter() - t0) * 1000.0
        result.request_id = resp.getheader("x-request-id") or resp.getheader("x-vllm-request-id")

        if resp.status != 200:
            detail = resp.read(4096).decode("utf-8", errors="replace")
            result.error = f"HTTP {resp.status}: {detail[:500]}"
            result.e2e_ms = (time.perf_counter() - t0) * 1000.0
            return result

        if not stream:
            payload = json.loads(resp.read().decode("utf-8"))
            result.e2e_ms = (time.perf_counter() - t0) * 1000.0
            _absorb_payload(result, payload)
            result.ok = True
            return result

        last_ts = t0
        first_content_ms: float | None = None
        while True:
            raw = resp.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            now = time.perf_counter()
            has_content = _absorb_chunk(result, chunk)
            if has_content:
                if first_content_ms is None:
                    first_content_ms = (now - t0) * 1000.0
                    result.ttft_ms = first_content_ms
                else:
                    result.itl_ms.append((now - last_ts) * 1000.0)
                result.chunks += 1
                last_ts = now
        result.e2e_ms = (time.perf_counter() - t0) * 1000.0
        if result.ttft_ms is not None and result.e2e_ms is not None:
            result.decode_ms = max(0.0, result.e2e_ms - result.ttft_ms)
            tokens = result.completion_tokens
            if tokens and tokens > 1 and result.decode_ms is not None:
                result.tpot_ms = result.decode_ms / (tokens - 1)
            elif result.chunks > 1:
                result.tpot_ms = result.decode_ms / (result.chunks - 1)
        result.ok = result.ttft_ms is not None or (result.completion_tokens or 0) > 0
        if not result.ok and result.error is None:
            result.error = "未收到任何内容块"
        return result
    except Exception as exc:  # noqa: BLE001 - 探测工具必须容错
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass


def _absorb_payload(result: RequestResult, payload: dict[str, Any]) -> None:
    """非流式响应：抽取 usage / metrics。"""
    usage = payload.get("usage") or {}
    result.prompt_tokens = usage.get("prompt_tokens")
    result.completion_tokens = usage.get("completion_tokens")
    result.cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    result.finish_reason = _finish_reason(payload)
    if isinstance(payload.get("metrics"), dict):
        result.server_metrics = payload["metrics"]
        if isinstance(result.server_metrics.get("speculative_decoding"), dict):
            result.spec_decode = result.server_metrics["speculative_decoding"]
    result.request_id = result.request_id or payload.get("id")


def _absorb_chunk(result: RequestResult, chunk: dict[str, Any]) -> bool:
    """流式 chunk：返回该块是否携带了新的输出内容（含 reasoning）。"""
    got_content = False
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        # 注意 delta 键名的三种形态：OpenAI 的 content、DeepSeek 系的
        # reasoning_content、以及 vLLM 本仓库 reasoning parser（qwen3）输出的
        # **reasoning**。现网 qwen3.8-27b 在 reasoning_effort=low 时前若干个 chunk
        # 全部走 reasoning 键；漏掉它会导致客户端 TTFT/ITL 全是 None（实测踩过）。
        text = (
            delta.get("content")
            or delta.get("reasoning_content")
            or delta.get("reasoning")
            or ""
        )
        if not text and choice.get("text"):
            text = choice["text"]
        if text:
            got_content = True
        if choice.get("finish_reason"):
            result.finish_reason = choice["finish_reason"]
    usage = chunk.get("usage")
    if usage:
        result.prompt_tokens = usage.get("prompt_tokens", result.prompt_tokens)
        result.completion_tokens = usage.get("completion_tokens", result.completion_tokens)
        details = usage.get("prompt_tokens_details") or {}
        if "cached_tokens" in details:
            result.cached_tokens = details["cached_tokens"]
    if isinstance(chunk.get("metrics"), dict):
        result.server_metrics = chunk["metrics"]
        if isinstance(result.server_metrics.get("speculative_decoding"), dict):
            result.spec_decode = result.server_metrics["speculative_decoding"]
    result.request_id = result.request_id or chunk.get("id")
    return got_content


def _finish_reason(payload: dict[str, Any]) -> str | None:
    for choice in payload.get("choices") or []:
        if choice.get("finish_reason"):
            return choice["finish_reason"]
    return None


# --------------------------------------------------------------------------
# 批量矩阵
# --------------------------------------------------------------------------


def run_matrix(
    cfg: ProbeConfig,
    warmup: int = 1,
    progress: bool = True,
) -> tuple[list[RequestResult], dict[str, Any]]:
    """按 ``concurrency × repeat`` 压测，返回逐请求结果与汇总。"""
    total = max(1, cfg.concurrency) * max(1, cfg.repeat)
    results: list[RequestResult] = []
    lock = threading.Lock()

    for _ in range(max(0, warmup)):
        request_once(cfg, index=-1, max_tokens=min(8, cfg.max_tokens))

    def job(i: int) -> RequestResult:
        item = request_once(cfg, index=i)
        if progress:
            with lock:
                flag = "ok " if item.ok else "ERR"
                ttft = "—" if item.ttft_ms is None else f"{item.ttft_ms:.0f}ms"
                print(
                    f"  [{i + 1}/{total}] {flag} ttft={ttft} "
                    f"e2e={0 if item.e2e_ms is None else item.e2e_ms:.0f}ms "
                    f"tok={item.completion_tokens} {('err=' + item.error) if item.error else ''}",
                    flush=True,
                )
        return item

    with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as pool:
        for item in pool.map(job, range(total)):
            results.append(item)

    return results, summarize(results, cfg)


def summarize(results: list[RequestResult], cfg: ProbeConfig) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    def pct(attr: str, q: float) -> float | None:
        return percentile([getattr(r, attr) for r in ok if getattr(r, attr) is not None], q)

    itl_all: list[float] = []
    for r in ok:
        itl_all.extend(r.itl_ms)

    summary: dict[str, Any] = {
        "config": cfg.dump(),
        "requests": len(results),
        "ok": len(ok),
        "failed": len(failed),
        "errors": [r.error for r in failed][:5],
        "ttft_ms": {f"p{q}": pct("ttft_ms", q) for q in (50, 90, 99)},
        "e2e_ms": {f"p{q}": pct("e2e_ms", q) for q in (50, 90, 99)},
        "decode_ms": {f"p{q}": pct("decode_ms", q) for q in (50, 90, 99)},
        "headers_ms": {f"p{q}": pct("headers_ms", q) for q in (50, 90, 99)},
        "tpot_ms": {f"p{q}": pct("tpot_ms", q) for q in (50, 90, 99)},
        "itl_ms": {
            "count": len(itl_all),
            "mean_ms": statistics.fmean(itl_all) if itl_all else None,
            "p50_ms": percentile(itl_all, 50),
            "p90_ms": percentile(itl_all, 90),
            "p99_ms": percentile(itl_all, 99),
            "max_ms": max(itl_all) if itl_all else None,
        },
        "prompt_tokens": [r.prompt_tokens for r in ok if r.prompt_tokens is not None][:1],
        "completion_tokens_avg": (
            statistics.fmean([r.completion_tokens for r in ok if r.completion_tokens])
            if any(r.completion_tokens for r in ok)
            else None
        ),
        "cached_tokens": [r.cached_tokens for r in ok if r.cached_tokens is not None],
        "server_side": _server_side_summary(ok),
        "spec_decode": _spec_decode_summary(ok),
        "throughput": _throughput(results, ok, cfg),
    }
    return summary


def _server_side_summary(ok: list[RequestResult]) -> dict[str, Any]:
    """服务端 ``--enable-per-request-metrics`` 给出的逐请求指标聚合。"""
    rows = [r.server_metrics for r in ok if r.server_metrics]
    if not rows:
        return {"available": False, "hint": "服务端未开启 --enable-per-request-metrics"}
    out: dict[str, Any] = {"available": True, "requests": len(rows)}
    for key in (
        "queue_time_ms",
        "time_to_first_token_ms",
        "generation_time_ms",
        "mean_itl_ms",
        "tokens_per_second",
    ):
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            out[key] = {
                "p50": percentile(values, 50),
                "p90": percentile(values, 90),
                "mean": statistics.fmean(values),
            }
    return out


def _spec_decode_summary(ok: list[RequestResult]) -> dict[str, Any]:
    rows = [r.spec_decode for r in ok if r.spec_decode]
    if not rows:
        return {"available": False, "hint": "需要 --per-request-spec-decode-metrics summary|detailed 且启用投机解码"}
    hist_len = max((len(row.get("acceptance_histogram") or []) for row in rows), default=0)
    hist = [0] * hist_len
    for row in rows:
        for i, value in enumerate(row.get("acceptance_histogram") or []):
            hist[i] += value
    per_step_accepted: list[int] = []
    per_step_drafted: list[int] = []
    for row in rows:
        per_step_accepted.extend(row.get("per_step_accepted") or [])
        per_step_drafted.extend(row.get("per_step_drafted") or [])
    return {
        "available": True,
        "requests": len(rows),
        "mean_acceptance_length": statistics.fmean(
            [row["mean_acceptance_length"] for row in rows if row.get("mean_acceptance_length") is not None]
        ),
        "draft_acceptance_rate": statistics.fmean(
            [row["draft_acceptance_rate"] for row in rows if row.get("draft_acceptance_rate") is not None]
        ),
        "acceptance_histogram": hist,
        "num_spec_steps": sum(row.get("num_spec_steps") or 0 for row in rows),
        "num_draft_tokens": sum(row.get("num_draft_tokens") or 0 for row in rows),
        "num_accepted_draft_tokens": sum(row.get("num_accepted_draft_tokens") or 0 for row in rows),
        "per_step_accepted": per_step_accepted,
        "per_step_drafted": per_step_drafted,
    }


def _throughput(
    results: list[RequestResult], ok: list[RequestResult], cfg: ProbeConfig
) -> dict[str, Any]:
    if not results:
        return {}
    span = max((r.started_at for r in results), default=0) - min(
        (r.started_at for r in results), default=0
    )
    last_end = max((r.started_at + (r.e2e_ms or 0) / 1000.0 for r in results), default=0)
    first_start = min((r.started_at for r in results), default=0)
    wall = max(1e-6, last_end - first_start)
    gen_tokens = sum(r.completion_tokens or 0 for r in ok)
    prompt_tokens = sum(r.prompt_tokens or 0 for r in ok)
    return {
        "wall_s": wall,
        "client_span_s": span,
        "generation_tokens": gen_tokens,
        "prompt_tokens": prompt_tokens,
        "output_tokens_per_s": gen_tokens / wall,
        "total_tokens_per_s": (gen_tokens + prompt_tokens) / wall,
        "requests_per_s": len(ok) / wall,
        "concurrency": cfg.concurrency,
    }


def results_to_jsonl(results: list[RequestResult], path: str) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for item in results:
            fh.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def format_summary(summary: dict[str, Any]) -> str:
    """把汇总渲染成一段人类可读文本（CLI 打印用）。"""
    def fmt(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.1f}"
        return str(value)

    lines = [
        f"请求数 {summary['ok']}/{summary['requests']} 成功",
        f"TTFT  p50={fmt(summary['ttft_ms']['p50'])}ms  p90={fmt(summary['ttft_ms']['p90'])}ms  "
        f"p99={fmt(summary['ttft_ms']['p99'])}ms",
        f"E2E   p50={fmt(summary['e2e_ms']['p50'])}ms  p90={fmt(summary['e2e_ms']['p90'])}ms  "
        f"p99={fmt(summary['e2e_ms']['p99'])}ms",
        f"TPOT  p50={fmt(summary['tpot_ms']['p50'])}ms  ITL 均值={fmt(summary['itl_ms']['mean_ms'])}ms"
        f"  p99={fmt(summary['itl_ms']['p99_ms'])}ms",
        f"吞吐  输出={fmt(summary['throughput'].get('output_tokens_per_s'))} tok/s  "
        f"总={fmt(summary['throughput'].get('total_tokens_per_s'))} tok/s  "
        f"请求={fmt(summary['throughput'].get('requests_per_s'))} req/s",
    ]
    server = summary.get("server_side") or {}
    if server.get("available"):
        queue = server.get("queue_time_ms", {})
        lines.append(
            f"服务端  queue p50={fmt(queue.get('p50'))}ms  "
            f"TTFT p50={fmt((server.get('time_to_first_token_ms') or {}).get('p50'))}ms  "
            f"generation p50={fmt((server.get('generation_time_ms') or {}).get('p50'))}ms"
        )
    spec = summary.get("spec_decode") or {}
    if spec.get("available"):
        lines.append(
            f"投机解码  平均接受长度={fmt(spec.get('mean_acceptance_length'))}  "
            f"草稿接受率={fmt((spec.get('draft_acceptance_rate') or 0) * 100)}%  "
            f"直方图={spec.get('acceptance_histogram')}"
        )
    if summary.get("cached_tokens"):
        lines.append(f"前缀缓存命中 token 数序列: {summary['cached_tokens']}")
    if summary.get("errors"):
        lines.append(f"错误示例: {summary['errors']}")
    return "\n".join(lines)
