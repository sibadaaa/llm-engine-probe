"""本地 mock vLLM 服务：用于在没有 GPU 的机器上端到端自测探针链路。

它实现探针真正会用到的端点与响应形状（与 vLLM 0.18.x 的 OpenAI 兼容接口一致）：

* ``GET  /health`` / ``GET /v1/models`` / ``GET /version``
* ``GET  /metrics``             Prometheus 文本（含真实指标名，计数器随请求增长）
* ``POST /v1/chat/completions``  SSE 流式：逐 token 输出，最后一块带
                                ``usage`` + ``metrics`` + ``speculative_decoding``
* ``POST /v1/completions``       同上（非流式也支持）
* ``POST /start_profile`` / ``POST /stop_profile``
                                在 ``MOCK_TRACE_DIR`` 里生成一个假的 Chrome trace

用法::

    python3 tests/mock_vllm_server.py --port 18000
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL = os.environ.get("MOCK_MODEL", "qwen3.8-27b")
STATE = {"requests": 0, "prompt_tokens": 0, "generation_tokens": 0, "cached_tokens": 0,
         "drafts": 0, "draft_tokens": 0, "accepted_tokens": 0, "profiling": False}
LOCK = threading.Lock()
TRACE_DIR = Path(os.environ.get("MOCK_TRACE_DIR", "/tmp/mock-vllm-traces"))


def metrics_text() -> str:
    """生成一份与真实部署同形状的 /metrics 文本（计数器随请求增长）。"""
    with LOCK:
        st = dict(STATE)
    n = st["requests"]
    lines: list[str] = []

    def family(name: str, kind: str, help_text: str, samples: list[tuple[dict, float]]) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {kind}")
        for labels, value in samples:
            rendered = ",".join(f'{k}="{v}"' for k, v in labels.items())
            lines.append(f"{name}{{{rendered}}} {value}")

    base = {"engine": "0", "model_name": MODEL}
    family("vllm:num_requests_running", "gauge", "Number of requests in model execution batches.",
           [(base, 0.0)])
    family("vllm:num_requests_waiting", "gauge", "Number of requests waiting to be processed.",
           [(base, 0.0)])
    family("vllm:num_requests_waiting_by_reason", "gauge", "Number of waiting requests by reason.",
           [({**base, "reason": "capacity"}, 0.0), ({**base, "reason": "deferred"}, 0.0)])
    family("vllm:kv_cache_usage_perc", "gauge", "KV-cache usage. 1 means 100 percent usage.",
           [(base, 0.12)])
    family("vllm:cache_config_info", "gauge", "Information of the LLMEngine CacheConfig",
           [({**base, "block_size": "16", "enable_prefix_caching": "True",
              "kv_cache_size_tokens": "737849"}, 1.0)])
    family("vllm:prompt_tokens_total", "counter", "Number of prefill tokens processed.",
           [(base, float(st["prompt_tokens"]))])
    family("vllm:generation_tokens_total", "counter", "Number of generation tokens processed.",
           [(base, float(st["generation_tokens"]))])
    family("vllm:prompt_tokens_cached_total", "counter", "Number of cached prompt tokens.",
           [(base, float(st["cached_tokens"]))])
    family("vllm:prefix_cache_queries_total", "counter", "Prefix cache queries.",
           [(base, float(st["prompt_tokens"]))])
    family("vllm:prefix_cache_hits_total", "counter", "Prefix cache hits.",
           [(base, float(st["cached_tokens"]))])
    family("vllm:num_preemptions_total", "counter", "Cumulative number of preemption.",
           [(base, 0.0)])
    family("vllm:request_success_total", "counter", "Count of successfully processed requests.",
           [({**base, "finished_reason": "stop"}, float(n))])

    # 生命周期直方图（桶阶梯与真实部署一致）
    base_labels = ",".join(f'{k}="{v}"' for k, v in base.items())

    def hist(name: str, help_text: str, buckets: list[float], observations: list[float]) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} histogram")
        for bound in buckets:
            count = sum(1 for v in observations if v <= bound)
            rendered = ",".join(f'{k}="{v}"' for k, v in {**base, "le": bound}.items())
            lines.append(f"{name}_bucket{{{rendered}}} {float(count)}")
        inf = ",".join(f'{k}="{v}"' for k, v in {**base, "le": "+Inf"}.items())
        lines.append(f"{name}_bucket{{{inf}}} {float(len(observations))}")
        lines.append(f"{name}_count{{{base_labels}}} {float(len(observations))}")
        lines.append(f"{name}_sum{{{base_labels}}} {sum(observations)}")

    obs = [0.02 + 0.01 * i for i in range(max(1, n))]
    hist("vllm:request_queue_time_seconds", "Histogram of time spent in WAITING phase for request.",
         [0.3, 0.5, 0.8, 1.0, 2.0, 5.0], obs)
    hist("vllm:request_prefill_time_seconds", "Histogram of time spent in PREFILL phase for request.",
         [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0], [0.3 + 0.05 * i for i in range(max(1, n))])
    hist("vllm:request_decode_time_seconds", "Histogram of time spent in DECODE phase for request.",
         [0.1, 0.5, 1.0, 2.0, 5.0], [1.2 + 0.1 * i for i in range(max(1, n))])
    hist("vllm:request_inference_time_seconds", "Histogram of time spent in RUNNING phase for request.",
         [0.1, 0.5, 1.0, 2.0, 5.0], [1.5 + 0.1 * i for i in range(max(1, n))])
    hist("vllm:e2e_request_latency_seconds", "Histogram of e2e request latency in seconds.",
         [0.1, 0.5, 1.0, 2.0, 5.0], [1.6 + 0.1 * i for i in range(max(1, n))])
    hist("vllm:time_to_first_token_seconds", "Histogram of time to first token in seconds.",
         [0.05, 0.1, 0.25, 0.5, 1.0, 2.5], [0.35 + 0.05 * i for i in range(max(1, n))])
    hist("vllm:inter_token_latency_seconds", "Histogram of inter-token latency in seconds.",
         [0.01, 0.05, 0.1, 0.25, 0.5], [0.05 + 0.001 * i for i in range(max(1, n) * 4)])
    hist("vllm:request_time_per_output_token_seconds",
         "Histogram of time_per_output_token_seconds per request.",
         [0.01, 0.05, 0.1, 0.25, 0.5], [0.06 + 0.001 * i for i in range(max(1, n))])
    hist("vllm:iteration_tokens_total", "Histogram of number of tokens per engine_step.",
         [1.0, 8.0, 16.0, 32.0, 64.0, 128.0], [8.0] * max(1, n))

    # 投机解码
    lines.append("# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.")
    lines.append("# TYPE vllm:spec_decode_num_drafts_total counter")
    lines.append(
        f"vllm:spec_decode_num_drafts_total{{engine=\"0\",model_name=\"{MODEL}\"}} {float(st['drafts'])}"
    )
    for name, value in (
        ("vllm:spec_decode_num_draft_tokens_total", st["draft_tokens"]),
        ("vllm:spec_decode_num_accepted_tokens_total", st["accepted_tokens"]),
    ):
        lines.append(f"# HELP {name} {'Number of draft tokens.' if 'draft_tokens' in name else 'Number of accepted tokens.'}")
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name}{{engine=\"0\",model_name=\"{MODEL}\"}} {float(value)}")
    for pos in range(7):
        lines.append(
            f"vllm:spec_decode_num_accepted_tokens_per_pos_total"
            f"{{engine=\"0\",model_name=\"{MODEL}\",position=\"{pos}\"}} "
            f"{float(max(0, 40 - pos * 12))}"
        )

    # MFU / KV block
    for name, help_text, value in (
        ("vllm:estimated_flops_per_gpu_total",
         "Estimated number of floating point operations per GPU.", 1.2e12),
        ("vllm:estimated_read_bytes_per_gpu_total",
         "Estimated number of bytes read from memory per GPU.", 3.4e10),
        ("vllm:estimated_write_bytes_per_gpu_total",
         "Estimated number of bytes written to memory per GPU.", 1.1e10),
    ):
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name}{{engine=\"0\",model_name=\"{MODEL}\"}} {value}")
    hist("vllm:kv_block_lifetime_seconds", "Histogram of KV cache block lifetime.",
         [0.1, 1.0, 10.0, 60.0, 600.0], [12.0, 30.0, 45.0])
    return "\n".join(lines) + "\n"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "MockVLLM/1.0"

    def log_message(self, fmt: str, *args) -> None:  # 静音
        if os.environ.get("MOCK_VERBOSE"):
            super().log_message(fmt, *args)

    # -- 辅助 -------------------------------------------------------------
    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode() or "{}")
        except json.JSONDecodeError:
            return {}

    # -- GET --------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/health":
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"OK")
        elif path == "/v1/models":
            self._json({"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        elif path == "/version":
            self._json({"version": "0.18.1-mock"})
        elif path == "/metrics":
            body = metrics_text().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json({"error": "not found"}, 404)

    # -- POST -------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/start_profile", "/stop_profile"):
            with LOCK:
                STATE["profiling"] = path == "/start_profile"
            if path == "/stop_profile":
                write_fake_trace()
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path in ("/v1/chat/completions", "/v1/completions"):
            self.handle_inference(path)
            return
        self._json({"error": "not found"}, 404)

    # -- 推理 -------------------------------------------------------------
    def handle_inference(self, path: str) -> None:
        body = self._read_body()
        stream = bool(body.get("stream"))
        max_tokens = int(body.get("max_tokens") or 8)
        max_tokens = max(1, min(max_tokens, 32))
        if path.endswith("chat/completions"):
            prompt = json.dumps(body.get("messages", []), ensure_ascii=False)
        else:
            prompt = body.get("prompt") or ""
        prompt_tokens = max(1, len(prompt) // 4)
        cached = max(0, prompt_tokens - 8) if os.environ.get("MOCK_CACHE_HIT") == "1" else 0
        with LOCK:
            STATE["requests"] += 1
            STATE["prompt_tokens"] += prompt_tokens
            STATE["generation_tokens"] += max_tokens
            STATE["cached_tokens"] += cached
            drafts = max_tokens // 8 + 1
            STATE["drafts"] += drafts
            STATE["draft_tokens"] += drafts * 7
            STATE["accepted_tokens"] += drafts * 2

        req_id = f"cmpl-mock-{int(time.time() * 1000) % 100000}"
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": max_tokens,
            "total_tokens": prompt_tokens + max_tokens,
            "prompt_tokens_details": {"cached_tokens": cached},
        }
        per_request_metrics = {
            "time_to_first_token_ms": 120.5,
            "generation_time_ms": 800.0,
            "queue_time_ms": 3.4,
            "mean_itl_ms": 55.2,
            "tokens_per_second": 40.0,
            "speculative_decoding": {
                "mean_acceptance_length": 1.29,
                "draft_acceptance_rate": 0.286,
                "acceptance_histogram": [drafts - 1, 1, 0, 0, 0, 0, 0, 0],
                "num_spec_steps": drafts,
                "num_accepted_draft_tokens": drafts * 2,
                "num_draft_tokens": drafts * 7,
                "num_spec_tokens": 7,
                "per_step_accepted": [2] * drafts,
                "per_step_drafted": [7] * drafts,
            },
        }

        if not stream:
            payload = {
                "id": req_id,
                "object": "chat.completion" if path.endswith("chat/completions") else "text_completion",
                "created": int(time.time()),
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "mock answer"},
                        "text": "mock answer",
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
                "metrics": per_request_metrics,
            }
            self._json(payload)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send_event(payload: dict) -> None:
            data = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        try:
            for i in range(max_tokens):
                if path.endswith("chat/completions"):
                    delta = {"role": "assistant"} if i == 0 else {}
                    delta["content"] = f"tok{i} "
                    choice = {"index": 0, "delta": delta, "finish_reason": None}
                else:
                    choice = {"index": 0, "text": f"tok{i} ", "finish_reason": None}
                send_event({"id": req_id, "object": "chat.completion.chunk", "model": MODEL,
                            "choices": [choice]})
                time.sleep(float(os.environ.get("MOCK_ITL_S", "0.005")))
            final_choice = (
                {"index": 0, "delta": {}, "finish_reason": "stop"}
                if path.endswith("chat/completions")
                else {"index": 0, "text": "", "finish_reason": "stop"}
            )
            send_event({"id": req_id, "object": "chat.completion.chunk", "model": MODEL,
                        "choices": [final_choice]})
            send_event({"id": req_id, "object": "chat.completion.chunk", "model": MODEL,
                        "choices": [], "usage": usage, "metrics": per_request_metrics})
            done = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(done):X}\r\n".encode() + done + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def write_fake_trace() -> Path:
    """生成一个结构真实的 Chrome trace（含 kernel / 注解 / ProfilerStep）。"""
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    events: list[dict] = []
    ts = 0
    for step in range(4):
        step_ts = ts
        events.append({"ph": "X", "cat": "user_annotation", "name": f"ProfilerStep#{step}",
                       "pid": 4242, "tid": 4242, "ts": step_ts, "dur": 1000})
        events.append({"ph": "X", "cat": "user_annotation",
                       "name": "execute_context_1(1024)_generation_4(32)",
                       "pid": 4242, "tid": 4242, "ts": step_ts, "dur": 1000})
        events.append({"ph": "X", "cat": "gpu_user_annotation",
                       "name": "execute_context_1(1024)_generation_4(32)",
                       "pid": 0, "tid": 20, "ts": step_ts, "dur": 980})
        # 模拟 GPU kernel：GEMM 占大头，通信其次
        for i in range(6):
            events.append({"ph": "X", "cat": "kernel", "name": "fp8_emu_gemm_block_kernel<32,64>",
                           "pid": 0, "tid": 20, "ts": step_ts + i * 100, "dur": 80})
        events.append({"ph": "X", "cat": "kernel", "name": "ncclDevKernel_AllReduce_Ring",
                       "pid": 0, "tid": 21, "ts": step_ts + 600, "dur": 200})
        events.append({"ph": "X", "cat": "kernel", "name": "aten::copy_", "pid": 0, "tid": 20,
                       "ts": step_ts + 800, "dur": 120})
        ts += 1000
    payload = {"schemaVersion": 1, "deviceProperties": [{"id": 0, "name": "Z100L"}],
               "traceEvents": events}
    plain = TRACE_DIR / "rank0.mock.pt.trace.json"
    plain.write_text(json.dumps(payload), encoding="utf-8")
    with gzip.open(str(plain) + ".gz", "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    (TRACE_DIR / "profiler_out_0.txt").write_text(
        "-------------------------------------------------------  ------------  ------------\n"
        "                                                   Name    Self CPU %      Self CPU\n"
        "-------------------------------------------------------  ------------  ------------\n"
        "                            fp8_emu_gemm_block_kernel         0.38%      30.040ms\n"
        "                                       vllm::all_reduce         1.50%     117.514ms\n",
        encoding="utf-8",
    )
    return plain


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock vLLM 服务已启动：http://{args.host}:{args.port}（model={MODEL}）", flush=True)
    print(f"trace 输出目录：{TRACE_DIR}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
