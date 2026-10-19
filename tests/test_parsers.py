"""单元测试：用真实日志/指标样例 + 合成 trace 验证各层解析器。

运行::

    python3 -m unittest discover -s tests -v      # 在项目根目录
    bash tests/run_tests.sh                       # 含 mock 端到端
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from probe.config import ProbeConfig  # noqa: E402
from probe.http_probe import RequestResult, build_prompt, request_once, summarize  # noqa: E402
from probe.log_probe import parse_line, read_events, summarize as log_summarize, summarize_text  # noqa: E402
from probe.metrics_probe import (  # noqa: E402
    MetricsSnapshot,
    diff_snapshots,
    format_stage_table,
    histogram_quantile,
    histogram_quantile_detail,
    parse_metrics_text,
    stage_table,
)
from probe.report import Artifacts, build_report, diagnose  # noqa: E402
from probe.trace_probe import analyze_trace, parse_key_averages, summarize_key_averages  # noqa: E402

SAMPLES = ROOT / "samples"


# ---------------------------------------------------------------------------
# 日志层
# ---------------------------------------------------------------------------

ITER_LINE = (
    "(APIServer pid=696031) INFO 10-15 23:56:58 [loggers.py:191] Engine 000: "
    "Iteration(752): 0 context requests, 0 context tokens, 1 generation requests, "
    "8 generation tokens, iteration elapsed time: 184.79 ms, GPU KV cache usage: 3.5%"
)
ITER_ENCODER_LINE = (
    "INFO 10-15 23:56:58 [loggers.py:191] Engine 000: Iteration(9): 2 context requests, "
    "1024 context tokens, 1 generation requests, 4 generation tokens, "
    "iteration elapsed time: 55.00 ms, encoder inputs: 3, encoder output embeddings: 12, "
    "GPU KV cache usage: 7.2%"
)
STATS_LINE = (
    "(APIServer pid=696031) INFO 10-15 23:57:01 [loggers.py:320] Engine 000: Avg prompt "
    "throughput: 0.0 tokens/s, Avg generation throughput: 8.0 tokens/s, Running: 0 reqs, "
    "Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 27.0%"
)
SPEC_LINE = (
    "(APIServer pid=696031) INFO 10-15 23:57:01 [metrics.py:120] SpecDecoding metrics: "
    "Mean acceptance length: 1.69, Accepted throughput: 3.34 tokens/s, Drafted throughput: "
    "33.73 tokens/s, Accepted: 18 tokens, Drafted: 182 tokens, Per-position acceptance rate: "
    "0.462, 0.115, 0.077, 0.038, 0.000, 0.000, 0.000, Avg Draft acceptance rate: 9.9%"
)
MFU_LINE = "(APIServer pid=696031) INFO 10-15 23:57:01 [perf.py:1726] Engine 000: MFU: 0.6 TF/s/device 32.4 GB/s/device"
CG_ROW = "(APIServer pid=696031) INFO 10-15 23:57:01 [cuda_graph.py:123] | 8               | 8             | 0            | NONE         | 26    |"


class LogParserTests(unittest.TestCase):
    def test_iteration_line(self) -> None:
        event = parse_line(ITER_LINE)
        assert event is not None
        self.assertEqual(event.kind, "iteration")
        self.assertEqual(event.data["iteration"], 752)
        self.assertEqual(event.data["gen_tokens"], 8)
        self.assertAlmostEqual(event.data["elapsed_ms"], 184.79)
        self.assertAlmostEqual(event.data["kv_usage_perc"], 3.5)
        self.assertIsNotNone(event.ts)

    def test_iteration_with_encoder(self) -> None:
        event = parse_line(ITER_ENCODER_LINE)
        assert event is not None
        self.assertEqual(event.data["ctx_tokens"], 1024)
        self.assertEqual(event.data["encoder_inputs"], 3)
        self.assertAlmostEqual(event.data["kv_usage_perc"], 7.2)

    def test_stats_line(self) -> None:
        event = parse_line(STATS_LINE)
        assert event is not None
        self.assertEqual(event.kind, "stats")
        self.assertEqual(event.data["running"], 0)
        self.assertAlmostEqual(event.data["generation_tps"], 8.0)
        self.assertAlmostEqual(event.data["prefix_cache_hit_rate"], 27.0)

    def test_spec_decode_line(self) -> None:
        event = parse_line(SPEC_LINE)
        assert event is not None
        self.assertEqual(event.kind, "spec_decode")
        self.assertAlmostEqual(event.data["mean_acceptance_length"], 1.69)
        self.assertEqual(len(event.data["per_position_acceptance"]), 7)
        self.assertAlmostEqual(event.data["per_position_acceptance"][0], 0.462)

    def test_mfu_and_cudagraph(self) -> None:
        mfu = parse_line(MFU_LINE)
        assert mfu is not None
        self.assertEqual(mfu.kind, "mfu")
        self.assertAlmostEqual(mfu.data["tflops_per_device"], 0.6)
        row = parse_line(CG_ROW)
        assert row is not None
        self.assertEqual(row.kind, "cudagraph_row")
        self.assertEqual(row.data["runtime_mode"], "NONE")
        self.assertEqual(row.data["count"], 26)

    def test_noise_line_returns_none(self) -> None:
        self.assertIsNone(parse_line(""))
        self.assertIsNone(parse_line("just some unrelated text"))

    def test_real_log_sample(self) -> None:
        """用真实日志样例跑一遍，确认四类事件都能被识别。"""
        events = read_events(SAMPLES / "serve.log")
        kinds = {e.kind for e in events}
        for expected in ("iteration", "stats", "spec_decode", "mfu", "cudagraph_row"):
            self.assertIn(expected, kinds)
        summary = log_summarize(events)
        self.assertGreater(summary["steps"]["total"], 100)
        self.assertIsNotNone(summary["efficiency"]["decode_ms_per_token"])
        self.assertGreater(summary["spec_decode"]["mean_acceptance_length"]["mean"], 1.0)
        text = summarize_text(summary)
        self.assertIn("step 总数", text)


# ---------------------------------------------------------------------------
# 指标层
# ---------------------------------------------------------------------------


class MetricsParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = (SAMPLES / "metrics.txt").read_text(encoding="utf-8")
        self.snap = parse_metrics_text(self.text, meta={"model": "qwen3.8-27b"})

    def test_families_and_histogram(self) -> None:
        self.assertGreater(len(self.snap.families), 90)
        fam = self.snap.family("vllm:time_to_first_token_seconds")
        assert fam is not None
        self.assertEqual(fam.type, "histogram")
        self.assertGreater(fam.count, 0)
        # 均值 = _sum / _count，量级应在合理范围内（这里是真实抓取的数据）
        self.assertGreater(fam.mean(), 0.0)

    def test_quantile_monotonic(self) -> None:
        fam = self.snap.family("vllm:time_to_first_token_seconds")
        values = [histogram_quantile(fam, q) for q in (0.1, 0.5, 0.9, 0.99)]
        self.assertTrue(all(v is not None for v in values))
        self.assertEqual(values, sorted(values))

    def test_coarse_bucket_detection(self) -> None:
        """队列时间直方图首桶是 0.3s，所有观测都落进去，必须标记为"分辨率不足"。"""
        fam = self.snap.family("vllm:request_queue_time_seconds")
        detail = histogram_quantile_detail(fam, 0.5)
        self.assertTrue(detail.coarse)
        assert detail.upper is not None
        self.assertLessEqual(detail.upper, 0.3 + 1e-9)
        rows = stage_table(self.snap)
        rendered = format_stage_table(rows)
        self.assertIn("≤", rendered)
        self.assertIn("分辨率不足", rendered)

    def test_engine_state_and_kv_blocks(self) -> None:
        from probe.metrics_probe import engine_state, kv_block_metrics

        state = engine_state(self.snap)
        self.assertIn("prefix_cache_hit_rate", state)
        self.assertGreater(state["prompt_tokens_total"], 0)
        blocks = kv_block_metrics(self.snap)
        self.assertIn("lifetime", blocks)

    def test_roundtrip_and_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            self.snap.save(path)
            loaded = MetricsSnapshot.load(path)
            self.assertEqual(set(loaded.families), set(self.snap.families))
            self.assertAlmostEqual(
                loaded.total("vllm:prompt_tokens_total"),
                self.snap.total("vllm:prompt_tokens_total"),
            )
        later = parse_metrics_text(
            self.text.replace("vllm:prompt_tokens_total{", "vllm:prompt_tokens_total{", 1),
            meta=self.snap.meta,
        )
        # 人为把计数器改大后再做差
        for fam in later.families.values():
            for series in fam.series:
                if series.name == "vllm:prompt_tokens_total":
                    series.value += 1000
        diff = diff_snapshots(self.snap, later, 10.0)
        self.assertAlmostEqual(diff["counters"]["vllm:prompt_tokens_total"], 1000.0)
        self.assertAlmostEqual(diff["rates"]["vllm:prompt_tokens_total"], 100.0)

    def test_diff_histogram_units(self) -> None:
        """只有 *_seconds 直方图换算 ms；token 类必须按原值（不能用 709000.0 ms）。"""
        before = parse_metrics_text(
            "\n".join(
                [
                    "# TYPE vllm:request_prompt_tokens histogram",
                    'vllm:request_prompt_tokens_bucket{le="+Inf"} 1',
                    "vllm:request_prompt_tokens_sum 100",
                    "vllm:request_prompt_tokens_count 1",
                    "# TYPE vllm:e2e_request_latency_seconds histogram",
                    'vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 1',
                    "vllm:e2e_request_latency_seconds_sum 0.5",
                    "vllm:e2e_request_latency_seconds_count 1",
                ]
            )
        )
        after = parse_metrics_text(
            "\n".join(
                [
                    "# TYPE vllm:request_prompt_tokens histogram",
                    'vllm:request_prompt_tokens_bucket{le="+Inf"} 3',
                    "vllm:request_prompt_tokens_sum 300",
                    "vllm:request_prompt_tokens_count 3",
                    "# TYPE vllm:e2e_request_latency_seconds histogram",
                    'vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 3',
                    "vllm:e2e_request_latency_seconds_sum 1.5",
                    "vllm:e2e_request_latency_seconds_count 3",
                ]
            )
        )
        diff = diff_snapshots(before, after, 1.0)
        tokens = diff["histograms"]["vllm:request_prompt_tokens"]
        self.assertEqual(tokens["unit"], "tokens")
        self.assertAlmostEqual(tokens["mean"], 100.0)
        self.assertIsNone(tokens["mean_ms"])
        seconds = diff["histograms"]["vllm:e2e_request_latency_seconds"]
        self.assertEqual(seconds["unit"], "s")
        self.assertAlmostEqual(seconds["mean_ms"], 500.0)

    def test_metrics_diff_print_handles_token_histograms(self) -> None:
        """回归：token 直方图 mean_ms=None，`probe metrics --diff --print` 不能按 ms 格式化。

        scripts/collect_metrics.sh 的 --diff 路径固定带 --print，曾因 TypeError 中断。
        """
        before = parse_metrics_text(
            "\n".join(
                [
                    "# TYPE vllm:request_prompt_tokens histogram",
                    'vllm:request_prompt_tokens_bucket{le="+Inf"} 1',
                    "vllm:request_prompt_tokens_sum 100",
                    "vllm:request_prompt_tokens_count 1",
                ]
            )
        )
        after = parse_metrics_text(
            "\n".join(
                [
                    "# TYPE vllm:request_prompt_tokens histogram",
                    'vllm:request_prompt_tokens_bucket{le="+Inf"} 3',
                    "vllm:request_prompt_tokens_sum 300",
                    "vllm:request_prompt_tokens_count 3",
                ]
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            before_path = Path(tmp) / "before.json"
            after_path = Path(tmp) / "after.json"
            before.save(before_path)
            after.save(after_path)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT) + (
                os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "probe",
                    "metrics",
                    "--diff",
                    str(before_path),
                    str(after_path),
                    "--elapsed",
                    "1",
                    "--print",
                    "--outdir",
                    tmp,
                ],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("tokens", proc.stdout)
            self.assertNotIn("Traceback", proc.stderr)


# ---------------------------------------------------------------------------
# trace 层
# ---------------------------------------------------------------------------


class TraceParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import mock_vllm_server

        cls.tmp = tempfile.TemporaryDirectory()
        import os

        os.environ["MOCK_TRACE_DIR"] = cls.tmp.name
        mock_vllm_server.TRACE_DIR = Path(cls.tmp.name)
        cls.trace = mock_vllm_server.write_fake_trace()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_leaf_and_range_split(self) -> None:
        summary = analyze_trace(self.trace)
        self.assertGreater(summary.gpu_leaf_us, 0)
        self.assertGreater(summary.gpu_range_us, 0)
        # 注解区间覆盖整步，必须大于纯 kernel 时间
        self.assertGreater(summary.gpu_range_us, summary.gpu_leaf_us * 0.5)
        self.assertEqual(len(summary.steps), 4)

    def test_categories_and_share(self) -> None:
        summary = analyze_trace(self.trace)
        shares = sum(row["gpu_share"] for row in summary.categories.values())
        self.assertLess(abs(shares - 1.0), 0.02)
        gemm = next((v for k, v in summary.categories.items() if "GEMM" in k), None)
        self.assertIsNotNone(gemm)
        self.assertGreater(gemm["gpu_share"], 0.2)

    def test_step_annotations(self) -> None:
        summary = analyze_trace(self.trace)
        self.assertTrue(summary.step_annotations)
        mixed = next((v for k, v in summary.step_annotations.items() if "混合" in k), None)
        self.assertIsNotNone(mixed)
        self.assertGreater(mixed["gpu_time_us"], 0)

    def test_gz_support(self) -> None:
        gz = Path(str(self.trace) + ".gz")
        self.assertTrue(gz.exists())
        summary = analyze_trace(gz)
        self.assertEqual(len(summary.steps), 4)

    def test_key_averages(self) -> None:
        rows = parse_key_averages(Path(self.tmp.name) / "profiler_out_0.txt")
        self.assertTrue(rows)
        summary = summarize_key_averages(rows)
        self.assertGreater(summary["rows"], 0)


# ---------------------------------------------------------------------------
# 请求层（对 mock 服务端到端）
# ---------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class HttpProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import mock_vllm_server

        cls.port = free_port()
        cls.server = mock_vllm_server.ThreadingHTTPServer(
            ("127.0.0.1", cls.port), mock_vllm_server.Handler
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.2)
        cls.cfg = ProbeConfig(
            base_url=f"http://127.0.0.1:{cls.port}",
            model="qwen3.8-27b",
            max_tokens=8,
            prompt_tokens=64,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def test_streaming_request(self) -> None:
        result = request_once(self.cfg, index=0, prompt="ping " * 20, max_tokens=8)
        self.assertTrue(result.ok, msg=result.error)
        self.assertIsNotNone(result.ttft_ms)
        self.assertEqual(result.completion_tokens, 8)
        self.assertGreaterEqual(result.chunks, 8)
        self.assertIsNotNone(result.server_metrics)
        self.assertIsNotNone(result.spec_decode)
        self.assertEqual(result.server_metrics["queue_time_ms"], 3.4)

    def test_non_streaming_request(self) -> None:
        cfg = ProbeConfig(**{**self.cfg.__dict__, "stream": False})
        result = request_once(cfg, index=0, prompt="ping", max_tokens=4)
        self.assertTrue(result.ok, msg=result.error)
        self.assertEqual(result.completion_tokens, 4)
        self.assertIsNotNone(result.server_metrics)

    def test_summary_and_throughput(self) -> None:
        results = [request_once(self.cfg, index=i, prompt=f"ping {i}", max_tokens=8) for i in range(3)]
        summary = summarize(results, self.cfg)
        self.assertEqual(summary["ok"], 3)
        self.assertGreater(summary["throughput"]["output_tokens_per_s"], 0)
        self.assertTrue(summary["server_side"]["available"])
        self.assertTrue(summary["spec_decode"]["available"])

    def test_prompt_builder_tokens(self) -> None:
        text = build_prompt(256, shared_prefix_tokens=128, index=3)
        self.assertIn("SHARED-PREFIX", text)
        self.assertIn("UNIQUE-SUFFIX-3", text)

    def test_failure_is_captured(self) -> None:
        cfg = ProbeConfig(base_url="http://127.0.0.1:1", model="x", timeout=2)
        result = request_once(cfg, index=0, prompt="x", max_tokens=1)
        self.assertFalse(result.ok)
        self.assertIsNotNone(result.error)


# ---------------------------------------------------------------------------
# 报告层
# ---------------------------------------------------------------------------


class ReportTests(unittest.TestCase):
    def test_report_from_synthetic_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "run_meta.json").write_text(
                json.dumps(
                    {
                        "started_at": "2026-10-16T10:00:00",
                        "image": "z100-s9-snap-s14",
                        "container": "vllm-probe",
                        "model": "/mnt/game/models/Qwen3.8-27B-FP8",
                        "served_name": "qwen3.8-27b",
                        "port": 8100,
                        "tensor_parallel_size": 4,
                        "max_model_len": 32768,
                        "max_num_batched_tokens": 8192,
                        "enforce_eager": True,
                        "probe_level": "deep",
                        "command": "python3 -c '...' serve /mnt/game/models/...",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            # 请求级
            (run / "http_summary.json").write_text(
                json.dumps(
                    {
                        "config": {"prompt_tokens": 512, "max_tokens": 128, "concurrency": 1,
                                   "repeat": 3, "stream": True, "endpoint": "chat"},
                        "requests": 3,
                        "ok": 3,
                        "failed": 0,
                        "ttft_ms": {"p50": 320.0, "p90": 400.0, "p99": 420.0},
                        "e2e_ms": {"p50": 1500.0, "p90": 1600.0, "p99": 1700.0},
                        "decode_ms": {"p50": 1100.0, "p90": 1200.0, "p99": 1300.0},
                        "headers_ms": {"p50": 5.0, "p90": 6.0, "p99": 7.0},
                        "tpot_ms": {"p50": 60.0, "p90": 62.0, "p99": 63.0},
                        "itl_ms": {"count": 100, "mean_ms": 62.0, "p50_ms": 60.0,
                                   "p90_ms": 70.0, "p99_ms": 90.0, "max_ms": 120.0},
                        "completion_tokens_avg": 20.0,
                        "cached_tokens": [0, 400, 410],
                        "server_side": {"available": True, "queue_time_ms": {"p50": 5.0, "p90": 60.0, "mean": 20.0},
                                        "time_to_first_token_ms": {"p50": 300.0, "p90": 380.0, "mean": 320.0},
                                        "generation_time_ms": {"p50": 1000.0, "p90": 1100.0, "mean": 1050.0}},
                        "spec_decode": {"available": True, "mean_acceptance_length": 1.6,
                                        "draft_acceptance_rate": 0.15,
                                        "acceptance_histogram": [5, 3, 1, 0, 0, 0, 0, 0],
                                        "num_spec_steps": 9, "per_step_accepted": [2, 1, 2]},
                        "throughput": {"wall_s": 5.0, "output_tokens_per_s": 12.0,
                                       "total_tokens_per_s": 320.0, "requests_per_s": 0.6},
                        "errors": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (run / "http_requests.jsonl").write_text(
                json.dumps({"index": 0, "ok": True, "prompt_tokens": 512, "cached_tokens": 0,
                            "ttft_ms": 330.0, "e2e_ms": 1500.0, "tpot_ms": 60.0,
                            "completion_tokens": 20, "finish_reason": "stop"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            # step 级
            steps = log_summarize(read_events(SAMPLES / "serve.log"))
            (run / "steps_summary.json").write_text(
                json.dumps(steps, ensure_ascii=False), encoding="utf-8"
            )
            # 指标
            snap = parse_metrics_text((SAMPLES / "metrics.txt").read_text(encoding="utf-8"))
            snap.save(run / "metrics_after.json")
            # trace
            import mock_vllm_server

            with tempfile.TemporaryDirectory() as tdir:
                mock_vllm_server.TRACE_DIR = Path(tdir)
                trace = mock_vllm_server.write_fake_trace()
                from probe.trace_probe import to_dict

                summary = analyze_trace(trace)
                payload = to_dict(summary, top=10)
                payload["path"] = str(trace)
                (run / "trace_summary.json").write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
                ka = summarize_key_averages(parse_key_averages(Path(tdir) / "profiler_out_0.txt"))
                (run / "key_averages.json").write_text(
                    json.dumps(ka, ensure_ascii=False), encoding="utf-8"
                )

            text = build_report(run)
            for heading in ("# vLLM 推理全流程探测报告", "## 2. 请求级生命周期", "## 3. 引擎 step 级",
                            "## 4. 调度 / 队列 / KV 缓存", "## 5. 投机解码", "## 7. 算子 / kernel 级",
                            "## 8. 自动归因与优化建议", "## 9. 产物清单与复现"):
                self.assertIn(heading, text, msg=f"缺少章节 {heading}")
            # 关键数字应出现在报告里
            self.assertIn("320.0", text)
            self.assertIn("qwen3.8-27b", text)
            findings = diagnose(Artifacts.scan(run))
            self.assertTrue(findings)

    def test_report_on_empty_dir_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = build_report(tmp)
            self.assertIn("运行概览", text)


class DocsConsistencyTests(unittest.TestCase):
    """文档/代码里写到的 ``vllm:*`` 指标必须真的存在于本版本的 `/metrics`。

    ``samples/metrics.txt`` 是现网实例的真实抓取（104 个指标族 / 577 条序列），
    所以这个测试能挡住"文档引用了一个本版本根本不存在的指标名"这类错误——
    例如早期遗留的 ``gpu_cache_usage_perc``、``cpu_cache_usage_perc``、
    ``num_requests_swapped``（本版本都不存在，引用它们只会静默拿到空序列）。

    带通配/缩写的写法（如 ``vllm:estimated_{...}``、``vllm:mm_cache_*``）按前缀匹配。
    """

    # 允许在文档里作为"历史名字/不存在"出现的名字，需写明原因
    ALLOWED_ABSENT = {
        "vllm:cpu_cache_usage_perc": "本版本未导出（文档中已标注删除线）",
        "vllm:gpu_cache_usage_perc": "被 kv_cache_usage_perc 取代（文档中已标注删除线）",
        "vllm:num_requests_swapped": "本版本已移除（文档中已标注删除线）",
        "vllm:model_forward_time_milliseconds": "文档中作为『--collect-detailed-traces 在本版本为空操作』的反例引用",
    }

    # 需要扫描的文件：文档 + Prometheus 配置 + Grafana 看板 + compose
    SCAN_FILES = (
        "README.md",
        "docs/观测字段对照表.md",
        "configs/prometheus.yml",
        "configs/otel-collector.yaml",
        "configs/grafana/dashboards/vllm-inference-steps.json",
        "docker/docker-compose.yml",
        "docker/docker-compose.obs.yml",
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls.snap = parse_metrics_text((SAMPLES / "metrics.txt").read_text(encoding="utf-8"))
        cls.families = set(cls.snap.families)
        cls.series = {s.name for f in cls.snap.families.values() for s in f.series}

    @classmethod
    def _exists(cls, name: str) -> bool:
        """指标名是否存在：兼容 _bucket/_sum/_count 子序列与 Prometheus 抓取目标 vllm:8000。"""
        if re.fullmatch(r"vllm:\d+", name):
            return True
        if name in cls.families or name in cls.series:
            return True
        # 文档里可能用缩写/通配（如 vllm:estimated_{...}、vllm:mm_cache_*），按前缀放行：
        # 只要存在某个真实指标族以它开头就算通过（此时真实名字由其它地方写全）。
        return any(family.startswith(name) for family in cls.families)

    def test_documented_and_dashboard_metrics_exist_in_real_dump(self) -> None:
        """文档/看板/PromQL 里出现的每个 ``vllm:*`` 名字都必须在真实抓取里存在。

        ``samples/metrics.txt`` 是现网实例的真实抓取（104 个指标族 / 577 条序列）。
        这个测试能挡住两类静默故障：文档引用旧版指标名（如 gpu_cache_usage_perc），
        以及 Grafana 看板 PromQL 引用了本版本不存在的指标（面板永远空白）。
        """
        text = ""
        for rel in self.SCAN_FILES:
            text += (ROOT / rel).read_text(encoding="utf-8")
        mentioned = sorted(set(re.findall(r"vllm:[a-z0-9_]+", text)))
        self.assertGreater(len(mentioned), 25, "扫描到的指标名太少，检查 SCAN_FILES")

        missing = [
            name
            for name in mentioned
            if not self._exists(name) and name not in self.ALLOWED_ABSENT
        ]
        self.assertEqual(missing, [], f"引用了本版本不存在的指标：{missing}")

    def test_engine_state_missing_family_is_none_not_zero(self) -> None:
        from probe.metrics_probe import engine_state

        state = engine_state(self.snap)
        # 本版本不存在该族，必须是 None（而不是看起来像"使用率 0%"）
        self.assertIsNone(state["cpu_cache_usage_perc"])
        self.assertIsNotNone(state["kv_cache_usage_perc"])

    def test_dashboard_promql_quantile_inputs_are_histograms(self) -> None:
        """看板里所有 ``histogram_quantile(...)`` 用到的 ``*_bucket`` 必须是真实 histogram。

        （PromQL 里常见 ``histogram_quantile(0.9, sum by (le) (rate(..._bucket[1m])))``，
        括号嵌套，所以只在「含 histogram_quantile 的表达式」里抽取 ``vllm:*_bucket`` 名字。）
        """
        dashboard = json.loads(
            (ROOT / "configs/grafana/dashboards/vllm-inference-steps.json").read_text(encoding="utf-8")
        )
        exprs: list[str] = []
        for panel in dashboard.get("panels", []):
            for target in panel.get("targets", []) or []:
                if target.get("expr"):
                    exprs.append(target["expr"])
        self.assertTrue(exprs, "看板里应该至少有若干 PromQL 查询")

        checked = 0
        for expr in exprs:
            if "histogram_quantile" not in expr:
                continue
            # 同时覆盖 vLLM 指标与 API server 的 FastAPI instrumentator 指标
            buckets = re.findall(r"((?:vllm:|http_)[a-z0-9_]*_bucket)", expr)
            self.assertTrue(buckets, f"histogram_quantile 表达式里找不到 *_bucket：{expr[:120]}")
            for bucket in buckets:
                base = bucket[: -len("_bucket")]
                family = self.snap.family(base)
                self.assertIsNotNone(family, f"看板引用了不存在的直方图：{bucket}")
                self.assertEqual(
                    family.type, "histogram", f"{base} 类型是 {family.type}，不能走 histogram_quantile"
                )
                checked += 1
        self.assertGreater(checked, 0, "应该至少校验到一个 histogram_quantile 查询")


class ConfigSafetyTests(unittest.TestCase):
    """产物里不能出现明文 API key。"""

    def test_api_key_masked_in_dump(self) -> None:
        raw = "sk-secret-1234567890"
        cfg = ProbeConfig(api_key=raw)
        data = cfg.dump()
        self.assertNotEqual(data["api_key"], raw)
        self.assertNotIn(raw, json.dumps(data))
        # 打码后仍可辨识（前 4 + 后 4），且配置对象本身仍保留原值供请求使用
        self.assertTrue(data["api_key"].startswith("sk-s"))
        self.assertTrue(data["api_key"].endswith("7890"))
        self.assertEqual(cfg.api_key, raw)

    def test_empty_api_key_kept(self) -> None:
        self.assertEqual(ProbeConfig(api_key="EMPTY").dump()["api_key"], "EMPTY")

    def test_auth_headers(self) -> None:
        self.assertEqual(ProbeConfig(api_key="EMPTY").auth_headers(), {})
        self.assertEqual(
            ProbeConfig(api_key="sk-abc").auth_headers(),
            {"Authorization": "Bearer sk-abc"},
        )


class ReportExternalLogTests(unittest.TestCase):
    """外部启动实例：RUN_DIR 没有 serve.log 时回退读 run_meta["serve_log"]。"""

    def test_meta_serve_log_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            external_log = run / "external-serve.log"
            external_log.write_text("hello\n", encoding="utf-8")
            (run / "run_meta.json").write_text(
                json.dumps(
                    {"runtime": "host", "model": "m", "serve_log": str(external_log)},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            art = Artifacts.scan(run)
            self.assertEqual(art.serve_log, external_log)

    def test_report_histogram_units_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "metrics_after.json").write_text(
                json.dumps({"families": {}}), encoding="utf-8"
            )
            (run / "metrics_diff.json").write_text(
                json.dumps(
                    {
                        "elapsed_s": 18.0,
                        "counters": {},
                        "rates": {},
                        "histograms": {
                            "vllm:request_prompt_tokens": {
                                "count": 4,
                                "unit": "tokens",
                                "mean": 709.0,
                                "mean_ms": None,
                            },
                            "vllm:e2e_request_latency_seconds": {
                                "count": 4,
                                "unit": "s",
                                "mean": 4.5038,
                                "mean_ms": 4503.8,
                            },
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            text = build_report(run)
            self.assertIn("709.0 tokens", text)
            self.assertIn("4503.8 ms", text)
            self.assertIn("均值", text)
            self.assertNotIn("709000.0", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
