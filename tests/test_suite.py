"""标准化对比套件测试：矩阵选择、schema、mock 端到端、compare 渲染。

运行::

    python3 -m unittest tests.test_suite -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from probe.config import ProbeConfig  # noqa: E402
from probe.suite import (  # noqa: E402
    COMPARE_METRICS,
    compare_payloads,
    default_scenarios,
    load_suite,
    run_suite,
    select_scenarios,
)

import mock_vllm_server  # noqa: E402


class MatrixTests(unittest.TestCase):
    def test_default_matrix_is_eight_fixed_scenarios(self) -> None:
        matrix = default_scenarios()
        self.assertEqual(len(matrix), 8)
        self.assertEqual(
            [s.id for s in matrix],
            ["s01_warmup", "s02_chat_short_c1", "s03_decode_heavy_c1",
             "s04_prefill_heavy", "s05_concurrent_c8", "s06_prefix_cache",
             "s07_long_context", "s08_spec_decode"],
        )
        warmups = [s for s in matrix if s.warmup]
        self.assertEqual(len(warmups), 1)
        optional = [s for s in matrix if s.optional]
        self.assertEqual([s.id for s in optional], ["s08_spec_decode"])

    def test_long_context_override(self) -> None:
        matrix = default_scenarios(long_context_tokens=8192)
        long_ctx = next(s for s in matrix if s.id == "s07_long_context")
        self.assertEqual(long_ctx.prompt_tokens, 8192)

    def test_select_scenarios(self) -> None:
        matrix = default_scenarios()
        picked = select_scenarios("s02,s05", matrix)
        self.assertEqual([s.id for s in picked], ["s02_chat_short_c1", "s05_concurrent_c8"])
        self.assertEqual(len(select_scenarios(None, matrix)), 8)
        with self.assertRaises(KeyError):
            select_scenarios("nope", matrix)


class SuiteE2ETests(unittest.TestCase):
    """对进程内 mock vLLM 起套件（无 GPU/Docker）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), mock_vllm_server.Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.tmp = tempfile.TemporaryDirectory(prefix="probe-suite-")
        cls.outdir = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def _cfg(self) -> ProbeConfig:
        return ProbeConfig(
            base_url=f"http://127.0.0.1:{self.port}",
            model=mock_vllm_server.MODEL,
            outdir=str(self.outdir),
            timeout=60.0,
        )

    def test_run_suite_writes_stable_schema(self) -> None:
        matrix = select_scenarios("s02,s06", default_scenarios())
        payload = run_suite(self._cfg(), matrix, verbose=False)

        self.assertEqual(payload["kind"], "vllm-probe-suite")
        for key in ("created_at", "vendor", "hardware", "engine", "matrix", "scenarios", "ok"):
            self.assertIn(key, payload)
        self.assertEqual(set(payload["scenarios"]), {"s02_chat_short_c1", "s06_prefix_cache"})

        scenario = payload["scenarios"]["s02_chat_short_c1"]
        self.assertTrue(scenario["ok"], scenario.get("error"))
        self.assertEqual(scenario["requests"], scenario["ok_requests"])
        for block in ("ttft_ms", "e2e_ms", "tpot_ms", "itl_ms", "throughput", "engine"):
            self.assertIn(block, scenario)
        self.assertIsNotNone(scenario["ttft_ms"]["p50"])
        # 引擎侧差分来自 mock 的 Prometheus 计数器
        self.assertIsNotNone(scenario["engine"]["generation_tokens_per_s"])
        self.assertGreater(scenario["engine"]["generation_tokens_per_s"], 0)

        # 落盘 + 重新加载
        path = self.outdir / "suite_results.json"
        self.assertTrue(path.exists())
        reloaded = load_suite(path)
        self.assertEqual(reloaded["scenarios"].keys(), payload["scenarios"].keys())

        # 每个场景的原始逐请求数据
        self.assertTrue((self.outdir / "suite" / "s02_chat_short_c1" / "http_requests.jsonl").exists())

    def test_run_suite_marks_failed_scenario_without_abort(self) -> None:
        """把端口指向不存在的服务：场景应记录错误而不是抛异常。"""
        cfg = ProbeConfig(base_url="http://127.0.0.1:1", model="m",
                          outdir=str(self.outdir), timeout=0.5)
        matrix = select_scenarios("s02", default_scenarios())
        payload = run_suite(cfg, matrix)
        scenario = payload["scenarios"]["s02_chat_short_c1"]
        self.assertFalse(scenario["ok"])
        self.assertFalse(payload["ok"])

    def test_compare_payloads_renders_markdown(self) -> None:
        matrix = select_scenarios("s02", default_scenarios())
        a = run_suite(self._cfg(), matrix)
        b = json.loads(json.dumps(a))
        b["vendor"] = {"id": "nvidia", "label": "NVIDIA CUDA"}
        b["scenarios"]["s02_chat_short_c1"]["ttft_ms"]["p50"] *= 0.5
        text = compare_payloads(a, b)
        self.assertIn("标准化套件对比", text)
        self.assertIn("NVIDIA CUDA", text)
        self.assertIn("s02_chat_short_c1", text)
        for path, name, _unit, _higher in COMPARE_METRICS:
            self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
