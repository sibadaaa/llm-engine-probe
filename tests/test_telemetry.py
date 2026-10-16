"""GPU 遥测抽象层测试：五家 SMI 输出解析 + 归一化 schema + 汇总。

用 samples/smi/ 下的真实格式样例（hygon 的格式来自现网正则，其余为官方输出样式）。
运行::

    python3 -m unittest tests.test_telemetry -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from probe.telemetry import (  # noqa: E402
    BACKEND_BY_SMI,
    HyBackend,
    NpuBackend,
    NvidiaBackend,
    RocmBackend,
    XpuBackend,
    backend_by_name,
    candidate_backends,
    format_record,
    load_records,
    sample_loop,
    sample_once,
    summarize_cards,
    summarize_records,
    vram_perc_sum,
)
from probe.vendors import get_vendor  # noqa: E402

SMI_SAMPLES = ROOT / "samples" / "smi"

HY_SAMPLE = """
======================= Hygon System Management Interface ========================
DCU  Temp   AvgPwr  Perf  PwrCap  VRAM%  DCU%  Mode
  0   45.0C   120.0W   sp     300.0W   55%   N/A   Normal
  1   47.0C   130.0W   sp     300.0W   56%  87%   Normal
"""


class BackendRegistryTests(unittest.TestCase):
    def test_backend_by_smi_covers_all_vendors(self) -> None:
        for smi in ("hy-smi", "nvidia-smi", "rocm-smi", "amd-smi", "npu-smi", "xpu-smi"):
            self.assertIn(smi, BACKEND_BY_SMI)

    def test_backend_by_name_aliases(self) -> None:
        self.assertIsInstance(backend_by_name("hygon"), HyBackend)
        self.assertIsInstance(backend_by_name("nvidia"), NvidiaBackend)
        self.assertIsInstance(backend_by_name("rocm"), RocmBackend)
        with self.assertRaises(KeyError):
            backend_by_name("nonexistent")

    def test_candidate_backends_follow_vendor(self) -> None:
        backends = candidate_backends(get_vendor("hygon"))
        self.assertIsInstance(backends[0], HyBackend)
        # hygon 的 fallback 是 rocm-smi
        self.assertTrue(any(isinstance(b, RocmBackend) for b in backends))

    def test_candidate_backends_generic_sniff(self) -> None:
        def fake_which(cmd: str) -> str | None:
            return "/usr/bin/nvidia-smi" if cmd == "nvidia-smi" else None

        with mock.patch("probe.telemetry.shutil.which", side_effect=fake_which):
            backends = candidate_backends(get_vendor("generic"))
        self.assertTrue(any(isinstance(b, NvidiaBackend) for b in backends))


class HygonParserTests(unittest.TestCase):
    def test_parse(self) -> None:
        cards = HyBackend().parse(HY_SAMPLE)
        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0]["index"], 0)
        self.assertAlmostEqual(cards[0]["temp_c"], 45.0)
        self.assertAlmostEqual(cards[0]["power_w"], 120.0)
        self.assertAlmostEqual(cards[0]["vram_perc"], 55.0)
        self.assertIsNone(cards[0]["util_perc"])  # N/A
        self.assertAlmostEqual(cards[1]["util_perc"], 87.0)

    def test_live_sample_on_hygon_host(self) -> None:
        """本机若装了 hy-smi 且能解析出卡，就校验归一化字段（否则跳过）。"""
        record = sample_once(get_vendor("hygon"))
        if not record.get("ok"):
            self.skipTest("本机没有可用的 hy-smi")
        self.assertEqual(record["vendor"], "hygon")
        self.assertTrue(record["cards"])
        card = record["cards"][0]
        for key in ("index", "vram_perc", "power_w", "temp_c"):
            self.assertIn(key, card)


class NvidiaParserTests(unittest.TestCase):
    def test_parse_csv(self) -> None:
        text = (SMI_SAMPLES / "nvidia-smi.txt").read_text(encoding="utf-8")
        cards = NvidiaBackend().parse(text)
        self.assertEqual(len(cards), 3)
        first = cards[0]
        self.assertEqual(first["name"], "NVIDIA H100 PCIe")
        self.assertAlmostEqual(first["temp_c"], 45.0)
        self.assertAlmostEqual(first["power_w"], 120.53)
        self.assertAlmostEqual(first["vram_used_mb"], 12345.0)
        self.assertAlmostEqual(first["vram_total_mb"], 81559.0)
        self.assertAlmostEqual(first["util_perc"], 96.0)
        # N/A 值 → None（第三张卡）
        third = cards[2]
        self.assertIsNone(third["power_w"])
        self.assertIsNone(third["util_perc"])


class RocmParserTests(unittest.TestCase):
    def test_parse_json(self) -> None:
        text = (SMI_SAMPLES / "rocm-smi.json.txt").read_text(encoding="utf-8")
        cards = RocmBackend().parse(text)
        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0]["name"], "Radeon Instinct MI50")
        self.assertAlmostEqual(cards[0]["vram_total_mb"], 32768.0)
        self.assertAlmostEqual(cards[0]["vram_used_mb"], 16384.0)
        self.assertAlmostEqual(cards[0]["vram_perc"], 50.0)
        self.assertAlmostEqual(cards[0]["util_perc"], 87.0)
        self.assertAlmostEqual(cards[0]["power_w"], 180.0)

    def test_parse_human(self) -> None:
        text = (SMI_SAMPLES / "rocm-smi-human.txt").read_text(encoding="utf-8")
        cards = RocmBackend().parse(text)
        self.assertEqual(len(cards), 2)
        self.assertAlmostEqual(cards[0]["temp_c"], 54.0)
        self.assertAlmostEqual(cards[1]["power_w"], 95.0)
        self.assertAlmostEqual(cards[1]["vram_perc"], 25.0)


class NpuParserTests(unittest.TestCase):
    def test_parse(self) -> None:
        text = (SMI_SAMPLES / "npu-smi.txt").read_text(encoding="utf-8")
        cards = NpuBackend().parse(text)
        self.assertEqual(len(cards), 3)
        first = cards[0]
        self.assertEqual(first["name"], "910B2")
        self.assertAlmostEqual(first["power_w"], 92.3)
        self.assertAlmostEqual(first["temp_c"], 42.0)
        self.assertAlmostEqual(first["vram_total_mb"], 65536.0)
        second = cards[1]
        self.assertAlmostEqual(second["util_perc"], 35.0)
        self.assertAlmostEqual(second["vram_used_mb"], 8192.0)
        self.assertAlmostEqual(second["vram_perc"], 12.5)


class XpuParserTests(unittest.TestCase):
    def test_parse(self) -> None:
        text = (SMI_SAMPLES / "xpu-smi.txt").read_text(encoding="utf-8")
        cards = XpuBackend().parse(text)
        self.assertEqual(len(cards), 2)
        first = cards[0]
        self.assertEqual(first["index"], 0)
        self.assertAlmostEqual(first["util_perc"], 78.0)
        self.assertAlmostEqual(first["power_w"], 89.5)
        self.assertAlmostEqual(first["temp_c"], 50.0)
        self.assertAlmostEqual(first["vram_used_mb"], 12345.0)


class RecordTests(unittest.TestCase):
    def test_summarize_cards(self) -> None:
        cards = [
            {"index": 0, "vram_perc": 50.0, "power_w": 100.0, "temp_c": 40.0, "util_perc": 90.0},
            {"index": 1, "vram_perc": 60.0, "power_w": 110.0, "temp_c": 45.0, "util_perc": 80.0},
        ]
        summary = summarize_cards(cards)
        self.assertAlmostEqual(summary["vram_perc_sum"], 110.0)
        self.assertAlmostEqual(summary["power_w_sum"], 210.0)
        self.assertAlmostEqual(summary["temp_c_max"], 45.0)
        self.assertAlmostEqual(summary["util_perc_mean"], 85.0)

    def test_sample_once_generic_no_smi(self) -> None:
        with mock.patch("probe.telemetry.shutil.which", return_value=None):
            record = sample_once(get_vendor("generic"))
        self.assertFalse(record["ok"])
        self.assertIn("无可用 SMI", record["error"] or "")
        self.assertIn("vendor", record)

    def test_vram_perc_sum_no_smi(self) -> None:
        with mock.patch("probe.telemetry.shutil.which", return_value=None):
            self.assertIsNone(vram_perc_sum(get_vendor("generic")))

    def test_sample_loop_writes_jsonl(self) -> None:
        fake_record = {"ts": 1.0, "ok": True, "cards": [{"index": 0, "vram_perc": 10.0}],
                       "vendor": "generic", "smi": "fake-smi", "vram_perc_sum": 10.0}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "gpu_telem.jsonl"
            with mock.patch("probe.telemetry.sample_once", return_value=fake_record):
                count = sample_loop(0.0, 0.05, out, raw_file=None, profile=get_vendor("generic"))
            self.assertGreaterEqual(count, 1)
            records = load_records(out)
            self.assertGreaterEqual(len(records), 1)
            self.assertEqual(records[0]["ok"], True)

    def test_summarize_records(self) -> None:
        records = [
            {"ok": True, "vendor": "hygon", "smi": "hy-smi", "vram_perc_sum": 50.0,
             "power_w_sum": 100.0, "temp_c_max": 40.0, "util_perc_mean": None,
             "cards": [{"name": "Z100L"}]},
            {"ok": True, "vendor": "hygon", "smi": "hy-smi", "vram_perc_sum": 60.0,
             "power_w_sum": 120.0, "temp_c_max": 42.0, "util_perc_mean": None,
             "cards": [{"name": "Z100L"}]},
            {"ok": False, "vendor": "hygon", "error": "x"},
        ]
        summary = summarize_records(records)
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["ok_samples"], 2)
        self.assertAlmostEqual(summary["vram_perc_sum"]["mean"], 55.0)
        self.assertAlmostEqual(summary["power_w_sum"]["max"], 120.0)
        self.assertEqual(summary["cards"], ["Z100L"])

    def test_format_record(self) -> None:
        record = {"ok": True, "smi": "hy-smi", "vram_perc_sum": 50.0, "power_w_sum": 100.0,
                  "temp_c_max": 40.0, "util_perc_mean": None,
                  "cards": [{"index": 0, "name": "Z100L", "vram_perc": 50.0, "power_w": 100.0}]}
        text = format_record(record)
        self.assertIn("hy-smi", text)
        self.assertIn("card0", text)


if __name__ == "__main__":
    unittest.main()
