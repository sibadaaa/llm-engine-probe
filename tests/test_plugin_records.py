"""插件记录抽取的宿主机单测（不需要 vLLM / GPU）。

``plugin_probe/vllm_probe_stat_logger/records.py`` 刻意不 import vllm，
所以这里可以用鸭子类型的假对象覆盖所有分组；同时验证
"字段改名/缺失时只丢该项、不抛异常" 这一硬性要求。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_records_module():
    """按文件路径加载 ``vllm_probe_stat_logger/records.py``。

    刻意**不走包导入**：包的 ``__init__.py`` 需要 ``import vllm``（宿主机没有），
    而 ``records.py`` 被设计成纯标准库、可离线测试。这里用按路径加载来锁住这个约束。
    """
    path = ROOT / "plugin_probe" / "vllm_probe_stat_logger" / "records.py"
    spec = importlib.util.spec_from_file_location("vllm_probe_stat_logger_records", path)
    assert spec and spec.loader, f"无法加载 {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_records = _load_records_module()
MAX_SAMPLES = _records.MAX_SAMPLES
SCHEMA_VERSION = _records.SCHEMA_VERSION
build_record = _records.build_record
summarise = _records.summarise

from probe.plugin_records import load_records, summarize as summarize_file  # noqa: E402


def fake_stats() -> SimpleNamespace:
    """构造一份与 vLLM 0.18.1 字段同名、但完全不依赖 vllm 的统计对象。"""
    return SimpleNamespace(
        scheduler_stats=SimpleNamespace(
            num_running_reqs=3,
            num_waiting_reqs=2,
            num_skipped_waiting_reqs=1,
            step_counter=42,
            current_wave=5,
            kv_cache_usage=0.37,
            iteration_details=SimpleNamespace(
                iteration_index=11,
                num_ctx_requests=1,
                num_ctx_tokens=1024,
                num_generation_requests=2,
                num_generation_tokens=16,
                elapsed_ms=210.5,
                num_encoder_inputs=0,
                num_encoder_output_tokens=0,
                is_dummy=False,
            ),
            prefix_cache_stats=SimpleNamespace(
                requests=4, queries=4096, hits=2048,
                preempted_requests=0, preempted_queries=0, preempted_hits=0,
            ),
            connector_prefix_cache_stats=None,
            spec_decoding_stats=SimpleNamespace(
                num_spec_tokens=7,
                num_drafts=2,
                num_draft_tokens=14,
                num_accepted_tokens=6,
                num_accepted_tokens_per_pos=[3, 2, 1, 0, 0, 0, 0],
                num_draft_tokens_per_pos=[2] * 7,
            ),
            kv_cache_eviction_events=[
                SimpleNamespace(lifetime_seconds=12.5, idle_seconds=3.0, reuse_gaps_seconds=(0.1, 0.2)),
            ],
            cudagraph_stats=SimpleNamespace(
                num_unpadded_tokens=8, num_padded_tokens=8, num_paddings=0, runtime_mode="NONE"
            ),
            perf_stats=SimpleNamespace(
                num_flops_per_gpu=1.5e12, num_read_bytes_per_gpu=3.0e10,
                num_write_bytes_per_gpu=1.0e10,
            ),
        ),
        iteration_stats=SimpleNamespace(
            iteration_timestamp=1000.5,
            num_generation_tokens=16,
            prompt_token_stats=SimpleNamespace(
                computed=512, local_cache_hit=512, external_kv_transfer=0,
                cached_tokens=512, total=1024,
            ),
            num_preempted_reqs=0,
            max_num_generation_tokens_iter=8,
            n_params_iter=None,
            num_corrupted_reqs=0,
            time_to_first_tokens_iter=[0.31, 0.29],
            inter_token_latencies_iter=[0.05, 0.06, 0.055],
            finished_requests=[
                SimpleNamespace(
                    finish_reason="stop", request_id="req-1", e2e_latency=1.85,
                    num_prompt_tokens=1024, num_preemptions=1, num_generation_tokens=24,
                    max_tokens_param=128, queued_time=0.004, prefill_time=0.31,
                    inference_time=1.54, decode_time=1.23, mean_time_per_output_token=0.052,
                    is_corrupted=False, num_cached_tokens=512,
                )
            ],
        ),
    )


class RecordsExtractionTests(unittest.TestCase):
    def test_records_module_stays_vllm_free(self) -> None:
        """``records.py`` 必须保持"纯标准库"：一旦 import vllm，宿主机就无法单测。"""
        source = (ROOT / "plugin_probe" / "vllm_probe_stat_logger" / "records.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("\nimport vllm", source)
        self.assertNotIn("\nfrom vllm", source)

    def test_build_record_all_groups(self) -> None:
        stats = fake_stats()
        record = build_record(stats.scheduler_stats, stats.iteration_stats, engine_index=0, step=7)
        self.assertEqual(record["schema"], SCHEMA_VERSION)
        self.assertEqual(record["step"], 7)
        self.assertEqual(record["iter"]["ctx_tokens"], 1024)
        self.assertEqual(record["sched"]["running"], 3)
        self.assertAlmostEqual(record["sched"]["kv_cache_usage"], 0.37)
        self.assertEqual(record["prefix"]["hits"], 2048)
        self.assertAlmostEqual(record["spec"]["draft_acceptance_rate"], 6 / 14)
        self.assertAlmostEqual(record["spec"]["mean_acceptance_length"], 1 + 6 / 2)
        self.assertEqual(record["spec"]["num_accepted_tokens_per_pos"][:3], [3, 2, 1])
        self.assertEqual(record["evict"][0]["lifetime_s"], 12.5)
        self.assertEqual(record["graph"]["runtime_mode"], "NONE")
        self.assertAlmostEqual(record["perf"]["flops_per_gpu"], 1.5e12)
        self.assertEqual(record["prompt_tokens"]["local_cache_hit"], 512)
        req = record["reqs"][0]
        self.assertEqual(req["id"], "req-1")
        self.assertEqual(req["cached_tokens"], 512)
        self.assertEqual(req["preemptions"], 1)
        self.assertAlmostEqual(req["prefill_s"], 0.31)
        self.assertAlmostEqual(req["decode_s"], 1.23)
        self.assertEqual(len(record["samples"]["ttft_s"]), 2)
        # JSON 可序列化（这是插件真正写盘前的最后一道关）
        json.dumps(record)

    def test_none_stats_is_safe(self) -> None:
        record = build_record(None, None, engine_index=0, step=1)
        self.assertEqual(record["step"], 1)
        self.assertNotIn("iter", record)
        self.assertNotIn("reqs", record)

    def test_broken_object_does_not_raise(self) -> None:
        class Exploding:
            def __getattr__(self, name):  # noqa: D105
                raise RuntimeError("field exploded")

        record = build_record(Exploding(), Exploding(), engine_index=0, step=2)
        self.assertEqual(record["step"], 2)  # 只有 ts/engine/step
        json.dumps(record)

    def test_unknown_fields_are_dropped(self) -> None:
        """vLLM 字段改名时，缺失项应为 None 而不是异常。"""
        stats = SimpleNamespace(scheduler_stats=SimpleNamespace(num_running_reqs=1),
                                iteration_stats=SimpleNamespace())
        record = build_record(stats.scheduler_stats, stats.iteration_stats)
        self.assertEqual(record["sched"]["running"], 1)
        self.assertNotIn("kv_cache_usage", record["sched"])
        self.assertNotIn("iter", record)

    def test_samples_are_bounded(self) -> None:
        stats = fake_stats()
        stats.iteration_stats.inter_token_latencies_iter = [0.01] * (MAX_SAMPLES * 3)
        record = build_record(stats.scheduler_stats, stats.iteration_stats)
        self.assertEqual(len(record["samples"]["itl_s"]), MAX_SAMPLES)

    def test_summarise(self) -> None:
        stats = fake_stats()
        records = [
            build_record(stats.scheduler_stats, stats.iteration_stats, step=i, ts=1000.0 + i)
            for i in range(4)
        ]
        summary = summarise(records)
        self.assertEqual(summary["steps"], 4)
        self.assertAlmostEqual(summary["step_ms"]["mean"], 210.5)
        self.assertEqual(summary["context_tokens"], 4096)
        self.assertEqual(summary["generation_tokens"], 64)
        self.assertEqual(summary["spec_decode"]["accepted_tokens"], 24)
        self.assertEqual(summary["kv_evictions"], 4)
        self.assertEqual(summary["finished_requests"], 4)
        self.assertEqual(summary["finished_finish_reasons"], {"stop": 4})
        self.assertAlmostEqual(summary["request_prefill_s"]["mean"], 0.31)
        self.assertEqual(summary["request_cached_tokens"], 2048)
        self.assertEqual(summary["request_preemptions"], 4)
        self.assertEqual(summary["cudagraph_modes"], {"NONE": 4})


class PluginReaderTests(unittest.TestCase):
    """主工具箱侧的读取/汇总（probe/plugin_records.py）。"""

    def _write_jsonl(self, path: Path) -> None:
        stats = fake_stats()
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"schema": 1, "kind": "header", "vllm_version": "0.18.1+das"}) + "\n")
            for i in range(3):
                fh.write(
                    json.dumps(
                        build_record(stats.scheduler_stats, stats.iteration_stats, step=i, ts=100.0 + i)
                    )
                    + "\n"
                )

    def test_load_and_summarize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe_steps.jsonl"
            self._write_jsonl(path)
            records, header = load_records(path)
            self.assertEqual(len(records), 3)
            self.assertEqual(header.get("kind"), "header")
            summary = summarize_file(records, header)
            self.assertEqual(summary["steps"], 3)
            self.assertEqual(summary["finished_requests"], 3)
            self.assertAlmostEqual(summary["request_e2e_ms"]["mean_ms"], 1850.0)
            self.assertAlmostEqual(summary["request_tpot_ms"]["mean_ms"], 52.0)
            self.assertAlmostEqual(summary["kv_cache_usage_perc"]["max"], 37.0)
            self.assertAlmostEqual(summary["prefix_cache_hit_rate"], 0.5)
            per_pos = summary["spec_decode"]["accepted_per_pos"]
            # 假数据里每个 step 是 [3,2,1,0,0,0,0]（k=7），3 个 step 求和后仍保留 7 个位置
            self.assertEqual([per_pos[i] for i in sorted(per_pos)], [9, 6, 3, 0, 0, 0, 0])
            self.assertAlmostEqual(summary["ttft_ms"]["mean_ms"], 300.0)
            json.dumps(summary)

    def test_missing_file_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records, header = load_records(Path(tmp) / "nope.jsonl")
            self.assertEqual(records, [])
            self.assertEqual(header, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
