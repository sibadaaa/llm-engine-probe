"""``probe hostcheck`` 单元测试：结构契约、退出码语义、优雅降级。

运行::

    python3 -m unittest tests.test_hostcheck -v      # 在项目根目录

测试策略：用临时目录 + 假 ``HOST_PYTHON``（bash 脚本按 ``FAKE_PY_MODE`` 输出
hostcheck 汇总探针期望的 JSON），因此**不依赖真 vllm/torch/GPU**。
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TOP_KEYS = {
    "kind",
    "created_at",
    "runtime",
    "host_python",
    "vendor",
    "ok",
    "summary",
    "versions",
    "checks",
}
CHECK_KEYS = {"id", "name", "level", "ok", "detail", "hint"}
CHECK_IDS = [
    "os",
    "host_python",
    "python_version",
    "vllm",
    "torch",
    "gpu_smi",
    "model_path",
    "port",
    "curl",
    "py_spy",
    "ptrace",
    "ulimit",
    "shm",
    "prelude",
]

# 假 HOST_PYTHON：按 FAKE_PY_MODE 输出 JSON（含 hostcheck 的标记行）。
FAKE_PY = r"""#!/usr/bin/env bash
# 假 HOST_PYTHON：按 FAKE_PY_MODE 输出 hostcheck 汇总探针期望的 JSON。
# 不真正 import torch/vllm，因此测试不依赖 GPU / 真实依赖。
set -u
mode="${FAKE_PY_MODE:-full}"
if [[ "$mode" == "broken" ]]; then
  printf 'fake-python: 模拟解释器失败\n' >&2
  printf 'this is not json\n'
  exit 3
fi
pyver="3.11.0"
pyarr="[3, 11, 0]"
torch_json='"2.6.0"'
torch_err="null"
cuda="true"
count="4"
hip="null"
npu="null"
vllm_json='"0.18.0"'
vllm_err="null"
case "$mode" in
  full) ;;
  no-vllm)
    vllm_json="null"
    vllm_err='"ModuleNotFoundError: No module named vllm"'
    ;;
  no-torch)
    torch_json="null"
    torch_err='"ModuleNotFoundError: No module named torch"'
    cuda="false"
    count="0"
    vllm_json="null"
    vllm_err='"RuntimeError: vllm requires torch"'
    ;;
  low-python)
    pyver="3.8.10"
    pyarr="[3, 8, 10]"
    ;;
  *)
    printf 'fake-python: 未知 FAKE_PY_MODE=%s\n' "$mode" >&2
    exit 2
    ;;
esac
printf '__HOSTCHECK_PYINFO__{"executable":"%s","version":"%s","version_info":%s,"platform":"Linux-fake","machine":"x86_64","torch":%s,"torch_error":%s,"cuda_available":%s,"hip":%s,"device_count":%s,"npu_available":%s,"vllm":%s,"vllm_error":%s}\n' \
  "$0" "$pyver" "$pyarr" "$torch_json" "$torch_err" "$cuda" "$hip" "$count" "$npu" "$vllm_json" "$vllm_err"
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class HostCheckTestBase(unittest.TestCase):
    """公共夹具：临时目录、假解释器、合法模型目录、空闲端口。"""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="hostcheck-test-")
        self.tmp = Path(self._tmpdir.name)
        self.outdir = self.tmp / "out"
        self.model = self._make_model("model")
        self.port = _free_port()
        self.has_curl = shutil.which("curl") is not None
        self._fake_cache: dict[str, Path] = {}

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    # -- helpers --------------------------------------------------------
    def _make_model(self, name: str, tokenizer: bool = True) -> Path:
        model = self.tmp / name
        model.mkdir(parents=True, exist_ok=True)
        (model / "config.json").write_text("{}", encoding="utf-8")
        if tokenizer:
            (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer.json").write_text("{}", encoding="utf-8")
        return model

    def fake_python(self, mode: str = "full") -> Path:
        if mode not in self._fake_cache:
            path = self.tmp / f"fake-python-{mode}"
            path.write_text(FAKE_PY, encoding="utf-8")
            path.chmod(0o755)
            self._fake_cache[mode] = path
        return self._fake_cache[mode]

    def _clean_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        for name in ("HOST_PYTHON", "MODEL", "PORT", "HOST_PRELUDE_FILES", "PROBE_OUTDIR", "PYSPY_BIN"):
            env.pop(name, None)
        env["VENDOR"] = "generic"
        env["FAKE_PY_MODE"] = "full"
        return env

    def run_cli(self, *args: str, mode: str = "full", env_extra: dict[str, str] | None = None):
        env = self._clean_env()
        env["FAKE_PY_MODE"] = mode
        if env_extra:
            env.update(env_extra)
        return subprocess.run(
            [sys.executable, "-m", "probe", "hostcheck", *args],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )

    def run_json(self, *args: str, mode: str = "full", env_extra: dict[str, str] | None = None):
        """跑一次 --json 并返回 (proc, payload, json_path)。"""
        outdir = Path(args[args.index("--outdir") + 1]) if "--outdir" in args else self.outdir
        proc = self.run_cli(*args, mode=mode, env_extra=env_extra)
        json_path = outdir / "host_check.json"
        self.assertTrue(json_path.is_file(), f"未生成 JSON：{json_path}\n{proc.stdout}\n{proc.stderr}")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        return proc, payload, json_path

    def base_args(self, fake: Path) -> list[str]:
        return [
            "--host-python", str(fake),
            "--model", str(self.model),
            "--port", str(self.port),
            "--json",
            "--outdir", str(self.outdir),
            "--skip-gpu",
        ]

    @staticmethod
    def checks_by_id(payload: dict) -> dict[str, dict]:
        return {item["id"]: item for item in payload["checks"]}


class StructureTests(HostCheckTestBase):
    def test_structure_and_json_output(self) -> None:
        fake = self.fake_python("full")
        proc, payload, json_path = self.run_json(*self.base_args(fake))

        # 顶层 schema 稳定
        self.assertEqual(set(payload), TOP_KEYS)
        self.assertEqual(payload["kind"], "vllm-probe-hostcheck")
        self.assertEqual(payload["runtime"], "host")
        self.assertEqual(payload["host_python"], str(fake))
        self.assertEqual(payload["vendor"], "generic")
        self.assertIsInstance(payload["ok"], bool)
        self.assertTrue(json_path.is_file())

        # created_at 是 ISO 8601
        from datetime import datetime

        datetime.fromisoformat(payload["created_at"])

        # summary 与退出码一致
        summary = payload["summary"]
        self.assertEqual(set(summary), {"pass", "warn", "fail"})
        self.assertTrue(all(isinstance(value, int) for value in summary.values()))
        self.assertEqual(payload["ok"], summary["fail"] == 0)
        self.assertEqual(proc.returncode, 0 if payload["ok"] else 1)

        # checks：固定 id、固定顺序、字段齐全、level 枚举合法
        self.assertEqual([item["id"] for item in payload["checks"]], CHECK_IDS)
        self.assertEqual(summary["pass"] + summary["warn"] + summary["fail"], len(CHECK_IDS))
        for item in payload["checks"]:
            self.assertEqual(set(item), CHECK_KEYS)
            self.assertIn(item["level"], {"fail", "warn", "info"})
            self.assertIsInstance(item["ok"], bool)
            self.assertIsInstance(item["detail"], str)
            self.assertIsInstance(item["hint"], str)
            if item["ok"]:
                self.assertEqual(item["level"], "info")
            else:
                self.assertIn(item["level"], {"fail", "warn"})

        # versions 记录了解释器 / torch / vllm
        versions = payload["versions"]
        self.assertIsInstance(versions, dict)
        self.assertEqual(versions["python"], str(fake))
        self.assertEqual(versions["torch"], "2.6.0")
        self.assertEqual(versions["vllm"], "0.18.0")
        self.assertEqual(versions["device_count"], 4)

        # 人读输出含汇总，且写盘提示出现
        self.assertIn("汇总", proc.stdout)
        self.assertIn("JSON 已写入", proc.stdout)

    def test_json_default_outdir_uses_probe_outdir(self) -> None:
        """--json 未指定 --outdir 时使用 $PROBE_OUTDIR。"""
        fake = self.fake_python("full")
        probe_outdir = self.tmp / "probe-outdir"
        args = [
            "--host-python", str(fake),
            "--model", str(self.model),
            "--port", str(self.port),
            "--json",
            "--skip-gpu",
        ]
        proc = self.run_cli(*args, env_extra={"PROBE_OUTDIR": str(probe_outdir)})
        json_path = probe_outdir / "host_check.json"
        self.assertTrue(json_path.is_file(), proc.stdout + proc.stderr)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["kind"], "vllm-probe-hostcheck")


class ExitCodeTests(HostCheckTestBase):
    def test_missing_vllm_is_fail_and_exit_1(self) -> None:
        fake = self.fake_python("no-vllm")
        proc, payload, _ = self.run_json(*self.base_args(fake), mode="no-vllm")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertFalse(payload["ok"])
        checks = self.checks_by_id(payload)
        self.assertFalse(checks["vllm"]["ok"])
        self.assertEqual(checks["vllm"]["level"], "fail")
        self.assertIn("vllm", checks["vllm"]["detail"])
        self.assertGreaterEqual(payload["summary"]["fail"], 1)
        # torch 本身可用 → 不应被连带判 fail
        self.assertTrue(checks["torch"]["ok"])

    def test_skip_flags_exit_zero_with_valid_model_and_port(self) -> None:
        if not self.has_curl:
            self.skipTest("环境缺少 curl（curl 是 fail 级检查项），跳过退出码 0 用例")
        fake = self.fake_python("full")
        args = [
            "--host-python", str(fake),
            "--model", str(self.model),
            "--port", str(self.port),
            "--json",
            "--outdir", str(self.outdir),
            "--skip-vllm-import",
            "--skip-gpu",
        ]
        proc, payload, _ = self.run_json(*args)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["summary"]["fail"], 0)
        checks = self.checks_by_id(payload)
        # 被跳过的项是 info，不是 fail
        self.assertEqual(checks["vllm"]["level"], "info")
        self.assertIn("跳过", checks["vllm"]["detail"])
        self.assertEqual(checks["gpu_smi"]["level"], "info")
        self.assertIn("跳过", checks["gpu_smi"]["detail"])

    def test_invalid_model_path_is_fail(self) -> None:
        fake = self.fake_python("full")
        args = [
            "--host-python", str(fake),
            "--model", str(self.tmp / "no-such-model"),
            "--port", str(self.port),
            "--json",
            "--outdir", str(self.outdir),
            "--skip-vllm-import",
            "--skip-gpu",
        ]
        proc, payload, _ = self.run_json(*args)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        checks = self.checks_by_id(payload)
        self.assertFalse(checks["model_path"]["ok"])
        self.assertEqual(checks["model_path"]["level"], "fail")
        self.assertIn("不存在", checks["model_path"]["detail"])
        self.assertFalse(payload["ok"])

        # 目录存在但缺 config.json 也是 fail
        empty_model = self.tmp / "empty-model"
        empty_model.mkdir()
        args[args.index(str(self.tmp / "no-such-model"))] = str(empty_model)
        proc2, payload2, _ = self.run_json(*args)
        self.assertEqual(proc2.returncode, 1)
        checks2 = self.checks_by_id(payload2)
        self.assertFalse(checks2["model_path"]["ok"])
        self.assertIn("config.json", checks2["model_path"]["detail"])

    def test_low_python_version_is_warn_only(self) -> None:
        fake = self.fake_python("low-python")
        proc, payload, _ = self.run_json(*self.base_args(fake), mode="low-python")
        checks = self.checks_by_id(payload)
        self.assertFalse(checks["python_version"]["ok"])
        self.assertEqual(checks["python_version"]["level"], "warn")
        # warn 不阻塞：除非 curl 缺失，退出码应为 0
        if self.has_curl:
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertTrue(payload["ok"])


class GracefulTests(HostCheckTestBase):
    def test_missing_host_python_graceful(self) -> None:
        args = [
            "--host-python", str(self.tmp / "no-such-python"),
            "--model", str(self.model),
            "--port", str(self.port),
            "--json",
            "--outdir", str(self.outdir),
            "--skip-gpu",
        ]
        proc, payload, _ = self.run_json(*args)
        self.assertEqual(proc.returncode, 1)
        self.assertNotIn("Traceback", proc.stderr)
        checks = self.checks_by_id(payload)
        self.assertFalse(checks["host_python"]["ok"])
        self.assertEqual(checks["host_python"]["level"], "fail")
        # 依赖解释器的检查优雅跳过，而不是崩栈
        self.assertEqual(checks["vllm"]["level"], "info")
        self.assertIn("跳过", checks["vllm"]["detail"])

    def test_broken_python_output_graceful(self) -> None:
        """HOST_PYTHON 输出垃圾/非零退出：转成 fail/warn，不抛栈。"""
        fake = self.fake_python("broken")
        proc, payload, _ = self.run_json(*self.base_args(fake), mode="broken")
        self.assertEqual(proc.returncode, 1)
        self.assertNotIn("Traceback", proc.stderr)
        checks = self.checks_by_id(payload)
        self.assertFalse(checks["torch"]["ok"])
        self.assertEqual(checks["torch"]["level"], "fail")
        self.assertFalse(checks["vllm"]["ok"])
        self.assertEqual(checks["vllm"]["level"], "fail")
        self.assertEqual(checks["python_version"]["level"], "warn")

    def test_invalid_vendor_does_not_crash(self) -> None:
        fake = self.fake_python("full")
        proc, payload, _ = self.run_json(*self.base_args(fake), env_extra={"VENDOR": "bogus-vendor"})
        self.assertNotIn("Traceback", proc.stderr)
        self.assertEqual(payload["vendor"], "bogus-vendor")
        self.assertEqual(proc.returncode, 0 if payload["ok"] else 1)


class ModeTests(HostCheckTestBase):
    def test_start_mode_skips_slow_checks(self) -> None:
        fake = self.fake_python("full")
        args = [
            "--host-python", str(fake),
            "--model", str(self.model),
            "--port", str(self.port),
            "--json",
            "--outdir", str(self.outdir),
            "--start",
        ]
        proc, payload, _ = self.run_json(*args)
        checks = self.checks_by_id(payload)
        # --start：只跑 fail 级核心项，其余全部跳过
        for check_id in ("vllm", "torch", "gpu_smi", "py_spy", "ptrace", "ulimit", "shm", "prelude"):
            self.assertEqual(checks[check_id]["level"], "info", check_id)
            self.assertIn("跳过", checks[check_id]["detail"], check_id)
        for check_id in ("os", "host_python", "python_version", "model_path", "port", "curl"):
            self.assertNotIn("已跳过", checks[check_id]["detail"], check_id)
        if self.has_curl:
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertTrue(payload["ok"])


class ShellWrapperTests(HostCheckTestBase):
    def test_shell_wrapper_syntax(self) -> None:
        script = ROOT / "scripts" / "host_check.sh"
        self.assertTrue(script.is_file())
        proc = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_shell_wrapper_end_to_end(self) -> None:
        """start_vllm.sh 的调用方式：bash scripts/host_check.sh --json --outdir <dir>。"""
        if shutil.which("bash") is None:  # pragma: no cover
            self.skipTest("环境缺少 bash")
        if not self.has_curl:
            self.skipTest("环境缺少 curl（curl 是 fail 级检查项），跳过退出码 0 用例")
        fake = self.fake_python("full")
        outdir = self.tmp / "wrapper-out"
        env = self._clean_env()
        env.update(
            {
                "HOST_PYTHON": str(fake),
                "MODEL": str(self.model),
                "PORT": str(self.port),
                "VENDOR": "generic",
                "FAKE_PY_MODE": "full",
                "PROBE_OUTDIR": str(outdir),
            }
        )
        proc = subprocess.run(
            ["bash", "scripts/host_check.sh", "--json", "--outdir", str(outdir), "--skip-gpu"],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        json_path = outdir / "host_check.json"
        self.assertTrue(json_path.is_file(), proc.stdout + proc.stderr)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["kind"], "vllm-probe-hostcheck")
        self.assertEqual(payload["runtime"], "host")
        self.assertEqual(payload["host_python"], str(fake))


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
