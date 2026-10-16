"""厂商抽象层测试：shell ↔ Python 注册表一致性、探测优先级、显式选择。

运行::

    python3 -m unittest tests.test_vendors -v      # 在项目根目录
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from probe.vendors import (  # noqa: E402
    AUTO_DETECT_ORDER,
    VENDOR_REGISTRY,
    detect_vendor,
    format_vendor_info,
    get_vendor,
    list_vendors,
    resolve_vendor,
    vendor_info,
)

VENDORS_DIR = ROOT / "vendors"
COMMON_SH = ROOT / "scripts" / "lib" / "common.sh"


def sh_var(profile: Path, key: str) -> str:
    """在子 shell 中 source profile 并读取变量（与 common.sh::vendor_meta 同法）。"""
    result = subprocess.run(
        ["bash", "-c", 'source "$1" >/dev/null 2>&1 || exit 1; printf "%s" "${!2:-}"', "_",
         str(profile), key],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise AssertionError(f"source {profile} 失败: {result.stderr}")
    return result.stdout


class RegistryConsistencyTests(unittest.TestCase):
    """shell 端 vendors/*.sh 与 Python 端 probe/vendors.py 必须一一对应。"""

    def test_shell_profiles_match_python_registry(self) -> None:
        shell_ids = sorted(
            p.stem for p in VENDORS_DIR.glob("*.sh") if not p.stem.startswith("_")
        )
        self.assertEqual(shell_ids, list_vendors(),
                         "vendors/*.sh 与 probe/vendors.py 注册表键不一致")

    def test_each_profile_declares_id_and_label(self) -> None:
        for vid in list_vendors():
            profile = VENDORS_DIR / f"{vid}.sh"
            self.assertEqual(sh_var(profile, "VENDOR_ID"), vid, f"{profile} VENDOR_ID 与文件名不一致")
            self.assertTrue(sh_var(profile, "VENDOR_LABEL"), f"{profile} 缺 VENDOR_LABEL")

    def test_trace_pack_and_smi_match_python(self) -> None:
        for vid in list_vendors():
            profile = VENDORS_DIR / f"{vid}.sh"
            py = VENDOR_REGISTRY[vid]
            self.assertEqual(sh_var(profile, "VENDOR_TRACE_PACK"), py.trace_pack,
                             f"{vid}: VENDOR_TRACE_PACK 两端不一致")
            self.assertEqual(sh_var(profile, "VENDOR_SMI_CMD"), py.smi_cmd,
                             f"{vid}: VENDOR_SMI_CMD 两端不一致")

    def test_profiles_are_side_effect_free(self) -> None:
        """profile 在干净子 shell 中 source 必须成功（只允许变量/函数定义）。"""
        for path in sorted(VENDORS_DIR.glob("*.sh")):
            result = subprocess.run(
                ["/bin/bash", "-c", 'source "$1"', "_", str(path)],
                capture_output=True, text=True, timeout=30,
                env={"PATH": ""},
            )
            self.assertEqual(result.returncode, 0, f"{path}: {result.stderr}")

    def test_detect_order_matches_common_sh(self) -> None:
        text = COMMON_SH.read_text(encoding="utf-8")
        match = re.search(r"for candidate in ([a-z ]+); do", text)
        self.assertIsNotNone(match, "common.sh 中找不到厂商探测顺序")
        order = tuple(match.group(1).split())  # type: ignore[union-attr]
        self.assertEqual(order, AUTO_DETECT_ORDER, "common.sh 与 probe/vendors.py 探测顺序不一致")


class DetectionTests(unittest.TestCase):
    def test_detect_returns_registered_vendor(self) -> None:
        self.assertIn(detect_vendor(), VENDOR_REGISTRY)

    def test_detect_priority_ascend_over_nvidia(self) -> None:
        which = lambda cmd: f"/usr/bin/{cmd}" if cmd in ("npu-smi", "nvidia-smi") else None
        with mock.patch("probe.vendors.shutil.which", side_effect=which), \
             mock.patch("probe.vendors.glob.glob", return_value=[]):
            self.assertEqual(detect_vendor(), "ascend")

    def test_detect_nvidia_by_smi(self) -> None:
        which = lambda cmd: "/usr/bin/nvidia-smi" if cmd == "nvidia-smi" else None
        with mock.patch("probe.vendors.shutil.which", side_effect=which), \
             mock.patch("probe.vendors.glob.glob", return_value=[]):
            self.assertEqual(detect_vendor(), "nvidia")

    def test_detect_hygon_by_device_node(self) -> None:
        def fake_glob(pattern: str) -> list[str]:
            return ["/dev/mkfd"] if pattern == "/dev/mkfd" else []

        with mock.patch("probe.vendors.shutil.which", return_value=None), \
             mock.patch("probe.vendors.glob.glob", side_effect=fake_glob):
            self.assertEqual(detect_vendor(), "hygon")

    def test_detect_fallback_generic(self) -> None:
        with mock.patch("probe.vendors.shutil.which", return_value=None), \
             mock.patch("probe.vendors.glob.glob", return_value=[]):
            self.assertEqual(detect_vendor(), "generic")

    def test_resolve_explicit_wins(self) -> None:
        with mock.patch.dict("os.environ", {"VENDOR": "amd"}):
            self.assertEqual(resolve_vendor("nvidia").id, "nvidia")
            # 显式传 auto → 环境变量 VENDOR 仍然优先于探测
            self.assertEqual(resolve_vendor("auto").id, "amd")
        with mock.patch.dict("os.environ", {"VENDOR": "auto"}):
            self.assertEqual(resolve_vendor(None).id, detect_vendor())

    def test_unknown_vendor_raises(self) -> None:
        with self.assertRaises(KeyError):
            get_vendor("nonexistent-card")


class ShellDetectionTests(unittest.TestCase):
    """common.sh 的探测行为（真实子 shell）。"""

    def _source_common(self, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        # 需要 dirname/basename/date 等基础命令，因此保留真实 PATH；只是清掉 VENDOR
        full_env = {k: v for k, v in os.environ.items() if k not in ("VENDOR",)}
        if env:
            full_env.update(env)
        return subprocess.run(
            ["/bin/bash", "-c",
             'source "$1" && printf "VENDOR=%s\\nLABEL=%s\\n" "$VENDOR" "$VENDOR_LABEL"',
             "_", str(COMMON_SH)],
            capture_output=True, text=True, timeout=60, env=full_env,
        )

    def test_explicit_vendor_loads(self) -> None:
        result = self._source_common({"VENDOR": "generic"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("VENDOR=generic", result.stdout)

    def test_unknown_vendor_dies(self) -> None:
        result = self._source_common({"VENDOR": "not-a-vendor"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("未知厂商", result.stderr)

    def test_auto_detect_on_real_host(self) -> None:
        # 不带 PATH 限制跑一次真实探测：结果必须是注册表里的合法厂商
        result = subprocess.run(
            ["/bin/bash", "-c",
             'source "$1" && printf "%s" "$VENDOR"', "_", str(COMMON_SH)],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(result.stdout.strip(), VENDOR_REGISTRY)


class VendorInfoTests(unittest.TestCase):
    def test_vendor_info_structure(self) -> None:
        info = vendor_info(requested="ascend")
        self.assertEqual(info["id"], "ascend")
        self.assertIn("smi_available", info)
        self.assertEqual(info["trace_pack"], "ascend")
        self.assertEqual(info["device_vis_env"], "ASCEND_RT_VISIBLE_DEVICES")

    def test_format_vendor_info_contains_label(self) -> None:
        text = format_vendor_info(vendor_info(requested="nvidia"))
        self.assertIn("NVIDIA CUDA", text)
        self.assertIn("CUDA_VISIBLE_DEVICES", text)


class AscendDynamicArgsTests(unittest.TestCase):
    """Ascend 的 davinci 设备参数按 VISIBLE_DEVICES 动态生成（用 /tmp 假设备节点验证）。"""

    def test_dynamic_device_args_follow_visible_devices(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dev = Path(tmp) / "dev"
            dev.mkdir()
            for name in ("davinci2", "davinci3", "davinci_manager", "devmm_svm", "hisi_hdc"):
                (dev / name).touch()
            # 把 profile 里的 /dev/ 前缀换成临时目录，其余逻辑保持原样
            patched = Path(tmp) / "ascend-test.sh"
            patched.write_text(
                (VENDORS_DIR / "ascend.sh").read_text(encoding="utf-8").replace("/dev/", f"{dev}/"),
                encoding="utf-8",
            )
            result = subprocess.run(
                ["/bin/bash", "-c",
                 'source "$1" && VISIBLE_DEVICES=2,3 vendor_dynamic_docker_args',
                 "_", str(patched)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            tokens = result.stdout.split()
            self.assertEqual(
                tokens,
                [
                    "--device", f"{dev}/davinci_manager",
                    "--device", f"{dev}/devmm_svm",
                    "--device", f"{dev}/hisi_hdc",
                    "--device", f"{dev}/davinci2",
                    "--device", f"{dev}/davinci3",
                ],
            )


if __name__ == "__main__":
    unittest.main()
