"""引擎组件注册表测试：shell ↔ Python 注册表一致性、能力词表、解析优先级。

复刻 tests/test_vendors.py 的四件套模式，外加能力位（caps）专项；运行::

    python3 -m unittest tests.test_engines -v      # 在项目根目录

契约：docs/design/component-interface/设计.md §3/§5/§6。
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from probe.engines import (  # noqa: E402
    CAP_VOCAB,
    ENGINE_REGISTRY,
    effective_metric_prefix,
    engine_info,
    format_engine_info,
    get_engine,
    has_cap,
    list_engines,
    resolve_engine,
)

ENGINES_DIR = ROOT / "engines"
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


def sh_source_common(env: dict[str, str] | None = None, script: str = "") -> subprocess.CompletedProcess:
    """source common.sh 后执行 script（继承真实 PATH，只覆盖指定环境变量）。"""
    full_env = {k: v for k, v in os.environ.items() if k != "ENGINE"}
    if env:
        full_env.update(env)
    body = 'source "$1" || exit 1'
    if script:
        body += "\n" + script
    return subprocess.run(
        ["/bin/bash", "-c", body, "_", str(COMMON_SH)],
        capture_output=True, text=True, timeout=60, env=full_env,
    )


class RegistryConsistencyTests(unittest.TestCase):
    """shell 端 engines/*.sh 与 Python 端 probe/engines.py 必须一一对应。"""

    def test_shell_profiles_match_python_registry(self) -> None:
        shell_ids = sorted(
            p.stem for p in ENGINES_DIR.glob("*.sh") if not p.stem.startswith("_")
        )
        self.assertEqual(shell_ids, list_engines(),
                         "engines/*.sh 与 probe/engines.py 注册表键不一致")

    def test_each_profile_declares_id_and_label(self) -> None:
        for eid in list_engines():
            profile = ENGINES_DIR / f"{eid}.sh"
            self.assertEqual(sh_var(profile, "ENGINE_ID"), eid, f"{profile} ENGINE_ID 与文件名不一致")
            self.assertTrue(sh_var(profile, "ENGINE_LABEL"), f"{profile} 缺 ENGINE_LABEL")

    def test_metrics_flavor_args_mode_and_caps_match_python(self) -> None:
        for eid in list_engines():
            profile = ENGINES_DIR / f"{eid}.sh"
            py = ENGINE_REGISTRY[eid]
            self.assertEqual(sh_var(profile, "ENGINE_METRICS_FLAVOR"), py.metrics_flavor,
                             f"{eid}: ENGINE_METRICS_FLAVOR 两端不一致")
            self.assertEqual(sh_var(profile, "ENGINE_ARGS_MODE"), py.args_mode,
                             f"{eid}: ENGINE_ARGS_MODE 两端不一致")
            shell_caps = set(sh_var(profile, "ENGINE_CAPS").split())
            self.assertEqual(shell_caps, set(py.caps),
                             f"{eid}: ENGINE_CAPS 两端不一致（顺序无关的集合必须相等）")

    def test_mechanical_shell_fields_match_python(self) -> None:
        """双端逐字段一致（评审 M4）：所有可机械比对的 ENGINE_* 字段全部对齐，
        防止只改一端（如改 shell 的就绪指标不动 Python）时测试仍全绿。"""
        field_map = (
            ("ENGINE_ENTRYPOINT", "entrypoint"),
            ("ENGINE_PROCESS_PATTERN", "process_pattern"),
            ("ENGINE_HEALTH_PATH", "health_path"),
            ("ENGINE_MODELS_PATH", "models_path"),
            ("ENGINE_READY_METRIC", "ready_metric"),
            ("ENGINE_METRICS_FETCH_SUFFIX", "metrics_fetch_suffix"),
            ("ENGINE_VERSION_CMD", "version_cmd"),
            ("ENGINE_CHECK_SEVERITY", "check_severity"),
            ("ENGINE_DOCKER_IMAGE", "docker_image"),
        )
        for eid in list_engines():
            profile = ENGINES_DIR / f"{eid}.sh"
            py = ENGINE_REGISTRY[eid]
            for sh_key, py_attr in field_map:
                self.assertEqual(
                    sh_var(profile, sh_key), getattr(py, py_attr),
                    f"{eid}: {sh_key} 两端不一致",
                )

    def test_caps_within_vocab(self) -> None:
        """所有声明的能力键必须 ∈ CAP_VOCAB（防拼写错误静默失效）。"""
        for eid in list_engines():
            py_caps = ENGINE_REGISTRY[eid].caps
            self.assertFalse(set(py_caps) - set(CAP_VOCAB),
                             f"{eid}: Python 端声明了词表外的能力键: {set(py_caps) - set(CAP_VOCAB)}")
            shell_caps = set(sh_var(ENGINES_DIR / f"{eid}.sh", "ENGINE_CAPS").split())
            self.assertFalse(set(shell_caps) - set(CAP_VOCAB),
                             f"{eid}: shell 端声明了词表外的能力键: {set(shell_caps) - set(CAP_VOCAB)}")

    def test_profiles_are_side_effect_free(self) -> None:
        """profile 在干净子 shell（PATH=""）中 source 必须成功（只允许变量/函数定义）。"""
        for path in sorted(ENGINES_DIR.glob("*.sh")):
            result = subprocess.run(
                ["/bin/bash", "-c", 'source "$1"', "_", str(path)],
                capture_output=True, text=True, timeout=30,
                env={"PATH": ""},
            )
            self.assertEqual(result.returncode, 0, f"{path}: {result.stderr}")


class ResolveTests(unittest.TestCase):
    """resolve_engine 优先级：显式参数 > ENGINE 环境变量 > vllm；auto 别名。"""

    def test_explicit_wins(self) -> None:
        with mock_env("vllm"):
            self.assertEqual(resolve_engine("vllm").id, "vllm")

    def test_env_used_when_no_explicit(self) -> None:
        with mock_env("vllm"):
            self.assertEqual(resolve_engine().id, "vllm")

    def test_default_vllm(self) -> None:
        with mock_env(""):
            self.assertEqual(resolve_engine().id, "vllm")
        with mock_env(None):
            self.assertEqual(resolve_engine().id, "vllm")

    def test_auto_alias_ignores_empty_env(self) -> None:
        # auto 语义同 VENDOR=auto：忽略显式空值，直接用环境/默认
        with mock_env(""):
            self.assertEqual(resolve_engine("auto").id, "vllm")

    def test_unknown_engine_raises(self) -> None:
        with self.assertRaises(KeyError):
            get_engine("no-such-engine")
        with mock_env("no-such-engine"):
            with self.assertRaises(KeyError):
                resolve_engine()

    def test_has_cap(self) -> None:
        profile = get_engine("vllm")
        self.assertTrue(has_cap(profile, "api.openai-chat"))
        self.assertFalse(has_cap(profile, "obs.not-a-cap"))


class EngineInfoTests(unittest.TestCase):
    def test_engine_info_structure(self) -> None:
        info = engine_info(requested="vllm")
        self.assertEqual(info["id"], "vllm")
        self.assertIn("api.openai-chat", info["caps"])
        self.assertEqual(info["metrics_flavor"], "vllm-named")
        self.assertEqual(info["args_mode"], "vllm-cli")
        self.assertEqual(info["unknown_caps"], [])

    def test_format_engine_info_contains_label_and_caps(self) -> None:
        text = format_engine_info(engine_info(requested="vllm"))
        self.assertIn("vLLM", text)
        self.assertIn("api.openai-chat", text)
        self.assertIn("metrics 形态: vllm-named", text)


class ProfileExtensionTests(unittest.TestCase):
    """T3.1：Python 侧专有字段的派生规则与基线保护。"""

    def test_vllm_chat_body_extras_match_legacy_behavior(self) -> None:
        """vllm 的 chat_body_extras 必须等于历史硬编码注入（防手滑改坏请求体基线）。"""
        profile = get_engine("vllm")
        self.assertEqual(
            dict(profile.chat_body_extras),
            {"chat_template_kwargs": {"reasoning_effort": "low"}},
        )
        self.assertEqual(
            profile.delta_content_keys, ("content", "reasoning_content", "reasoning")
        )

    def test_metric_prefix_derivation(self) -> None:
        """vllm-named → 必须以 vllm: 开头（或显式设置）；prometheus/none → 不过滤。"""
        self.assertEqual(effective_metric_prefix(get_engine("vllm")), "vllm:")
        self.assertEqual(effective_metric_prefix(get_engine("strata")), "vllm:")
        prom = replace(get_engine("vllm"), metrics_flavor="prometheus", metric_prefix="")
        self.assertEqual(effective_metric_prefix(prom), "")
        explicit = replace(get_engine("strata"), metric_prefix="engine:")
        self.assertEqual(effective_metric_prefix(explicit), "engine:")

    def test_env_check_and_delta_keys_non_empty(self) -> None:
        for eid in list_engines():
            profile = ENGINE_REGISTRY[eid]
            self.assertTrue(profile.env_check_metrics, f"{eid}: env_check_metrics 不能为空")
            self.assertTrue(profile.delta_content_keys, f"{eid}: delta_content_keys 不能为空")

    def test_strata_python_only_fields(self) -> None:
        profile = get_engine("strata")
        self.assertEqual(dict(profile.chat_body_extras), {"reasoning_effort": "low"})
        self.assertEqual(profile.delta_content_keys, ("content", "reasoning_content"))
        self.assertIn("strata:last_", profile.extra_metric_families)
        self.assertNotIn("obs.per-request-metrics", profile.caps)


class ShellLoaderTests(unittest.TestCase):
    """common.sh 的引擎层函数（真实子 shell 验证）。"""

    def test_engine_meta_loads_vllm_profile(self) -> None:
        result = sh_source_common(
            {"ENGINE": "vllm"},
            'engine_meta; printf "ID=%s\\nLABEL=%s\\nCAPS=%s\\n" "$ENGINE_ID" "$ENGINE_LABEL" "$ENGINE_CAPS"',
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ID=vllm", result.stdout)
        self.assertIn("LABEL=vLLM（默认引擎）", result.stdout)
        self.assertIn("obs.step-log", result.stdout)

    def test_engine_meta_default_vllm(self) -> None:
        # ENGINE 未设置时默认 vllm
        result = sh_source_common(
            {"ENGINE": ""},
            'engine_meta; printf "ID=%s" "$ENGINE_ID"',
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ID=vllm")

    def test_unknown_engine_dies_with_list(self) -> None:
        result = sh_source_common({"ENGINE": "no-such-engine"}, "engine_meta")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("未知引擎", result.stderr)
        self.assertIn("vllm", result.stderr)  # 文案列出可用引擎

    def test_engine_has_cap_word_match(self) -> None:
        script = (
            'engine_meta; '
            'if engine_has_cap obs.step-log; then echo "A:0"; else echo "A:1"; fi; '
            'if engine_has_cap api.openai-chat; then echo "B:0"; else echo "B:1"; fi; '
            'if engine_has_cap obs.step; then echo "C:0"; else echo "C:1"; fi; '
            'if engine_has_cap step-log; then echo "D:0"; else echo "D:1"; fi; '
            'if engine_has_cap api.not-in-caps; then echo "E:0"; else echo "E:1"; fi'
        )
        result = sh_source_common({"ENGINE": "vllm"}, script)
        self.assertEqual(result.returncode, 0, result.stderr)
        flags = dict(line.split(":", 1) for line in result.stdout.splitlines() if ":" in line)
        # 声明的能力键必须命中；词前缀/后缀/未声明键不得命中（防 grep -qw 的边界误判）
        self.assertEqual(flags["A"], "0", "obs.step-log 应命中")
        self.assertEqual(flags["B"], "0", "api.openai-chat 应命中")
        self.assertEqual(flags["C"], "1", "obs.step 不得命中 obs.step-log（词边界）")
        self.assertEqual(flags["D"], "1", "step-log 不得作为后缀命中")
        self.assertEqual(flags["E"], "1", "未声明键不得命中")

    def test_engine_list_contains_vllm(self) -> None:
        result = sh_source_common({}, 'printf "%s\\n" $(engine_list)')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("vllm", result.stdout.split())

    def test_engine_process_pattern_chain(self) -> None:
        """进程正则优先级链（评审低-10）：显式 HOST_ENGINE_PATTERN > profile > HOST_VLLM_PATTERN。"""
        # 引擎 profile 声明的正则生效
        by_profile = sh_source_common(
            {"ENGINE": "strata"},
            'engine_meta; printf "%s" "$(engine_process_pattern)"',
        )
        self.assertEqual(by_profile.returncode, 0, by_profile.stderr)
        self.assertEqual(by_profile.stdout.strip(), r"strata|server\.py")
        # 显式 HOST_ENGINE_PATTERN 最高
        explicit = sh_source_common(
            {"ENGINE": "strata", "HOST_ENGINE_PATTERN": "my-own-pattern"},
            'engine_meta; printf "%s" "$(engine_process_pattern)"',
        )
        self.assertEqual(explicit.stdout.strip(), "my-own-pattern")
        # vllm profile 未声明 → 回落 HOST_VLLM_PATTERN
        fallback = sh_source_common(
            {"ENGINE": "vllm"},
            'engine_meta; printf "%s" "$(engine_process_pattern)"',
        )
        self.assertIn("EngineCore", fallback.stdout)
        self.assertIn("vllm.entrypoints.cli.main", fallback.stdout)

    def test_engine_meta_missing_required_field_dies(self) -> None:
        """engine_meta 必填字段校验（评审低-4）：profile 缺 ENGINE_CAPS 等在加载期 die。"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            engines = Path(tmp) / "engines"
            engines.mkdir()
            (engines / "bad-engine.sh").write_text(
                'ENGINE_ID="bad-engine"\nENGINE_LABEL="坏引擎"\n', encoding="utf-8"
            )
            # 用临时 engines 目录替换 engine_profile_path 的指向（函数查找是动态的），
            # 避免"缺必填字段"的路径依赖仓库真实 profile
            script = (
                f'engine_profile_path() {{ printf "%s/engines/%s.sh" {str(tmp)!r} "$1"; }}; '
                'engine_meta'
            )
            result = sh_source_common({"ENGINE": "bad-engine"}, script)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("缺必填字段", result.stderr)
            self.assertIn("ENGINE_CAPS", result.stderr)


class mock_env:
    """临时设置/清除环境变量的上下文管理器（ENGINE 专用）。"""

    def __init__(self, engine: str | None) -> None:
        self.engine = engine
        self._old: str | None = None

    def __enter__(self) -> "mock_env":
        self._old = os.environ.get("ENGINE")
        if self.engine is None:
            os.environ.pop("ENGINE", None)
        else:
            os.environ["ENGINE"] = self.engine
        return self

    def __exit__(self, *exc: object) -> None:
        if self._old is None:
            os.environ.pop("ENGINE", None)
        else:
            os.environ["ENGINE"] = self._old


if __name__ == "__main__":
    unittest.main()
