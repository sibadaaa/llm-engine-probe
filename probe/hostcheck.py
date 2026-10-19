"""物理机（host）模式环境预检：``python3 -m probe hostcheck``。

在物理机（裸机）上直接跑 vLLM 之前，先回答一个问题：**这台机器现在缺什么？**
所有检查都是 best-effort：任何一项出错都只会变成一条 ``fail``/``warn`` 记录，
不会抛栈崩溃（本模块只依赖 Python 标准库）。

用法::

    python3 -m probe hostcheck                    # 人读输出 + 末尾汇总
    python3 -m probe hostcheck --json --outdir runs/latest
    python3 -m probe hostcheck --start            # 启动前快速预检（只跑核心项）
    python3 -m probe hostcheck --quick            # 跳过耗时项
    python3 -m probe hostcheck --skip-vllm-import --skip-gpu

退出码：存在 ``fail`` 级不通过项 → 1，否则 0（``warn``/``info`` 不影响）。

JSON 结构（稳定，消费方按此解析）::

    {
      "kind": "vllm-probe-hostcheck",
      "created_at": "2026-10-18T19:20:00+08:00",
      "runtime": "host",
      "host_python": "/usr/bin/python3",
      "vendor": "hygon",
      "ok": true,
      "summary": {"pass": 10, "warn": 3, "fail": 0},
      "versions": {
        "python": "/usr/bin/python3", "python_version": "3.10.12",
        "torch": "2.6.0", "vllm": "0.18.0", "device_count": 4,
        "cuda_available": true, "hip": null, "npu_available": null,
        "smi": "hy-smi", "gpu_count": 4
      },
      "checks": [
        {"id": "os", "name": "操作系统", "level": "info", "ok": true,
         "detail": "Linux 6.8.0 ...", "hint": ""}
      ]
    }

``level`` 语义：通过 / 跳过 → ``info``；不通过按严重度 → ``fail``（阻塞启动）
或 ``warn``（建议处理但不阻塞）。``summary`` 的三项分别对应：ok 的检查项
（含跳过）、level=warn 的不通过项、level=fail 的不通过项。
"""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

KIND = "vllm-probe-hostcheck"
JSON_NAME = "host_check.json"
DEFAULT_OUTDIR = "runs/latest"
DEFAULT_PORT = 8100
DEFAULT_HOST_PYTHON = "python3"
MIN_PYTHON = (3, 9)
SHM_MIN_BYTES = 8 * 1024 ** 3  # 8 GiB：vLLM/ NCCL 常见共享内存需求

# 检查项固定 id / 中文名（顺序即输出顺序，禁止随意改名——报告与文档按 id 索引）
CHECK_DEFS: tuple[tuple[str, str], ...] = (
    ("os", "操作系统"),
    ("host_python", "宿主 Python 解释器"),
    ("python_version", "Python 版本"),
    ("vllm", "vLLM 可导入"),
    ("torch", "PyTorch / 加速器"),
    ("gpu_smi", "GPU 遥测（SMI）"),
    ("model_path", "模型目录"),
    ("port", "端口可用"),
    ("curl", "curl"),
    ("py_spy", "py-spy 火焰图工具"),
    ("ptrace", "ptrace 权限（yama）"),
    ("ulimit", "ulimit 资源限制"),
    ("shm", "/dev/shm 共享内存"),
    ("prelude", "厂商 prelude 环境脚本"),
)
CHECK_NAMES = dict(CHECK_DEFS)
CHECK_IDS = [cid for cid, _ in CHECK_DEFS]

# --start 只跑 fail 级核心项（其余全部跳过）；--quick 跳过耗时的 import/SMI 采样
CORE_CHECK_IDS = ("os", "host_python", "python_version", "model_path", "port", "curl")

TOKENIZER_HINTS = (
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "spiece.model",
)

_PY_MARKER = "__HOSTCHECK_PYINFO__"

# 在 HOST_PYTHON 下执行：一次子进程同时拿到 Python/torch/vllm/设备信息，减少开销。
# 标记行放在最后；stdout 里的其它噪声（vLLM/torch 的 warning）会被忽略。
_PY_PROBE_TEMPLATE = """
import json, platform, sys

info = {
    "executable": sys.executable,
    "version": platform.python_version(),
    "version_info": list(sys.version_info[:3]),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "torch": None,
    "torch_error": None,
    "cuda_available": None,
    "hip": None,
    "device_count": None,
    "npu_available": None,
    "vllm": None,
    "vllm_error": None,
}
CHECK_TORCH = __CHECK_TORCH__
CHECK_VLLM = __CHECK_VLLM__
if CHECK_TORCH:
    try:
        import torch
        info["torch"] = str(torch.__version__)
        try:
            info["hip"] = getattr(torch.version, "hip", None)
        except Exception:
            pass
        cuda_ok = False
        try:
            cuda_ok = bool(torch.cuda.is_available())
            info["cuda_available"] = cuda_ok
            info["device_count"] = int(torch.cuda.device_count())
        except Exception:
            info["cuda_available"] = False
            info["device_count"] = 0
        if not cuda_ok:
            # Ascend 等非 CUDA/HIP 后端：torch_npu/npu 可用也算有加速器
            try:
                import torch_npu  # noqa: F401  注册 torch.npu
                if bool(torch.npu.is_available()):
                    info["npu_available"] = True
                    info["device_count"] = int(torch.npu.device_count())
            except Exception:
                pass
    except Exception as exc:
        info["torch_error"] = "%s: %s" % (type(exc).__name__, exc)
if CHECK_VLLM:
    try:
        import vllm
        info["vllm"] = str(getattr(vllm, "__version__", "unknown"))
    except Exception as exc:
        info["vllm_error"] = "%s: %s" % (type(exc).__name__, exc)
print("__MARKER__" + json.dumps(info, ensure_ascii=False, default=str))
""".replace("__MARKER__", _PY_MARKER)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _tail(text: str | None, limit: int = 240) -> str:
    """取 stderr/stdout 末尾几行，便于把子进程错误塞进 detail。"""
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    tail = " | ".join(lines[-3:])
    return tail[-limit:] if len(tail) > limit else tail


def _resolve_exe(name: str) -> str | None:
    """把 ``HOST_PYTHON``（裸命令或路径）解析为可执行文件绝对/原样路径。"""
    name = (name or "").strip()
    if not name:
        return None
    if os.sep in name or name.startswith("~"):
        path = os.path.expanduser(name)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
        return None
    return shutil.which(name)


def _human_size(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PiB"


def _extract_py_info(stdout: str) -> dict[str, Any] | None:
    for line in reversed((stdout or "").splitlines()):
        if line.startswith(_PY_MARKER):
            try:
                data = json.loads(line[len(_PY_MARKER):])
            except json.JSONDecodeError:
                continue
            return data if isinstance(data, dict) else None
    return None


# HOST_PRELUDE_FILES 的 token 可能是 `K=V`（先 export，给后面的环境脚本准备变量）
# 或文件路径（存在则 source）。规则与 scripts/lib/common.sh 的 host_prelude_shell 一致。
_PRELUDE_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _prelude_tokens(raw: str) -> list[str]:
    raw = str(raw or "").strip()
    if not raw:
        return []
    try:
        tokens = shlex.split(raw)
    except ValueError:
        tokens = raw.split()
    return [item for item in tokens if item]


def _prelude_shell(raw: str) -> str:
    """把 ``HOST_PRELUDE_FILES`` 转成 shell 片段（K=V → export；文件 → source）。"""
    out: list[str] = []
    for token in _prelude_tokens(raw):
        if _PRELUDE_ASSIGN_RE.match(token):
            name, _, value = token.partition("=")
            out.append(f"export {name}={shlex.quote(value)}; ")
        else:
            path = os.path.expanduser(token)
            quoted = shlex.quote(path)
            out.append(f"if [ -f {quoted} ]; then . {quoted} >/dev/null 2>&1 || true; fi; ")
    return "".join(out)


def _host_pythonpath_default() -> str:
    """与 common.sh 的 host_pythonpath 对齐：SRC_DIR + 项目根 + TRITON_INSTALL_DIR。"""
    parts: list[str] = []
    src = os.environ.get("SRC_DIR", "").strip()
    if src:
        parts.append(src)
    parts.append(str(Path(__file__).resolve().parent.parent))
    triton = os.environ.get("TRITON_INSTALL_DIR", "").strip()
    if triton:
        parts.append(triton)
    return ":".join(parts)


def _probe_command(
    exe: str,
    script: str,
    *,
    prelude: str = "",
    pythonpath: str | None = None,
) -> list[str]:
    """构造 import 探针命令。

    有 prelude 或 PYTHONPATH 时经 bash 注入宿主环境（与 serve_launch.sh 一致），
    否则保持 `[exe, "-c", script]` 直跑（无额外依赖）。
    """
    shell_body = _prelude_shell(prelude)
    if pythonpath is None:
        pythonpath = os.environ.get("HOST_PROBE_PYTHONPATH") or _host_pythonpath_default()
    pythonpath = str(pythonpath or "")
    if not shell_body and not pythonpath:
        return [exe, "-c", script]
    bash = shutil.which("bash")
    if not bash:
        return [exe, "-c", script]
    parts = [shell_body]
    if pythonpath:
        parts.append(
            f"export PYTHONPATH={shlex.quote(pythonpath)}" + '"${PYTHONPATH:+:$PYTHONPATH}"; '
        )
    parts.append(f"exec {shlex.quote(exe)} -c {shlex.quote(script)}")
    return [bash, "-c", "".join(parts)]


def _probe_host_python(
    exe: str,
    *,
    check_torch: bool,
    check_vllm: bool,
    prelude: str = "",
    pythonpath: str | None = None,
    timeout: float = 180.0,
) -> tuple[dict[str, Any] | None, str | None]:
    """在宿主解释器里跑一次汇总探针，返回 (info, error)。

    import vllm/torch 与真实启动（serve_launch.sh）在同一环境里进行：
    先 source ``HOST_PRELUDE_FILES``（支持 K=V），再注入 host PYTHONPATH。
    """
    script = _PY_PROBE_TEMPLATE.replace("__CHECK_TORCH__", repr(bool(check_torch)))
    script = script.replace("__CHECK_VLLM__", repr(bool(check_vllm)))
    command = _probe_command(exe, script, prelude=prelude, pythonpath=pythonpath)
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    info = _extract_py_info(proc.stdout)
    if info is not None:
        return info, None
    detail = _tail(proc.stderr) or _tail(proc.stdout)
    return None, f"子进程未返回结果（exit={proc.returncode}）{('：' + detail) if detail else ''}"


# ---------------------------------------------------------------------------
# 结果收集
# ---------------------------------------------------------------------------


class _Tracker:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.versions: dict[str, Any] = {
            "python": None,
            "python_version": None,
            "torch": None,
            "vllm": None,
            "device_count": None,
            "cuda_available": None,
            "hip": None,
            "npu_available": None,
            "smi": None,
            "gpu_count": None,
            "vendor": None,
        }

    def has(self, check_id: str) -> bool:
        return any(item["id"] == check_id for item in self.checks)

    def add(
        self,
        check_id: str,
        ok: bool,
        detail: str,
        *,
        hint: str = "",
        severity: str | None = None,
    ) -> None:
        level = "info" if ok else (severity or "fail")
        if level not in ("fail", "warn", "info"):
            level = "warn"
        self.checks.append(
            {
                "id": check_id,
                "name": CHECK_NAMES[check_id],
                "level": level,
                "ok": bool(ok),
                "detail": str(detail),
                "hint": str(hint or ""),
            }
        )

    def skip(self, check_id: str, reason: str) -> None:
        self.add(check_id, True, f"已跳过：{reason}")

    def summary(self) -> dict[str, int]:
        passed = sum(1 for item in self.checks if item["ok"])
        warned = sum(1 for item in self.checks if not item["ok"] and item["level"] != "fail")
        failed = sum(1 for item in self.checks if not item["ok"] and item["level"] == "fail")
        return {"pass": passed, "warn": warned, "fail": failed}


@dataclass
class _Context:
    host_python_requested: str
    host_python_resolved: str
    host_python_ok: bool
    model: str
    port: int
    prelude: str
    quick: bool
    start: bool
    skip_vllm_import: bool
    skip_gpu: bool
    check_torch: bool
    check_vllm: bool
    py_info: dict[str, Any] | None
    py_error: str | None
    vendor_profile: Any = None
    vendor_error: str | None = None
    platform_info: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 各检查项
# ---------------------------------------------------------------------------


def _check_os(ctx: _Context, tr: _Tracker) -> None:
    system = platform.system() or "unknown"
    release = platform.release() or "?"
    machine = platform.machine() or "?"
    detail = f"{system} {release}（{machine}）"
    if system.lower() == "linux":
        tr.add("os", True, detail)
    else:
        tr.add(
            "os",
            False,
            detail,
            hint="物理机模式目前只支持 Linux（vLLM 官方支持）；其它系统请用 Docker 模式",
            severity="warn",
        )


def _check_host_python(ctx: _Context, tr: _Tracker) -> None:
    requested = ctx.host_python_requested
    if ctx.host_python_ok:
        detail = requested if ctx.host_python_resolved == requested else f"{requested} → {ctx.host_python_resolved}"
        tr.add("host_python", True, detail)
        return
    tr.add(
        "host_python",
        False,
        f"未找到可执行解释器：{requested}（HOST_PYTHON）",
        hint="安装 Python 3.9+，或用 HOST_PYTHON=/path/to/python（.env 或环境变量）指向 vLLM 所在的解释器",
        severity="fail",
    )


def _check_python_version(ctx: _Context, tr: _Tracker) -> None:
    if not ctx.host_python_ok:
        tr.skip("python_version", "HOST_PYTHON 不可用")
        return
    if not ctx.py_info:
        tr.add(
            "python_version",
            False,
            f"无法获取 Python 版本：{ctx.py_error or '未知原因'}",
            hint="手动排查：$HOST_PYTHON -c 'import sys; print(sys.version)'",
            severity="warn",
        )
        return
    version = str(ctx.py_info.get("version") or "")
    version_tuple = ctx.py_info.get("version_info")
    if not isinstance(version_tuple, (list, tuple)) or len(version_tuple) < 2:
        parts = version.split(".")
        try:
            version_tuple = [int(parts[0]), int(parts[1])]
        except (IndexError, ValueError):
            tr.add(
                "python_version",
                False,
                f"无法解析 Python 版本：{version or '空'}",
                severity="warn",
            )
            return
    major, minor = int(version_tuple[0]), int(version_tuple[1])
    if (major, minor) >= MIN_PYTHON:
        tr.add("python_version", True, f"Python {version}（≥ {MIN_PYTHON[0]}.{MIN_PYTHON[1]}）")
    else:
        tr.add(
            "python_version",
            False,
            f"Python {version} 低于 {MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
            hint=f"vLLM 需要 Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+；用 HOST_PYTHON 指向新版解释器（如 venv）",
            severity="warn",
        )


def _check_vllm(ctx: _Context, tr: _Tracker) -> None:
    if ctx.skip_vllm_import:
        tr.skip("vllm", "--skip-vllm-import")
        return
    if ctx.quick:
        tr.skip("vllm", "快速模式不 import vllm（耗时）")
        return
    if not ctx.host_python_ok:
        tr.skip("vllm", "HOST_PYTHON 不可用")
        return
    info = ctx.py_info or {}
    version = info.get("vllm")
    if version:
        tr.versions["vllm"] = version
        tr.add("vllm", True, f"vLLM {version}")
        return
    error = info.get("vllm_error") or ctx.py_error or "未知原因"
    tr.add(
        "vllm",
        False,
        f"在 HOST_PYTHON 下 import vllm 失败：{error}",
        hint=(
            "在 HOST_PYTHON 对应环境安装 vLLM（如 $HOST_PYTHON -m pip install vllm）；"
            "源码安装时确认 PYTHONPATH/SRC_DIR 与 start_vllm.sh 一致"
        ),
        severity="fail",
    )


def _check_torch(ctx: _Context, tr: _Tracker) -> None:
    if not ctx.check_torch:
        tr.skip("torch", "快速模式不 import torch（耗时）")
        return
    if not ctx.host_python_ok:
        tr.skip("torch", "HOST_PYTHON 不可用")
        return
    info = ctx.py_info or {}
    version = info.get("torch")
    if not version:
        error = info.get("torch_error") or ctx.py_error or "未知原因"
        tr.add(
            "torch",
            False,
            f"在 HOST_PYTHON 下 import torch 失败：{error}",
            hint="安装与 GPU 驱动/厂商 DTK 匹配的 PyTorch（NVIDIA/AMD/Hygon/Ascend 各有专用 wheel）",
            severity="fail",
        )
        return
    cuda = info.get("cuda_available")
    npu = info.get("npu_available")
    hip = info.get("hip")
    count = info.get("device_count")
    tr.versions.update(
        {
            "torch": version,
            "device_count": count,
            "cuda_available": bool(cuda) if cuda is not None else None,
            "hip": hip,
            "npu_available": bool(npu) if npu is not None else None,
        }
    )
    if cuda or npu:
        backend = "CUDA" if cuda else "NPU"
        extra = f" · HIP {hip}" if hip else ""
        tr.add("torch", True, f"torch {version} · {backend} 可用 · device_count={count}{extra}")
        return
    tr.add(
        "torch",
        False,
        f"torch {version} 已安装，但 CUDA/HIP/NPU 不可用（device_count={count}）",
        hint=(
            "检查 GPU 驱动与设备权限（/dev/nvidia*、/dev/kfd、/dev/davinci*）、"
            "设备可见性环境变量（CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES/ASCEND_RT_VISIBLE_DEVICES）"
        ),
        severity="warn",
    )


def _check_gpu_smi(ctx: _Context, tr: _Tracker) -> None:
    if ctx.skip_gpu:
        tr.skip("gpu_smi", "--skip-gpu")
        return
    if ctx.quick:
        tr.skip("gpu_smi", "快速模式不采样 SMI（耗时）")
        return
    profile = ctx.vendor_profile
    if profile is None:
        tr.add(
            "gpu_smi",
            False,
            f"厂商无法识别，跳过 SMI 采样：{ctx.vendor_error or '未知原因'}",
            hint="检查 .env 的 VENDOR（可用值见 `python3 -m probe vendor`）",
            severity="warn",
        )
        return
    try:
        from .telemetry import sample_once

        record = sample_once(profile)
    except Exception as exc:  # 遥测采样失败只降级为 warn
        tr.add(
            "gpu_smi",
            False,
            f"SMI 采样异常：{type(exc).__name__}: {exc}",
            hint=f"确认厂商 SMI 命令可用：{profile.smi_cmd or '（未知）'}",
            severity="warn",
        )
        return
    cards = record.get("cards") or []
    tr.versions["smi"] = record.get("smi")
    tr.versions["gpu_count"] = len(cards)
    if record.get("ok"):
        name = record.get("smi") or "厂商 SMI"
        vram = record.get("vram_perc_sum")
        vram_txt = f"，显存占用合计 {vram}%" if vram is not None else ""
        tr.add("gpu_smi", True, f"{name} 采样成功：{len(cards)} 卡{vram_txt}")
        return
    if profile.id == "generic":
        tr.add(
            "gpu_smi",
            True,
            "厂商为 generic / 未检测到 GPU SMI，跳过遥测（无卡环境属正常）",
        )
        return
    tr.add(
        "gpu_smi",
        False,
        f"SMI 采样失败：{record.get('error') or '无数据'}",
        hint=f"确认 {profile.smi_cmd or '厂商 SMI'} 已在 PATH 且当前用户有权限访问设备",
        severity="warn",
    )


def _check_model_path(ctx: _Context, tr: _Tracker) -> None:
    model = (ctx.model or "").strip()
    if not model:
        tr.add(
            "model_path",
            False,
            "未指定 MODEL（--model 或环境变量/ .env 的 MODEL）",
            hint="在 .env 中设置 MODEL=/path/to/model（目录内需有 config.json）",
            severity="fail",
        )
        return
    path = Path(os.path.expanduser(model))
    fails: list[str] = []
    notes: list[str] = []
    tokenizer_missing = False
    if not path.exists():
        fails.append(f"MODEL 不存在：{path}")
    elif not path.is_dir():
        fails.append(f"MODEL 不是目录：{path}")
    else:
        notes.append(str(path))
        if not os.access(path, os.R_OK | os.X_OK):
            fails.append(f"模型目录不可读：{path}")
        config = path / "config.json"
        if not config.is_file():
            fails.append(f"缺少 config.json：{config}")
        else:
            notes.append("config.json ✓")
        found = [name for name in TOKENIZER_HINTS if (path / name).is_file()]
        if found:
            notes.append(f"tokenizer 文件 ✓（{', '.join(found[:3])}）")
        else:
            tokenizer_missing = True
    if fails:
        tr.add(
            "model_path",
            False,
            "；".join(fails + notes),
            hint="确认 .env 的 MODEL 指向完整模型目录（含 config.json / tokenizer），且当前用户可读",
            severity="fail",
        )
        return
    if tokenizer_missing:
        # tokenizer 缺失只 warn，不阻塞启动
        tr.add(
            "model_path",
            False,
            "；".join(notes + ["未找到常见 tokenizer 文件（可能内嵌在 config 或运行时自动下载）"]),
            hint="确认模型目录完整（tokenizer_config.json/tokenizer.json 等）；离线环境必须齐全",
            severity="warn",
        )
        return
    tr.add("model_path", True, "；".join(notes))


def _check_port(ctx: _Context, tr: _Tracker) -> None:
    port = ctx.port
    if not isinstance(port, int) or not (1 <= port <= 65535):
        tr.add(
            "port",
            False,
            f"端口非法：{port!r}（应为 1..65535）",
            hint="设置 .env 的 PORT（如 PORT=8100）",
            severity="fail",
        )
        return
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError as exc:
        tr.add(
            "port",
            False,
            f"端口 {port} 不可绑定：{exc}",
            hint=f"换端口（PORT=其它值）或先停掉占用进程：ss -ltnp | grep :{port}；host 模式可 scripts/stop_vllm.sh",
            severity="fail",
        )
        return
    finally:
        try:
            sock.close()
        except OSError:
            pass
    tr.add("port", True, f"端口 {port} 可绑定")


def _check_curl(ctx: _Context, tr: _Tracker) -> None:
    path = shutil.which("curl")
    if path:
        tr.add("curl", True, path)
    else:
        tr.add(
            "curl",
            False,
            "PATH 中未找到 curl",
            hint="安装 curl（apt install curl / yum install curl）；等待就绪与 HTTP 探测都依赖它",
            severity="fail",
        )


def _check_py_spy(ctx: _Context, tr: _Tracker) -> None:
    binary = (os.environ.get("PYSPY_BIN") or "py-spy").strip()
    resolved = _resolve_exe(binary)
    if resolved:
        tr.add("py_spy", True, f"{binary} → {resolved}")
        return
    tr.add(
        "py_spy",
        False,
        f"未找到 {binary}（PYSPY_BIN）",
        hint=f"安装：$HOST_PYTHON -m pip install py-spy，或设置 PYSPY_BIN=/path/to/py-spy",
        severity="warn",
    )


def _check_ptrace(ctx: _Context, tr: _Tracker) -> None:
    scope_file = Path("/proc/sys/kernel/yama/ptrace_scope")
    if not scope_file.exists():
        tr.add("ptrace", True, "无 yama ptrace_scope（非 yama 内核），py-spy 一般可直接使用")
        return
    try:
        raw = scope_file.read_text(encoding="utf-8").strip()
        scope = int(raw)
    except (OSError, ValueError) as exc:
        tr.add(
            "ptrace",
            False,
            f"无法读取 {scope_file}：{exc}",
            hint="py-spy 需要 ptrace 权限；可尝试 sudo py-spy 或检查 /proc 挂载",
            severity="warn",
        )
        return
    if scope <= 1:
        tr.add("ptrace", True, f"yama ptrace_scope={scope}")
    else:
        tr.add(
            "ptrace",
            False,
            f"yama ptrace_scope={scope}（>1），普通用户无法 attach 进程",
            hint="sudo sysctl kernel.yama.ptrace_scope=1，或 sudo -E py-spy ...；否则火焰图/py-spy dump 会失败",
            severity="warn",
        )


def _check_ulimit(ctx: _Context, tr: _Tracker) -> None:
    try:
        import resource
    except ImportError:  # pragma: no cover - 非 Linux 大概率走这里
        tr.add("ulimit", True, "platform 不支持 resource 模块，跳过")
        return
    try:
        mem_soft, mem_hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    except (OSError, ValueError) as exc:
        tr.add("ulimit", False, f"无法读取 RLIMIT_MEMLOCK：{exc}", severity="warn")
        return
    unlimited = resource.RLIM_INFINITY
    mem_txt = "unlimited" if mem_soft == unlimited else _human_size(float(mem_soft))
    hard_txt = "unlimited" if mem_hard == unlimited else _human_size(float(mem_hard))
    try:
        stack_soft, _ = resource.getrlimit(resource.RLIMIT_STACK)
        stack_txt = "unlimited" if stack_soft == unlimited else _human_size(float(stack_soft))
    except (OSError, ValueError):
        stack_txt = "未知"
    if mem_soft == unlimited or mem_hard == unlimited:
        tr.add("ulimit", True, f"memlock={mem_txt}（hard={hard_txt}） · stack={stack_txt}")
    else:
        tr.add(
            "ulimit",
            False,
            f"memlock 未放开：soft={mem_txt} hard={hard_txt} · stack={stack_txt}",
            hint="ulimit -l unlimited（写入 /etc/security/limits.conf；RDMA/NCCL 与大页内存常需要）",
            severity="warn",
        )


def _check_shm(ctx: _Context, tr: _Tracker) -> None:
    shm = Path("/dev/shm")
    if not shm.exists():
        tr.add(
            "shm",
            False,
            "/dev/shm 不存在",
            hint="物理机模式建议挂载足够大的 /dev/shm（vLLM/NCCL 共享内存）",
            severity="warn",
        )
        return
    try:
        usage = shutil.disk_usage(str(shm))
    except OSError as exc:
        tr.add("shm", False, f"无法读取 /dev/shm 容量：{exc}", severity="warn")
        return
    free_txt = _human_size(float(usage.free))
    total_txt = _human_size(float(usage.total))
    if usage.free >= SHM_MIN_BYTES:
        tr.add("shm", True, f"可用 {free_txt} / 共 {total_txt}（≥ 8 GiB）")
    else:
        tr.add(
            "shm",
            False,
            f"/dev/shm 可用 {free_txt} / 共 {total_txt}（< 8 GiB）",
            hint="扩容 /dev/shm（mount -o remount,size=32G /dev/shm）或降低并发/上下文长度",
            severity="warn",
        )


def _check_prelude(ctx: _Context, tr: _Tracker) -> None:
    raw = (ctx.prelude or "").strip()
    if not raw:
        tr.add("prelude", True, "未配置 HOST_PRELUDE_FILES（无厂商 prelude）")
        return
    tokens = _prelude_tokens(raw)
    files: list[str] = []
    assignments: list[str] = []
    for token in tokens:
        if _PRELUDE_ASSIGN_RE.match(token):
            assignments.append(token)
        else:
            files.append(token)
    missing = [item for item in files if not Path(os.path.expanduser(item)).is_file()]
    if missing:
        tr.add(
            "prelude",
            False,
            f"缺失 {len(missing)} 个 prelude 文件：{', '.join(missing)}",
            hint=(
                "确认厂商环境脚本路径（HOST_PRELUDE_FILES），或清空该变量表示不需要 source；"
                "需要先设置变量时可用 K=V 写法，如 DTK_ROOT=/opt/dtk2604 /opt/dtk/env.sh"
            ),
            severity="warn",
        )
        return
    details: list[str] = []
    if files:
        details.append(f"{len(files)} 个 prelude 文件：{', '.join(files)}")
    if assignments:
        details.append(f"{len(assignments)} 个环境变量：{', '.join(assignments)}")
    tr.add(
        "prelude",
        True,
        "；".join(details) + "（import 探针已按该环境执行）",
    )


_CHECKS: tuple[tuple[str, Callable[[_Context, _Tracker], None]], ...] = (
    ("os", _check_os),
    ("host_python", _check_host_python),
    ("python_version", _check_python_version),
    ("vllm", _check_vllm),
    ("torch", _check_torch),
    ("gpu_smi", _check_gpu_smi),
    ("model_path", _check_model_path),
    ("port", _check_port),
    ("curl", _check_curl),
    ("py_spy", _check_py_spy),
    ("ptrace", _check_ptrace),
    ("ulimit", _check_ulimit),
    ("shm", _check_shm),
    ("prelude", _check_prelude),
)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def _resolve_vendor() -> tuple[Any, str | None]:
    from .vendors import resolve_vendor

    try:
        return resolve_vendor(), None
    except Exception as exc:  # 非法 VENDOR 等：不影响其它检查
        return None, f"{type(exc).__name__}: {exc}"


def collect_hostcheck(
    *,
    host_python: str | None = None,
    model: str | None = None,
    port: int | None = None,
    prelude: str | None = None,
    quick: bool = False,
    start: bool = False,
    skip_vllm_import: bool = False,
    skip_gpu: bool = False,
) -> dict[str, Any]:
    """执行全部检查并返回 host_check payload（不写文件、不打印）。"""
    requested_py = (
        host_python if host_python is not None else os.environ.get("HOST_PYTHON") or DEFAULT_HOST_PYTHON
    )
    requested_py = str(requested_py).strip() or DEFAULT_HOST_PYTHON
    resolved_py = _resolve_exe(requested_py)
    host_python_ok = resolved_py is not None
    py_for_probe = resolved_py or requested_py

    if model is None:
        model = os.environ.get("MODEL", "")
    model = str(model or "").strip()
    if port is None:
        port = _env_int("PORT", DEFAULT_PORT)
    try:
        port = int(port)
    except (TypeError, ValueError):
        port = DEFAULT_PORT
    if prelude is None:
        prelude = os.environ.get("HOST_PRELUDE_FILES", "")
    prelude = str(prelude or "")
    # import 探针与真实启动使用同一套 PYTHONPATH（shell 侧显式传入优先）
    probe_pythonpath = os.environ.get("HOST_PROBE_PYTHONPATH") or _host_pythonpath_default()

    quick = bool(quick or start)
    check_torch = not quick
    check_vllm = (not quick) and (not skip_vllm_import)

    tracker = _Tracker()
    tracker.versions["python"] = resolved_py or requested_py
    profile, vendor_error = _resolve_vendor()
    if profile is not None:
        tracker.versions["vendor"] = profile.id

    py_info: dict[str, Any] | None = None
    py_error: str | None = None
    if host_python_ok:
        try:
            py_info, py_error = _probe_host_python(
                py_for_probe,
                check_torch=check_torch,
                check_vllm=check_vllm,
                prelude=prelude,
                pythonpath=probe_pythonpath,
            )
        except Exception as exc:  # 探测子进程本身的异常也只降级为 warn
            py_info, py_error = None, f"{type(exc).__name__}: {exc}"
    if py_info:
        tracker.versions.update(
            {
                "python_version": py_info.get("version"),
                "torch": py_info.get("torch"),
                "vllm": py_info.get("vllm"),
                "device_count": py_info.get("device_count"),
                "cuda_available": (
                    bool(py_info["cuda_available"])
                    if py_info.get("cuda_available") is not None
                    else None
                ),
                "hip": py_info.get("hip"),
                "npu_available": (
                    bool(py_info["npu_available"])
                    if py_info.get("npu_available") is not None
                    else None
                ),
            }
        )

    ctx = _Context(
        host_python_requested=requested_py,
        host_python_resolved=resolved_py or requested_py,
        host_python_ok=host_python_ok,
        model=model,
        port=port,
        prelude=prelude,
        quick=quick,
        start=bool(start),
        skip_vllm_import=bool(skip_vllm_import),
        skip_gpu=bool(skip_gpu),
        check_torch=check_torch,
        check_vllm=check_vllm,
        py_info=py_info,
        py_error=py_error,
        vendor_profile=profile,
        vendor_error=vendor_error,
        platform_info={
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    )

    skipped_by_start = set(CHECK_IDS) - set(CORE_CHECK_IDS) if start else set()
    for check_id, fn in _CHECKS:
        if check_id in skipped_by_start:
            tracker.skip(check_id, "--start 只跑 fail 级核心项")
            continue
        try:
            fn(ctx, tracker)
        except Exception as exc:  # 任何检查异常都转成 warn，绝不崩栈
            if not tracker.has(check_id):
                tracker.add(
                    check_id,
                    False,
                    f"检查异常：{type(exc).__name__}: {exc}",
                    severity="warn",
                )

    summary = tracker.summary()
    return {
        "kind": KIND,
        "created_at": _iso_now(),
        "runtime": "host",
        "host_python": ctx.host_python_resolved,
        "vendor": profile.id if profile is not None else (os.environ.get("VENDOR") or "unknown"),
        "ok": summary["fail"] == 0,
        "summary": summary,
        "versions": tracker.versions,
        "checks": tracker.checks,
    }


def format_human(payload: dict[str, Any], *, json_path: str | Path | None = None) -> str:
    """把 payload 渲染成人类可读文本（含末尾汇总）。"""
    lines: list[str] = [
        "物理机（host）模式环境预检",
        f"  HOST_PYTHON={payload.get('host_python')}    VENDOR={payload.get('vendor')}    "
        f"时间={payload.get('created_at')}",
        "─" * 62,
    ]
    for item in payload.get("checks", []):
        if not item.get("ok"):
            mark = "⚠️ " if item.get("level") == "warn" else "❌"
        elif "已跳过" in str(item.get("detail", "")):
            mark = "ℹ️ "
        else:
            mark = "✅"
        lines.append(f"{mark} [{item.get('id')}] {item.get('name')}：{item.get('detail')}")
        hint = item.get("hint")
        if hint and not item.get("ok"):
            lines.append(f"   ↳ 处理建议：{hint}")
    summary = payload.get("summary", {})
    lines.append("─" * 62)
    lines.append(
        f"汇总：通过 {summary.get('pass', 0)} · 警告 {summary.get('warn', 0)} · 失败 {summary.get('fail', 0)}"
    )
    if payload.get("ok"):
        lines.append("✅ 预检通过，可以启动 vLLM（warn 项建议处理，不影响启动）")
    else:
        lines.append(
            f"❌ 预检未通过：{summary.get('fail', 0)} 项 fail 级问题；"
            "处理后可加 --skip-check 跳过预检（不推荐）"
        )
    if json_path is not None:
        lines.append(f"JSON 已写入 {json_path}")
    return "\n".join(lines)


def write_json(payload: dict[str, Any], outdir: str | Path) -> Path:
    """把 payload 写入 ``<outdir>/host_check.json``（目录不存在则创建）。"""
    directory = Path(os.path.expanduser(str(outdir)))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / JSON_NAME
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def run_hostcheck(
    *,
    host_python: str | None = None,
    model: str | None = None,
    port: int | None = None,
    outdir: str | None = None,
    prelude: str | None = None,
    quick: bool = False,
    start: bool = False,
    skip_vllm_import: bool = False,
    skip_gpu: bool = False,
    write_json_file: bool = False,
) -> int:
    """CLI 入口：收集 → 打印 →（可选）写 JSON → 返回退出码。"""
    payload = collect_hostcheck(
        host_python=host_python,
        model=model,
        port=port,
        prelude=prelude,
        quick=quick,
        start=start,
        skip_vllm_import=skip_vllm_import,
        skip_gpu=skip_gpu,
    )
    json_path: Path | None = None
    write_failed = False
    if write_json_file:
        directory = outdir or os.environ.get("PROBE_OUTDIR") or DEFAULT_OUTDIR
        try:
            json_path = write_json(payload, directory)
        except OSError as exc:
            write_failed = True
            print(f"[hostcheck] 写 JSON 失败：{directory}（{type(exc).__name__}: {exc}）", file=sys.stderr)
    print(format_human(payload, json_path=json_path))
    if write_failed:
        return 1
    return 0 if payload.get("ok") else 1


__all__ = [
    "CHECK_DEFS",
    "CHECK_IDS",
    "CHECK_NAMES",
    "DEFAULT_OUTDIR",
    "JSON_NAME",
    "KIND",
    "collect_hostcheck",
    "format_human",
    "run_hostcheck",
    "write_json",
]
