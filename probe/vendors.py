"""厂商适配注册表（Python 端）：分析 / 报告 / 遥测使用的厂商元数据。

与 Shell 端的 ``vendors/*.sh`` 一一对应（键集合由 tests/test_vendors.py 校验）：

* Shell 端（``vendors/<id>.sh``）负责**容器启动适配**：设备直通、可见性环境
  变量、启动前 source 的脚本、遥测命令名。
* Python 端（本模块）负责**分析适配**：trace 分类规则包、报告文案、诊断阈值
  覆盖、遥测解析 backend 选择。

新增一家厂商的完整流程见 ``vendors/README.md``。
"""

from __future__ import annotations

import glob
import os
import shutil
from dataclasses import dataclass, field
from typing import Any

# Shell 端自动探测顺序（保持与 common.sh::detect_vendor 一致）
AUTO_DETECT_ORDER: tuple[str, ...] = ("ascend", "nvidia", "hygon", "amd", "intel")


@dataclass(frozen=True)
class VendorProfile:
    """一家厂商在 Python 分析侧的元数据。"""

    id: str
    label: str
    smi_cmd: str = ""
    smi_fallbacks: tuple[str, ...] = ()
    trace_pack: str = "base"
    # 报告里对该厂商「图模式」的称呼（CUDAGraph 统计的文案）
    graph_label: str = "GPU Graph"
    # 设备可见性环境变量名（文档/报告展示用；实际注入在 Shell 端完成）
    device_vis_env: str = ""
    image_hint: str = ""
    notes: str = ""
    # 自动探测元数据
    detect_smi: tuple[str, ...] = ()
    detect_devices: tuple[str, ...] = ()
    # 诊断规则覆盖（report.py::diagnose 的阈值键 → 值），空表示用默认基线
    diagnose_overrides: dict[str, float] = field(default_factory=dict)


VENDOR_REGISTRY: dict[str, VendorProfile] = {
    "generic": VendorProfile(
        id="generic",
        label="通用 / 未识别（无设备直通）",
        trace_pack="base",
        graph_label="GPU Graph",
        notes="未识别厂商；不传设备参数，遥测为空。用于 CI / mock 测试。",
    ),
    "nvidia": VendorProfile(
        id="nvidia",
        label="NVIDIA CUDA",
        smi_cmd="nvidia-smi",
        trace_pack="nvidia",
        graph_label="CUDA Graph",
        device_vis_env="CUDA_VISIBLE_DEVICES",
        image_hint="vllm/vllm-openai:latest（需 nvidia-container-toolkit）",
        detect_smi=("nvidia-smi",),
        detect_devices=("/dev/nvidia0",),
    ),
    "amd": VendorProfile(
        id="amd",
        label="AMD ROCm",
        smi_cmd="amd-smi",
        smi_fallbacks=("rocm-smi",),
        trace_pack="rocm",
        graph_label="HIP Graph",
        device_vis_env="HIP_VISIBLE_DEVICES",
        image_hint="rocm/vllm:latest",
        detect_smi=("amd-smi", "rocm-smi"),
        detect_devices=("/dev/kfd",),
    ),
    "hygon": VendorProfile(
        id="hygon",
        label="Hygon DCU（DTK / ROCm 系）",
        smi_cmd="hy-smi",
        smi_fallbacks=("rocm-smi",),
        trace_pack="hygon",
        graph_label="HIP/CUDA Graph",
        device_vis_env="HIP_VISIBLE_DEVICES",
        image_hint="本站点 DTK + vLLM 快照镜像（见 .env 的 IMAGE=）",
        detect_smi=("hy-smi",),
        detect_devices=("/dev/mkfd",),
    ),
    "intel": VendorProfile(
        id="intel",
        label="Intel GPU（XPU）",
        smi_cmd="xpu-smi",
        trace_pack="intel",
        graph_label="XPU Graph",
        device_vis_env="ZE_AFFINITY_MASK",
        image_hint="intel/vllm:latest",
        detect_smi=("xpu-smi",),
        detect_devices=(),
    ),
    "ascend": VendorProfile(
        id="ascend",
        label="Huawei Ascend NPU",
        smi_cmd="npu-smi",
        trace_pack="ascend",
        graph_label="ACL Graph",
        device_vis_env="ASCEND_RT_VISIBLE_DEVICES",
        image_hint="quay.io/ascend/vllm-ascend:latest",
        notes="vllm-ascend 为独立 fork，kernel 档（torch profiler）为实验性支持。",
        detect_smi=("npu-smi",),
        detect_devices=("/dev/davinci0",),
    ),
}


def list_vendors() -> list[str]:
    """返回注册表里的全部厂商 id（排序）。"""
    return sorted(VENDOR_REGISTRY)


def detect_vendor() -> str:
    """按 AUTO_DETECT_ORDER 探测本机厂商；都不命中返回 ``generic``。"""
    for vid in AUTO_DETECT_ORDER:
        profile = VENDOR_REGISTRY[vid]
        if any(shutil.which(cmd) for cmd in profile.detect_smi):
            return vid
        for pattern in profile.detect_devices:
            if glob.glob(pattern):
                return vid
    return "generic"


def get_vendor(vendor_id: str) -> VendorProfile:
    try:
        return VENDOR_REGISTRY[vendor_id]
    except KeyError:
        raise KeyError(
            f"未知厂商 {vendor_id!r}（可用：{', '.join(list_vendors())}）"
        ) from None


def resolve_vendor(requested: str | None = None) -> VendorProfile:
    """解析厂商：显式参数 > 环境变量 VENDOR > 自动探测 > generic。"""
    if requested and requested != "auto":
        return get_vendor(requested)
    env_value = os.environ.get("VENDOR", "auto").strip() or "auto"
    if env_value != "auto":
        return get_vendor(env_value)
    return get_vendor(detect_vendor())


def smi_candidates(profile: VendorProfile) -> tuple[str, ...]:
    """该厂商实际可用的 SMI 命令列表（按优先级）。"""
    names = [profile.smi_cmd, *profile.smi_fallbacks]
    return tuple(name for name in names if name)


def vendor_info(profile: VendorProfile | None = None, requested: str | None = None) -> dict[str, Any]:
    """汇总厂商的静态元数据 + 本机运行时事实（SMI 是否存在、设备节点是否可见）。"""
    profile = profile or resolve_vendor(requested)
    candidates = smi_candidates(profile)
    available_smi = [name for name in candidates if shutil.which(name)]
    devices_found: dict[str, list[str]] = {}
    for pattern in profile.detect_devices:
        matches = glob.glob(pattern)
        if matches:
            devices_found[pattern] = matches
    if not devices_found and profile.id in VENDOR_REGISTRY:
        # nvidia 的多卡节点不在 detect_devices 里逐个列举，这里补一个常见扫描
        if profile.id == "nvidia":
            extra = glob.glob("/dev/nvidia[0-9]*")
            if extra:
                devices_found["/dev/nvidia[0-9]*"] = extra
        if profile.id == "ascend":
            extra = glob.glob("/dev/davinci[0-9]*")
            if extra:
                devices_found["/dev/davinci[0-9]*"] = extra
    return {
        "id": profile.id,
        "label": profile.label,
        "requested": requested or os.environ.get("VENDOR", "auto"),
        "detected": detect_vendor(),
        "smi_cmd": profile.smi_cmd,
        "smi_candidates": list(candidates),
        "smi_available": available_smi,
        "device_vis_env": profile.device_vis_env,
        "trace_pack": profile.trace_pack,
        "graph_label": profile.graph_label,
        "image_hint": profile.image_hint,
        "notes": profile.notes,
        "devices_found": devices_found,
    }


def format_vendor_info(info: dict[str, Any]) -> str:
    """人类可读的 `probe vendor` 输出。"""
    lines = [
        f"厂商: {info['id']} — {info['label']}",
        f"请求值 VENDOR={info['requested']}  自动探测结果={info['detected']}",
        f"遥测 SMI: {info['smi_cmd'] or '（无）'}"
        + (f"  可用: {', '.join(info['smi_available'])}" if info["smi_available"] else "  ⚠️ 当前 PATH 中未找到"),
        f"设备可见性环境变量: {info['device_vis_env'] or '（不注入）'}",
        f"trace 规则包: {info['trace_pack']}   图模式称谓: {info['graph_label']}",
    ]
    if info["devices_found"]:
        for pattern, found in info["devices_found"].items():
            lines.append(f"设备节点 {pattern}: {', '.join(found)}")
    else:
        lines.append("设备节点: 未发现（无 GPU 或被沙箱隔离属正常）")
    if info["image_hint"]:
        lines.append(f"镜像提示: {info['image_hint']}")
    if info["notes"]:
        lines.append(f"备注: {info['notes']}")
    return "\n".join(lines)
