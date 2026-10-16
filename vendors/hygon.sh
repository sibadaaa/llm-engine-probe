#!/usr/bin/env bash
# 厂商 profile：Hygon DCU（DTK / ROCm 系）
#
# 本文件由 scripts/lib/common.sh 与 start_vllm.sh source（也可能在子 shell 中
# 被读取探测元数据），因此**只允许变量赋值与函数定义，不允许任何副作用命令**。
#
# 参考：现网 z100-vllm 的启动方式
#   docker run --device /dev/kfd --device /dev/mkfd --device /dev/dri \
#     --group-add video ... z100-s9-snap-s14

VENDOR_ID="hygon"
VENDOR_LABEL="Hygon DCU（DTK / ROCm 系）"

# --- 遥测 ---
VENDOR_SMI_CMD="hy-smi"
VENDOR_SMI_FALLBACKS="rocm-smi"

# --- 分析（probe/vendors.py 的 trace_pack 键）---
VENDOR_TRACE_PACK="hygon"

# --- 就绪判定：/metrics 中出现的指标名（egrep 模式）---
VENDOR_READY_METRIC="^vllm:cache_config_info"

# --- 设备可见性 / 分配器 ---
VENDOR_DEVICE_VIS_ENV="HIP_VISIBLE_DEVICES"
VENDOR_ALLOC_CONF_NAME="PYTORCH_HIP_ALLOC_CONF"
VENDOR_ALLOC_CONF_VALUE="expandable_segments:True"

# --- 容器设备直通（动态：设备节点不存在时自动跳过，避免 docker run 报错）---
vendor_dynamic_docker_args() {
  local dev
  for dev in /dev/kfd /dev/mkfd /dev/dri; do
    [[ -e "$dev" ]] && { printf '%s\n' "--device"; printf '%s\n' "$dev"; }
  done
  printf '%s\n' "--group-add" "video"
}

# --- 额外挂载（start_vllm.sh 只在宿主机路径存在时挂载）---
VENDOR_DOCKER_MOUNTS=(
  "/opt/hyhal:/opt/hyhal:ro"
)

# --- 容器内启动前必须 source 的环境脚本（不 source 时 import torch 会报
#     librocm_smi64.so.2: cannot open shared object file）---
VENDOR_PRELUDE_FILES=()
[[ -n "${DTK_ENV_FILE-/opt/dtk/env.sh}" ]] && VENDOR_PRELUDE_FILES+=("${DTK_ENV_FILE:-/opt/dtk/env.sh}")
[[ -n "${DCU_ENV_FILE-/src/tools/dcu/env.sh}" ]] && VENDOR_PRELUDE_FILES+=("${DCU_ENV_FILE:-/src/tools/dcu/env.sh}")

# --- 固定注入的额外环境变量 K=V ---
VENDOR_EXTRA_ENV=()
[[ -n "${HSA_FORCE_FINE_GRAIN_PCIE:-}" ]] && VENDOR_EXTRA_ENV+=("HSA_FORCE_FINE_GRAIN_PCIE=$HSA_FORCE_FINE_GRAIN_PCIE")

# --- 默认基础镜像（站点镜像请写在 .env 的 IMAGE= 里）---
VENDOR_DEFAULT_IMAGE=""

# --- 自动探测元数据 ---
VENDOR_DETECT_SMI="hy-smi"
VENDOR_DETECT_DEVICE="/dev/mkfd"

# --- 文档提示 ---
VENDOR_IMAGE_HINT="本站点使用已验证的 DTK + vLLM 快照镜像（见 .env 的 IMAGE=）"
