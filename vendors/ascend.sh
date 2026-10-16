#!/usr/bin/env bash
# 厂商 profile：Huawei Ascend NPU（vllm-ascend）
#
# 本文件由 scripts/lib/common.sh 与 start_vllm.sh source（也可能在子 shell 中
# 被读取探测元数据），因此**只允许变量赋值与函数定义，不允许任何副作用命令**。
#
# 说明：vllm-ascend 是独立 fork，日志/profiler 端点与 CUDA/ROCm 版可能有差异，
# 本工具箱对 Ascend 的 kernel 档（torch profiler）标注为「实验性」。

VENDOR_ID="ascend"
VENDOR_LABEL="Huawei Ascend NPU"

# --- 遥测 ---
VENDOR_SMI_CMD="npu-smi"
VENDOR_SMI_FALLBACKS=""

# --- 分析（probe/vendors.py 的 trace_pack 键）---
VENDOR_TRACE_PACK="ascend"

# --- 就绪判定：/metrics 中出现的指标名（egrep 模式）---
VENDOR_READY_METRIC="^vllm:cache_config_info"

# --- 设备可见性 / 分配器 ---
VENDOR_DEVICE_VIS_ENV="ASCEND_RT_VISIBLE_DEVICES"
VENDOR_ALLOC_CONF_NAME=""
VENDOR_ALLOC_CONF_VALUE=""

# --- 容器设备直通（davinci 节点按 VISIBLE_DEVICES 动态映射）---
VENDOR_DOCKER_DEVICE_ARGS=()
vendor_dynamic_docker_args() {
  local dev idx
  # 管理/通信节点（数量固定，存在才挂）
  for dev in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
    [[ -e "$dev" ]] && { printf '%s\n' "--device"; printf '%s\n' "$dev"; }
  done
  # 计算卡：VISIBLE_DEVICES 里的编号 → /dev/davinciN
  local vis="${VISIBLE_DEVICES:-0}"
  local IFS=','
  for idx in $vis; do
    [[ "$idx" =~ ^[0-9]+$ ]] || continue
    dev="/dev/davinci$idx"
    [[ -e "$dev" ]] && { printf '%s\n' "--device"; printf '%s\n' "$dev"; }
  done
  unset IFS
}

# --- 额外挂载（驱动/工具链，存在才挂；start_vllm.sh 会检查宿主机路径）---
VENDOR_DOCKER_MOUNTS=(
  "/usr/local/dcmi:/usr/local/dcmi:ro"
  "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro"
  "/usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro"
)

# --- 容器内启动前 source 的环境脚本 ---
VENDOR_PRELUDE_FILES=(
  "/usr/local/Ascend/ascend-toolkit/set_env.sh"
)

# --- 固定注入的额外环境变量 K=V ---
VENDOR_EXTRA_ENV=()

# --- 默认基础镜像 ---
VENDOR_DEFAULT_IMAGE="quay.io/ascend/vllm-ascend:latest"

# --- 自动探测元数据 ---
VENDOR_DETECT_SMI="npu-smi"
VENDOR_DETECT_DEVICE="/dev/davinci0"

# --- 文档提示 ---
VENDOR_IMAGE_HINT="官方镜像 quay.io/ascend/vllm-ascend:latest；需宿主机已装 Ascend 驱动"
