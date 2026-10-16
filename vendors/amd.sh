#!/usr/bin/env bash
# 厂商 profile：AMD ROCm
#
# 本文件由 scripts/lib/common.sh 与 start_vllm.sh source（也可能在子 shell 中
# 被读取探测元数据），因此**只允许变量赋值与函数定义，不允许任何副作用命令**。

VENDOR_ID="amd"
VENDOR_LABEL="AMD ROCm"

# --- 遥测 ---
VENDOR_SMI_CMD="amd-smi"
VENDOR_SMI_FALLBACKS="rocm-smi"

# --- 分析（probe/vendors.py 的 trace_pack 键）---
VENDOR_TRACE_PACK="rocm"

# --- 就绪判定：/metrics 中出现的指标名（egrep 模式）---
VENDOR_READY_METRIC="^vllm:cache_config_info"

# --- 设备可见性 / 分配器 ---
VENDOR_DEVICE_VIS_ENV="HIP_VISIBLE_DEVICES"
VENDOR_ALLOC_CONF_NAME="PYTORCH_HIP_ALLOC_CONF"
VENDOR_ALLOC_CONF_VALUE="expandable_segments:True"

# --- 容器设备直通 ---
VENDOR_DOCKER_DEVICE_ARGS=(--device /dev/kfd --device /dev/dri --group-add video)

# --- 额外挂载 ---
VENDOR_DOCKER_MOUNTS=()

# --- 容器内启动前 source 的环境脚本 ---
VENDOR_PRELUDE_FILES=()

# --- 固定注入的额外环境变量 K=V ---
VENDOR_EXTRA_ENV=()

# --- 默认基础镜像 ---
VENDOR_DEFAULT_IMAGE="rocm/vllm:latest"

# --- 自动探测元数据 ---
VENDOR_DETECT_SMI="amd-smi rocm-smi"
VENDOR_DETECT_DEVICE="/dev/kfd"

# --- 文档提示 ---
VENDOR_IMAGE_HINT="官方镜像 rocm/vllm:latest（或 ROCm 版 vLLM 自建镜像）"
