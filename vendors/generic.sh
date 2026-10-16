#!/usr/bin/env bash
# 厂商 profile：通用 / 未识别（无设备直通）
#
# 用途：
#   * CI / 无 GPU 机器上跑 mock 端到端测试（VENDOR=generic）；
#   * 自动探测失败时的兜底（不传设备参数，显存/功耗遥测为空）。
#
# 本文件由 scripts/lib/common.sh 与 start_vllm.sh source（也可能在子 shell 中
# 被读取探测元数据），因此**只允许变量赋值与函数定义，不允许任何副作用命令**。

VENDOR_ID="generic"
VENDOR_LABEL="通用 / 未识别（无设备直通）"

# --- 遥测（无 SMI；telemetry 会尝试常见工具，失败则只记录空数据）---
VENDOR_SMI_CMD=""
VENDOR_SMI_FALLBACKS=""

# --- 分析（probe/vendors.py 的 trace_pack 键；base=只启用通用规则）---
VENDOR_TRACE_PACK="base"

# --- 就绪判定：/metrics 中出现的指标名（egrep 模式）---
VENDOR_READY_METRIC="^vllm:cache_config_info"

# --- 设备可见性 / 分配器 ---
VENDOR_DEVICE_VIS_ENV=""
VENDOR_ALLOC_CONF_NAME=""
VENDOR_ALLOC_CONF_VALUE=""

# --- 容器设备直通（无）---
VENDOR_DOCKER_DEVICE_ARGS=()

# --- 额外挂载 ---
VENDOR_DOCKER_MOUNTS=()

# --- 容器内启动前 source 的环境脚本 ---
VENDOR_PRELUDE_FILES=()

# --- 固定注入的额外环境变量 K=V ---
VENDOR_EXTRA_ENV=()

# --- 默认基础镜像 ---
VENDOR_DEFAULT_IMAGE=""

# --- 自动探测元数据（generic 永不参与自动探测）---
VENDOR_DETECT_SMI=""
VENDOR_DETECT_DEVICE=""

# --- 文档提示 ---
VENDOR_IMAGE_HINT="不传设备直通参数；仅用于无 GPU 环境 / mock 测试"
