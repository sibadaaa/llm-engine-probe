#!/usr/bin/env bash
# 厂商 profile 模板：新增一家显卡厂商时复制本文件为 vendors/<厂商id>.sh 并填写。
#
# 命名规则：<厂商id> 用小写字母/数字/下划线，与 probe/vendors.py 注册表键一致。
# 约束：本文件会被 common.sh 在 `set -euo pipefail` 下 source，
#       也可能在子 shell 中被读取探测元数据（bash -c 'source ...'），
#       因此**只允许变量赋值与函数定义，不允许任何副作用命令**。
#
# 填写完运行 `bash tests/run_tests.sh`：其中的「厂商注册表一致性守卫」会检查
# shell 端与 Python 端（probe/vendors.py）的厂商集合是否一致。

# ===== 必填 =====

VENDOR_ID="myvendor"                       # 与文件名一致
VENDOR_LABEL="我的显卡（显示名，报告里用）"

# 遥测 SMI 命令：telemetry.py 会优先用 VENDOR_SMI_CMD，失败后试 VENDOR_SMI_FALLBACKS。
# 若厂商 SMI 输出格式特殊，需要在 probe/telemetry.py 中新增一个 Backend 并注册。
VENDOR_SMI_CMD="myvendor-smi"
VENDOR_SMI_FALLBACKS=""

# trace 规则包：probe/vendors.py 中 VendorProfile.trace_pack 的取值。
# 新增厂商时若需要专属 kernel 分类规则，在 probe/trace_probe.py 的
# CATEGORY_RULE_PACKS 里加一个同名键。
VENDOR_TRACE_PACK="base"

# 就绪判定：服务 /metrics 中出现该指标（egrep 模式）即认为引擎就绪。
VENDOR_READY_METRIC="^vllm:cache_config_info"

# 设备可见性环境变量名（如 CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES /
# ASCEND_RT_VISIBLE_DEVICES / ZE_AFFINITY_MASK）。留空表示不注入。
VENDOR_DEVICE_VIS_ENV=""

# PyTorch 显存分配器配置变量名/值（如 PYTORCH_CUDA_ALLOC_CONF）。
# 留空表示不注入；值可用 .env 的 EXTRA_ENV 覆盖为其它配置。
VENDOR_ALLOC_CONF_NAME=""
VENDOR_ALLOC_CONF_VALUE="expandable_segments:True"

# 容器设备直通参数（docker run 参数数组；每条设备一行）。
# 设备节点可能不存在的场景请改用 vendor_dynamic_docker_args 动态输出。
VENDOR_DOCKER_DEVICE_ARGS=()

# ===== 选填 =====

# 额外挂载，格式 宿主机路径:容器路径[:ro]；start_vllm.sh 只在宿主机路径存在时挂载。
VENDOR_DOCKER_MOUNTS=()

# 容器内启动 vLLM 前必须 source 的环境脚本（按顺序）。
# 如 DTK 的 /opt/dtk/env.sh、Ascend 的 set_env.sh。留空跳过。
VENDOR_PRELUDE_FILES=()

# 固定注入容器的环境变量 K=V 列表（如 HSA_FORCE_FINE_GRAIN_PCIE=1）。
VENDOR_EXTRA_ENV=()

# 默认基础镜像：.env 未设置 IMAGE 时的兜底（可留空——留空时启动会要求显式配置 IMAGE）。
VENDOR_DEFAULT_IMAGE=""

# 自动探测元数据（VENDOR=auto 时使用）：
#   VENDOR_DETECT_SMI    候选 SMI 命令（空格分隔，任一在 PATH 中即命中）
#   VENDOR_DETECT_DEVICE 候选设备节点 glob（compgen -G 判定，命中即算）
# 两者都留空的厂商不会参与自动探测（只能用 VENDOR=<id> 显式选择）。
VENDOR_DETECT_SMI=""
VENDOR_DETECT_DEVICE=""

# 文档/提示文案（启动脚本报错、probe vendor 输出里展示）。
VENDOR_IMAGE_HINT=""

# ===== 可选：动态设备参数 =====
# 定义该函数后，start_vllm.sh 会优先使用它的输出（每行一个 docker 参数），
# 而不是 VENDOR_DOCKER_DEVICE_ARGS。适合设备数量随 VISIBLE_DEVICES 变化的厂商。
#
# vendor_dynamic_docker_args() {
#   local idx
#   local IFS=','
#   for idx in ${VISIBLE_DEVICES:-0}; do
#     printf '%s\n' "--device"
#     printf '%s\n' "/dev/mycard$idx"
#   done
# }
