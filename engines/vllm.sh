#!/usr/bin/env bash
# 引擎组件 profile：vLLM（默认引擎）
#
# 本文件由 scripts/lib/common.sh 的 engine_meta source，也可能在子 shell 中
# 被 tests/test_engines.py 读取一致性比对，因此**只允许变量赋值与函数定义，
# 不允许任何副作用命令**（与 vendors/*.sh 同纪律）。
#
# 契约：docs/design/component-interface/设计.md §4.1（字段语义）与 §3（能力键词表）。
# 两端一致性：ENGINE_* ↔ probe/engines.py 的 EngineProfile，由 tests/test_engines.py 强制。

ENGINE_ID="vllm"
ENGINE_LABEL="vLLM（默认引擎）"

# --- host 模式默认 serve 入口 ---
# 空 = 走 host_serve_entrypoint() 的 vLLM 内置默认
# （`$HOST_PYTHON -c 'import torch; from vllm.entrypoints.cli.main import main; main()'`），
# 可被 HOST_SERVE_ENTRYPOINT 覆盖。
ENGINE_ENTRYPOINT=""

# --- 进程清理/查找正则（stop/status/pyspy 共用）---
# 空 = 回落 HOST_VLLM_PATTERN（common.sh 默认值，旧键保留兼容）；
# 优先级链：HOST_ENGINE_PATTERN > ENGINE_PROCESS_PATTERN > HOST_VLLM_PATTERN。
ENGINE_PROCESS_PATTERN=""

# --- HTTP 端点路径 ---
ENGINE_HEALTH_PATH="/health"
ENGINE_MODELS_PATH="/v1/models"

# --- /metrics 形态与参数拼装模式 ---
ENGINE_METRICS_FLAVOR="vllm-named"        # /metrics 以 vLLM 指标名输出 Prometheus 文本
ENGINE_ARGS_MODE="vllm-cli"               # 走 start_vllm.sh 现有 vLLM flag 拼装（唯一 vllm-cli 引擎）

# --- 就绪等待 / 抓取 / 版本 ---
# 空 = 回落 VENDOR_READY_METRIC（厂商默认 ^vllm:cache_config_info）
ENGINE_READY_METRIC=""
ENGINE_METRICS_FETCH_SUFFIX=""
# 空 = hostcheck 走 import vllm 探测（既有行为）
ENGINE_VERSION_CMD=""
ENGINE_CHECK_SEVERITY="fail"

# --- docker 支持 ---
# vLLM 的镜像选择走现有 IMAGE / VENDOR_DEFAULT_IMAGE 流程，此字段留空不参与。
ENGINE_DOCKER_IMAGE=""

# --- 能力位（空格分隔；封闭词表见 设计.md §3 / probe/engines.py 的 CAP_VOCAB）---
ENGINE_CAPS="api.openai-chat api.openai-completions api.models-list api.health api.props metrics.vllm-named metrics.prometheus obs.step-log obs.per-request-metrics obs.spec-metrics obs.trace-pytorch obs.graph-mode spec.decode runtime.docker runtime.host"

# --- 备注（进 `probe engine` 输出）---
ENGINE_NOTES=""
