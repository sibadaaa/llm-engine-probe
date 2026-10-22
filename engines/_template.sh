#!/usr/bin/env bash
# ============================================================================
# 引擎组件 profile 骨架：复制本文件为 engines/<id>.sh 后填写。
#
# 引擎组件 = **提供服务进程的软件**（替换 serve 入口，如 vLLM/Strata）；
# 只追加 flag/env 的增强型组件（DeepEP 等）不走本接口，继续用 hosts/<类目>/*.env。
#
# 与 vendors/*.sh 同纪律：profile 会在子 shell 与当前 shell 中被 source
# （engine_meta / tests/test_engines.py），因此**只允许变量赋值与函数定义，
# 不允许任何副作用命令**（不要执行命令、不要 cd、不要写文件）。
#
# 契约：docs/design/component-interface/设计.md §4.1；能力键是封闭词表
# （同文 §3 / probe/engines.py 的 CAP_VOCAB），词表外的键会被测试拦截。
# 两端一致性：ENGINE_* ↔ probe/engines.py 的 EngineProfile 逐字段对应，
# 由 tests/test_engines.py 强制；接入流程见 docs/design/README.md §4。
# ============================================================================

# 必填：必须与文件名一致（<id>.sh）
ENGINE_ID="<id>"
# 必填：报告/CLI 展示名
ENGINE_LABEL=""

# host 模式默认 serve 入口（可被 HOST_SERVE_ENTRYPOINT 覆盖）；空 = vLLM 内置默认
ENGINE_ENTRYPOINT=""
# 进程清理/查找正则（stop/status/pyspy 共用）；空 = 回落 HOST_VLLM_PATTERN。
# 优先级链：HOST_ENGINE_PATTERN > ENGINE_PROCESS_PATTERN > HOST_VLLM_PATTERN。
ENGINE_PROCESS_PATTERN=""

# 就绪探测 / 模型列表路径
ENGINE_HEALTH_PATH="/health"
ENGINE_MODELS_PATH="/v1/models"

# 必填：/metrics 形态：vllm-named | prometheus | none
ENGINE_METRICS_FLAVOR="vllm-named"
# 必填：参数拼装模式：vllm-cli（走现有 vLLM flag 拼装）| custom（只用 EXTRA_ENGINE_ARGS）
ENGINE_ARGS_MODE="vllm-cli"

# 就绪等待轮询 /metrics 的正则；空 = 回落 VENDOR_READY_METRIC（厂商默认）
ENGINE_READY_METRIC=""
# 抓 /metrics 追加的查询串（如 "?format=prometheus"，用于默认回 JSON 的引擎）
ENGINE_METRICS_FETCH_SUFFIX=""
# hostcheck 版本探测命令；空 = 走 HTTP 元数据 / import 探测
ENGINE_VERSION_CMD=""
# 引擎预检缺失时的处置：warn | fail
ENGINE_CHECK_SEVERITY="fail"
# 镜像引用；空 = 无 docker 支持（RUNTIME=docker 下显式拒绝）
ENGINE_DOCKER_IMAGE=""

# 必填：能力位（空格分隔；只能用 设计.md §3 词表里的键）
ENGINE_CAPS=""

# 一句话备注（进 `probe engine` 输出）
ENGINE_NOTES=""
