#!/usr/bin/env bash
# 引擎组件 profile：Strata（消费级 PC 的 MoE 专家分层推理引擎）。
# 与 vendors/*.sh 同纪律：只允许变量赋值/函数定义，source 必须无副作用。
# 事实依据：docs/design/strata-engine/事实核对.md；⭐ = 真机核对项
# （核对清单见 docs/design/strata-engine/施工图.md §4，有 Strata 机器时逐项回填）。
# 契约：docs/design/component-interface/设计.md §4.1；两端一致性由 tests/test_engines.py 守卫。

ENGINE_ID=strata
ENGINE_LABEL="Strata（MoE 专家分层推理引擎）"

# host 模式默认入口：无交互起服务。模型/端口/上下文优先经 strata-<model>.json 固化；
# ⭐ strata launch 的 --port 支持待核，未确认前端口经 HOST_SERVE_ENTRYPOINT 覆盖传入
# （如 HOST_SERVE_ENTRYPOINT="strata launch --model q2_0 --yes --no-tui --port ${PORT}"）。
ENGINE_ENTRYPOINT="strata launch --yes --no-tui"

# 进程清理正则：⭐ 占位（App/Terminal/server.py/engine 二进制的真实进程名待 ps -ef 核对）。
ENGINE_PROCESS_PATTERN="strata|server\.py"

ENGINE_HEALTH_PATH="/health"
ENGINE_MODELS_PATH="/v1/models"

# /metrics 以 vLLM 指标名输出 Prometheus 文本（metrics.vllm-named cap 的依据），
# 解析层直接复用；但缺失 5 个生命周期直方图与 cache_config_info（探针按缺项容错）。
ENGINE_METRICS_FLAVOR="vllm-named"

# 无 vLLM serve 子命令语义：跳过全部 vLLM flag 拼装，只用 EXTRA_ENGINE_ARGS/EXTRA_ARGS_FILE。
ENGINE_ARGS_MODE="custom"

# 就绪轮询：⭐ 无 cache_config_info，换 running/waiting 计数器；真机核对实际指标名。
ENGINE_READY_METRIC='^vllm:num_requests_(running|waiting)'
# /metrics 默认返回 JSON（监控页数据），必须显式要 Prometheus 文本（事实核对 §5 关键坑）。
ENGINE_METRICS_FETCH_SUFFIX="?format=prometheus"

# ⭐ 版本命令未确认；备选 strata doctor / GET /props 的引擎版本字段。
ENGINE_VERSION_CMD="strata --version"

# 常态是外部机器探测，本机没装不拦预检（hostcheck 的 engine 检查项按 warn 处置）。
ENGINE_CHECK_SEVERITY="warn"

# 无官方镜像：RUNTIME=docker 显式拒绝（红线 3，docker 支持列为暂缓项）。
ENGINE_DOCKER_IMAGE=""

# 能力位（封闭词表见 设计.md §3）：
#   * 无 api.openai-completions（上游无 /v1/completions）→ PROBE_ENDPOINT=completions 被拒；
#   * 无 obs.* 四项、无 runtime.docker → F1 的 caps 门禁自动完成降级/拒绝；
#   * spec.decode 有（MTP --spec N --mtp）但 obs.spec-metrics 无：/metrics 级 draft/accepted
#     计数器在 metrics.vllm-named 的通用解析里自然出现；归因小节依赖 per-request 数据，
#     等 F3 确认 Strata 计数器可支撑归因后再声明（事实核对 §7）。
ENGINE_CAPS="api.openai-chat api.models-list api.health api.props spec.decode metrics.vllm-named metrics.prometheus runtime.host"

# 备注（进 `probe engine` 输出）。
ENGINE_NOTES="MoE 专家分层(VRAM/内存/NVMe)+MTP 投机；默认单请求串行；无逐 step 日志/torch profiler/图模式；思考计入 max_tokens。"
