#!/usr/bin/env bash
# 引擎组件 profile：llama.cpp（llama-server，GGUF 本地推理 HTTP 服务）。
# 与 vendors/*.sh 同纪律：只允许变量赋值/函数定义，source 必须无副作用。
# 事实依据：上游 github.com/ggml-org/llama.cpp tools/server/README.md（2026-10 网上核对）；
# ⭐ = 真机核对项（无实测硬件期按上游文档落码，真机时逐项回填）。
# 契约：docs/design/component-interface/设计.md §4.1；两端一致性由 tests/test_engines.py 守卫。

ENGINE_ID=llamacpp
ENGINE_LABEL="llama.cpp（llama-server GGUF 推理）"

# host 模式默认入口：llama-server 二进制（上游预编译或自行 cmake 构建）；GGUF 模型、
# 端口、GPU 层数经 HOST_SERVE_ENTRYPOINT 或 EXTRA_ENGINE_ARGS 传入（推荐完整覆盖：
# HOST_SERVE_ENTRYPOINT="llama-server -m /path/model.gguf --port 8100 --host 0.0.0.0
# -ngl 99 --ctx-size 32768"）。
ENGINE_ENTRYPOINT="llama-server"

# 进程清理正则：就是 llama-server 二进制本身（单进程；无 vLLM 式 worker 子进程）。
ENGINE_PROCESS_PATTERN="llama-server"

# 就绪探测 / 模型列表：/health 在模型加载中返回 503、加载完成 200（探测就绪判定的
# 依据）；模型列表走 OpenAI 兼容 /v1/models。
ENGINE_HEALTH_PATH="/health"
ENGINE_MODELS_PATH="/v1/models"

# /metrics 输出 Prometheus 文本，指标名是 llamacpp: 前缀（非 vLLM 命名）→ prometheus
# flavor；Python 侧 metric_prefix="llamacpp:" 过滤（见 probe/engines.py 的 EngineProfile）。
# ⚠️ 关键坑：/metrics 需启动 flag --metrics（默认 disabled，否则 404）——不是构建开关；
# 推荐启动命令（见 hosts/engine/llamacpp.env）已显式带上。指标全是 Counter/Gauge，
# 无 TTFT/延迟直方图；吞吐是 llamacpp:prompt_tokens_seconds / predicted_tokens_seconds
# 平均速率 gauge（名字像直方图实为 gauge）。
ENGINE_METRICS_FLAVOR="prometheus"

# 无 vLLM serve 子命令语义：跳过全部 vLLM flag 拼装，只用 EXTRA_ENGINE_ARGS/EXTRA_ARGS_FILE
# （值内不能有空格；模型/端口/GPU 层数都从这里或 HOST_SERVE_ENTRYPOINT 传入）。
ENGINE_ARGS_MODE="custom"

# 就绪轮询：/metrics 在带 --metrics 启动时常开；用官方指标表里的常驻指标做就绪信号
# （Counter/Gauge 注册即输出 0 值，首抓即命中）。⭐ 若上游增删指标族再收窄/放宽。
ENGINE_READY_METRIC='^llamacpp:(n_decode_total|requests_processing|prompt_tokens_total)'
ENGINE_METRICS_FETCH_SUFFIX=""

# 版本探测：llama-server --version；新版输出形如
# "version: 0.7.0-dev (build 11719, commit 5aeac37a3)"，旧版为 "version: 4455-b3c4d"。
ENGINE_VERSION_CMD="llama-server --version"

# 常态是外部机器探测，本机没装不拦预检（hostcheck 的 engine 检查项按 warn 处置）。
ENGINE_CHECK_SEVERITY="warn"

# 有官方镜像（ghcr.io/ggml-org/llama.cpp:server），但本仓库 docker/ 不做非 vLLM 引擎
# 适配（design/README §5 红线 3）：留空 = RUNTIME=docker 显式拒绝，docker 支持列为暂缓项。
ENGINE_DOCKER_IMAGE=""

# 能力位（封闭词表见 设计.md §3）：
#   * api.openai-chat/completions、/v1/models、/health（503→200）均有；
#   * api.props 有：llama-server 提供 GET /props（默认生成设置等元数据）；
#   * 无 obs.*：逐 step 日志 / stat-logger / torch profiler 面不存在（C++ 推理栈）；
#   * spec.decode 有（--model-draft 投机解码 flag 面，⭐ 具体 flag 名以所装版本
#     llama-server --help 为准）；
#   * metrics.prometheus（llamacpp: 前缀）有；metrics.vllm-named 无；
#   * runtime.host 有；runtime.docker 无（见上）。
ENGINE_CAPS="api.openai-chat api.openai-completions api.models-list api.health api.props spec.decode metrics.prometheus runtime.host"

# 备注（进 `probe engine` 输出）。
ENGINE_NOTES="单节点 GGUF 推理：/metrics 需 --metrics 启动（默认关）；-ngl GPU 层数、--parallel 槽位、-c 总上下文按 slots 平分；无跨卡张量并行（--tensor-split 仅显存切分）；--jinja 新版默认开（旧版要显式）；思考模型经 --reasoning-format（默认 auto）输出 reasoning_content。"
