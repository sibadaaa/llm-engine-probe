#!/usr/bin/env bash
# 引擎组件 profile：Ollama（本地模型管理 + 推理服务）。
# 与 vendors/*.sh 同纪律：只允许变量赋值/函数定义，source 必须无副作用。
# 事实依据：上游 github.com/ollama/ollama / docs.ollama.com（2026-10 网上核对；
# 主线无 /metrics——指标 PR #16998/#18508 尚未合并）；⭐ = 真机核对项。
# 契约：docs/design/component-interface/设计.md §4.1；两端一致性由 tests/test_engines.py 守卫。

ENGINE_ID=ollama
ENGINE_LABEL="Ollama（本地模型管理/推理服务）"

# host 模式默认入口：`ollama serve`；监听地址/端口靠 OLLAMA_HOST 环境变量
# （形如 127.0.0.1:8100，经模板的 EXTRA_ENV 传入），命令行本身无 --port flag。
ENGINE_ENTRYPOINT="ollama serve"

# 进程清理正则：ollama serve 主进程 + 每个已加载模型的 llama-server runner 子进程
# （v0.40 起 GGUF 推理由 lib/ollama/llama-server 子进程承载，路径含 "ollama" 字样，
# 一个 "ollama" 都能盖住；⭐ 真机 ps -ef 核对后再决定是否细化）。
ENGINE_PROCESS_PATTERN="ollama"

# 就绪探测 / 模型列表：Ollama 无 /health；根路径 GET / 返回 200 "Ollama is running"，
# 用 health_path=/ 承担就绪判定（api.health cap 的语义就是「health_path 指定路径可达」）；
# 模型列表走 OpenAI 兼容 /v1/models（原生 /api/tags 不用于探测）。
ENGINE_HEALTH_PATH="/"
ENGINE_MODELS_PATH="/v1/models"

# 主线无 /metrics（见文件头）：无指标面 → metrics_flavor=none，探针跳过指标采集与
# 就绪指标轮询（wait_engine_ready 的 metrics 段按 caps 跳过；suite 对 404 已容错）。
ENGINE_METRICS_FLAVOR="none"

# 无 vLLM serve 子命令语义：跳过全部 vLLM flag 拼装，只用 EXTRA_ENGINE_ARGS/EXTRA_ARGS_FILE
# （值内不能有空格；OLLAMA_* 环境用 EXTRA_ENV：EXTRA_ENV="OLLAMA_HOST=127.0.0.1:8100"）。
ENGINE_ARGS_MODE="custom"

# 无指标面：就绪判定只走 health_path 的 200 轮询，无需指标正则。
ENGINE_READY_METRIC=""
ENGINE_METRICS_FETCH_SUFFIX=""

# 版本探测：ollama --version；连上服务时输出 "ollama version is 0.40.8"（取自
# GET /api/version），连不上时首行是 "Warning: could not connect..."（版本取自
# 客户端 Warning 行，hostcheck 首行解析在该场景下会拿到 Warning 文案——warn 处置可接受）。
ENGINE_VERSION_CMD="ollama --version"

# 常态是外部机器探测（本机装了 Ollama App 也不拦预检；hostcheck 按 warn 处置）。
ENGINE_CHECK_SEVERITY="warn"

# 有官方镜像（ollama/ollama），但本仓库 docker/ 不做非 vLLM 引擎适配（design/README
# §5 红线 3）：留空 = RUNTIME=docker 显式拒绝，docker 支持列为暂缓项。
ENGINE_DOCKER_IMAGE=""

# 能力位（封闭词表见 设计.md §3）：
#   * api.openai-chat/completions、/v1/models 均有（OpenAI 兼容层）；
#   * api.health 有：health_path=/ 的根路径 200 轮询（无 /health 端点，见上）；
#   * 无 api.props（无该端点）；
#   * 无 metrics.*：主线无 /metrics（指标能力待上游合并后按 prometheus flavor 回填）；
#   * 无 obs.*：无逐 step/trace/图模式观测面；无 spec.decode（无投机解码配置面）；
#   * runtime.host 有；runtime.docker 无（见上）。
ENGINE_CAPS="api.openai-chat api.openai-completions api.models-list api.health runtime.host"

# 备注（进 `probe engine` 输出）。
ENGINE_NOTES="模型以 tag 名寻址（需先 ollama pull）；无 /metrics 指标面（探测只走 HTTP 时延/吞吐）；并发默认 1（OLLAMA_NUM_PARALLEL）；keep_alive 默认 5m 到期卸载；思考内容在 delta.reasoning 字段。"
