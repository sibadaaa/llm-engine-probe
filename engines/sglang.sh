#!/usr/bin/env bash
# 引擎组件 profile：SGLang（sgl-project 高性能 OpenAI 兼容推理引擎）。
# 与 vendors/*.sh 同纪律：只允许变量赋值/函数定义，source 必须无副作用。
# 事实依据：上游 docs.sglang.io / github.com/sgl-project/sglang（2026-10 网上核对）；
# ⭐ = 真机核对项（无实测硬件期按上游文档落码，真机时逐项回填）。
# 契约：docs/design/component-interface/设计.md §4.1；两端一致性由 tests/test_engines.py 守卫。

ENGINE_ID=sglang
ENGINE_LABEL="SGLang（高性能 OpenAI 兼容推理引擎）"

# host 模式默认入口：上游官方新推荐 `sglang serve`（CLI 子命令，flag 一致），这里保留
# 兼容面最广的模块入口 `python3 -m sglang.launch_server`（上游仍支持并在运行时提示换新）；
# 模型/端口经 HOST_SERVE_ENTRYPOINT 或 EXTRA_ENGINE_ARGS 传入
# （推荐完整覆盖：HOST_SERVE_ENTRYPOINT="python3 -m sglang.launch_server
#   --model-path /path/to/model --port 8100 --tp-size 4"）。
ENGINE_ENTRYPOINT="python3 -m sglang.launch_server"

# 进程清理正则：launch_server 主进程与 scheduler/detokenizer 子进程命令行均含
# "sglang" 字样（setproctitle 形如 sglang::scheduler_TP0、sglang::detokenizer、
# sglang::data_parallel_controller）。上游官方 killall 正则是
# `sglang::|sglang\.launch_server|sglang serve|sglang\.srt|...`——"sglang" 单词覆盖全部，
# ⭐ 真机 ps -ef 核对后如需排除误伤（如 sglang.bench）再细化。
ENGINE_PROCESS_PATTERN="sglang"

# 就绪探测 / 模型列表：/health 加载期 503、就绪 200（默认零开销，不做真实生成；
# /health_generate 才是真跑一次生成的深度就绪，探活不要用它）；模型列表走 OpenAI
# 兼容 /v1/models。
ENGINE_HEALTH_PATH="/health"
ENGINE_MODELS_PATH="/v1/models"

# /metrics 输出 Prometheus 文本（plain GET 即可，无需查询串），指标名是 sglang: 前缀
# （非 vLLM 命名）→ prometheus flavor；Python 侧 metric_prefix="sglang:" 过滤
# （见 probe/engines.py 的 EngineProfile）。
# ⚠️ 关键坑：/metrics 需启动 flag --enable-metrics（默认 False，不加时基本为空）——
# 推荐启动命令（见 hosts/engine/sglang.env）已显式带上。主标签 model_name；
# dp_attention 场景指标默认挂 TP0，分 rank 需 --enable-metrics-for-all-schedulers。
ENGINE_METRICS_FLAVOR="prometheus"

# 无 vLLM serve 子命令语义：跳过全部 vLLM flag 拼装，只用 EXTRA_ENGINE_ARGS/EXTRA_ARGS_FILE
# （值内不能有空格；模型/端口/TP 都从这里或 HOST_SERVE_ENTRYPOINT 传入）。
# 注意显存旋钮是 --mem-fraction-static（无 vLLM 式 --gpu-memory-utilization）。
ENGINE_ARGS_MODE="custom"

# 就绪轮询：HTTP 200 后再等调度计数器出现；带 --enable-metrics 启动时注册即输出 0 值，
# 首抓即命中。⭐ 指标名以真机 GET /metrics 实测为准（上游 metrics_collector 实名核对：
# running 计数器是复数 num_running_reqs）。
ENGINE_READY_METRIC='^sglang:num_running_reqs'
ENGINE_METRICS_FETCH_SUFFIX=""

# 版本探测：新版包有 `sglang version` 子命令（输出 "sglang version: X.Y.Z"，首行即取；
# 无 `sglang --version`）。旧版包可能没有 CLI——命令缺失按 warn 处置。注意版本命令两个
# 执行点都按空白切分、不支持引号/管道，`|| python3 -c ...` 回退链写法不可用。
ENGINE_VERSION_CMD="sglang version"

# 常态是外部机器探测，本机没装不拦预检（hostcheck 的 engine 检查项按 warn 处置）。
ENGINE_CHECK_SEVERITY="warn"

# 有官方镜像（lmsysorg/sglang），但本仓库 docker/ 不做非 vLLM 引擎适配（design/README
# §5 红线 3）：留空 = RUNTIME=docker 显式拒绝，docker 支持列为暂缓项。
ENGINE_DOCKER_IMAGE=""

# 能力位（封闭词表见 设计.md §3）：
#   * api.openai-chat/completions、/v1/models、/health 均有；
#   * 无 api.props：sglang 的 /get_server_info 不是 vLLM /props 语义，hostcheck 不采元数据；
#   * 无 obs.*：逐 step 日志 / stat-logger / torch profiler 面未接入本工具
#     （sglang 有 /start_profile torch profiler，属后续增强，⭐ 待真机核对 trace 产物协议）；
#   * spec.decode 有（投机解码 flag 面：EAGLE/MTP/N-gram，⭐ 具体 flag 名以所装版本
#     `python3 -m sglang.launch_server --help` 为准）；
#   * metrics.prometheus（sglang: 前缀）有；metrics.vllm-named 无；
#   * runtime.host 有；runtime.docker 无（见上）。
ENGINE_CAPS="api.openai-chat api.openai-completions api.models-list api.health spec.decode metrics.prometheus runtime.host"

# 备注（进 `probe engine` 输出）。
ENGINE_NOTES="RadixAttention 前缀缓存 + PD 分离/EP；/metrics 需 --enable-metrics 启动（默认关）；无逐 step 日志/torch profiler 接入面；投机解码（EAGLE/NEXTN/NGRAM）flag 面可用；指标为 sglang: 前缀 Prometheus 文本；显存旋钮是 --mem-fraction-static。"
