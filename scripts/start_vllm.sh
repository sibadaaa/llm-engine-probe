#!/usr/bin/env bash
# 启动一个"可观测的" vLLM 实例。
#
# 运行时（RUNTIME，默认 docker）：
#   docker : 在容器里起（原有行为；设备直通/挂载来自 vendors/<id>.sh）
#   host   : 物理机（裸机）直接起，不经过容器（见 scripts/lib/common.sh 的 host_*）
#
# 多厂商：容器设备直通 / 可见性环境变量 / 启动前 source 的环境脚本全部来自
# vendors/<id>.sh（VENDOR=auto 时自动探测，见 vendors/README.md）。
# 观测档位与厂商、运行时均无关，见下。
#
# 多引擎：ENGINE=vllm（默认）| strata | …；引擎组件 = 提供服务进程的软件，
# 注册表在 engines/<id>.sh（shell）↔ probe/engines.py（Python），差异走能力位
# （caps）协商而非 ENGINE 分支——契约见 docs/design/component-interface/设计.md，
# 新引擎接入三步见 docs/design/README.md §4。
#
# 用法：
#   scripts/start_vllm.sh                        # 按 .env 的 RUNTIME/PROBE_LEVEL 启动
#   scripts/start_vllm.sh --host                 # 物理机裸跑（等价 RUNTIME=host）
#   scripts/start_vllm.sh --docker               # 强制 docker（覆盖 .env 的 RUNTIME）
#   scripts/start_vllm.sh --runtime host         # 同上；别名 container/bare/native 可用
#   scripts/start_vllm.sh --level kernel         # 临时切到算子级观测档位
#   scripts/start_vllm.sh --dry-run              # 只打印将执行的命令，不启动、不预检
#   scripts/start_vllm.sh --exec                 # 复用已有容器（docker exec -d；host 不支持）
#   scripts/start_vllm.sh --skip-check           # host 模式跳过 scripts/host_check.sh 预检
#   VENDOR=nvidia scripts/start_vllm.sh          # 显式指定厂商（覆盖自动探测）
#
# 观测档位 PROBE_LEVEL：
#   off     不加任何观测开关（对照基线）
#   basic   只依赖默认的周期 stats 日志 + /metrics
#   steps   逐 iteration 明细 + 逐请求指标 + 逐请求 metrics 字段      ← 默认
#   deep    steps + KV block 驻留 + 图调度统计 + MFU + 逐步投机解码
#   kernel  deep + torch profiler（配合 /start_profile）+ 逐层 NVTX
#
# 所有开关与取值都会写进 <RUN_DIR>/run_meta.json 与 <RUN_DIR>/cmd.txt，保证可复现。
# host 模式额外产物：<RUN_DIR>/serve.pid、serve.pgid、serve_launch.sh（见 common.sh）。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

LEVEL=""
DRY_RUN=0
FOREGROUND=0
TAG="serve"
DO_WAIT=1
SKIP_CHECK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --level) LEVEL="${2:?}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --foreground) FOREGROUND=1; shift ;;
    --tag) TAG="${2:?}"; shift 2 ;;
    --no-wait) DO_WAIT=0; shift ;;
    --run-dir) RUN_DIR="${2:?}"; shift 2 ;;
    --exec) MODE=exec; shift ;;
    --host) RUNTIME=host; shift ;;
    --docker) RUNTIME=docker; shift ;;
    --runtime) RUNTIME="${2:?}"; shift 2 ;;
    --skip-check) SKIP_CHECK=1; shift ;;
    -h|--help) sed -n '2,35p' "$0"; exit 0 ;;
    *) die "未知参数：$1（-h 查看用法）" ;;
  esac
done
[[ -n "$LEVEL" ]] || LEVEL="$PROBE_LEVEL"

# 运行时归一化（--host/--docker/--runtime 可覆盖 .env）+ 校验宿主依赖
RUNTIME="$(runtime_normalize "${RUNTIME:-docker}")"
export RUNTIME
runtime_require

case "$LEVEL" in off|basic|steps|deep|kernel) ;; *) die "PROBE_LEVEL 非法：$LEVEL（off|basic|steps|deep|kernel）" ;; esac

if runtime_is_host; then
  [[ "$MODE" == "run" ]] || die "RUNTIME=host 不支持 MODE=$MODE（MODE=exec 只用于复用已有容器；host 模式请去掉 --exec）"
else
  case "$MODE" in run|exec) ;; *) die "MODE 非法：$MODE（run|exec）" ;; esac
fi

# ---------------------------------------------------------------------------
# 引擎组件：加载 profile + caps 门禁（docs/design/component-interface/设计.md §7）
# ENGINE 未设置 / =vllm 时全部检查直通，拼装结果与历史版本逐字节一致。
# ---------------------------------------------------------------------------
engine_meta
if ! engine_has_cap runtime.docker && runtime_is_docker; then
  die "ENGINE=$ENGINE 该引擎暂无 docker 支持（engines/$ENGINE.sh 未声明 runtime.docker；docker 支持为暂缓项，见 docs/design/README.md §5 红线 3）；请用 RUNTIME=host 或外部只读探测（deploy/external-probe.sh）"
fi
if [[ -n "$DRAFT_MODEL" || -n "$SPEC_METHOD" ]] && ! engine_has_cap spec.decode; then
  die "ENGINE=$ENGINE 不支持投机解码（无 spec.decode 能力）：请清空 DRAFT_MODEL/SPEC_METHOD，或改用该引擎自己的投机解码配置面"
fi
if [[ "$LEVEL" == "steps" || "$LEVEL" == "deep" ]] && ! engine_has_cap obs.step-log; then
  warn "ENGINE=$ENGINE 不支持逐 step 日志观测（无 obs.step-log 能力）：PROBE_LEVEL=$LEVEL 降级为 basic"
  LEVEL="basic"
fi
if [[ "$LEVEL" == "kernel" ]] && ! engine_has_cap obs.trace-pytorch; then
  die "ENGINE=$ENGINE 不支持 torch profiler kernel 档（无 obs.trace-pytorch 能力，降级没有意义）：请用 deep/steps/basic 档"
fi
if [[ "$PROBE_PLUGIN" == "1" ]] && ! engine_has_cap obs.step-log; then
  warn "ENGINE=$ENGINE 不支持逐 step 日志/插件观测（无 obs.step-log 能力）：PROBE_PLUGIN 置 0"
  PROBE_PLUGIN=0
fi
if [[ "$PROBE_ENDPOINT" == "completions" ]] && ! engine_has_cap api.openai-completions; then
  die "ENGINE=$ENGINE 不提供 /v1/completions（无 api.openai-completions 能力）：PROBE_ENDPOINT=completions 无法使用，请改 PROBE_ENDPOINT=chat"
fi

[[ -n "$MODEL" ]] || die "MODEL 未设置：请在 .env 里指定模型目录（参考 .env.example）"
if runtime_is_docker; then
  [[ -n "$IMAGE" ]] || die "IMAGE 未设置且厂商（$VENDOR）没有默认镜像：请在 .env 里指定 IMAGE。提示：$VENDOR_IMAGE_HINT"
fi
if [[ -z "$SERVED_NAME" ]]; then SERVED_NAME="$(basename "$MODEL")"; fi
if [[ -n "$DRAFT_MODEL" && -z "$SPEC_METHOD" ]]; then
  die "配置了 DRAFT_MODEL 但没有 SPEC_METHOD（如 dflash / mtp / eagle）：请在 .env 里设置"
fi

RUN_DIR="$(new_run_dir "$TAG")"
LOG_FILE="$RUN_DIR/serve.log"
TRACE_DIR="$RUN_DIR/traces"
mkdir -p "$TRACE_DIR" "$RUN_DIR"

# PORT 是宿主机端口（探针访问用）；INTERNAL_PORT 是容器内 vLLM 监听端口。
# MODE=run 时二者默认相同；配合 docker-compose.yml 时通常 INTERNAL_PORT=8000。
# host 模式没有容器，vLLM 直接监听 PORT。
if runtime_is_host; then
  INTERNAL_PORT="$PORT"
else
  : "${INTERNAL_PORT:=$PORT}"
fi

# ---------------------------------------------------------------------------
# 1+2. 服务参数与观测开关（vLLM CLI 形态）
#   整段 flag/env 拼装是 vLLM 引擎的 `serve` 子命令形态（ENGINE_ARGS_MODE=vllm-cli，
#   由引擎 profile 声明）；custom 模式的引擎（无 vLLM CLI 语义）跳过全部 vLLM
#   flag 块，只用 EXTRA_ENGINE_ARGS/EXTRA_ARGS_FILE 追加与 MODEL/SERVED_NAME/PORT
#   等记录性字段（docs/design/component-interface/设计.md §7.2）。
#   ENGINE=vllm 时本门禁恒真，拼装结果与历史版本逐字节一致。
# ---------------------------------------------------------------------------
ARGS=()
SPEC_DECODE_ON=0

# ENV_KVS 是唯一的 K=V 来源：docker 侧再派生成 -e，host 侧写进 serve_launch.sh。
ENV_KVS=()
add_env() { ENV_KVS+=("$1"); }

case "$LEVEL" in
  off)   LEVEL_DESC="关闭全部可选观测开关（对照基线）" ;;
  basic) LEVEL_DESC="默认周期 stats（VLLM_LOG_STATS_INTERVAL=${LOG_STATS_INTERVAL}s）+ /metrics" ;;
  steps) LEVEL_DESC="逐 iteration 明细 + 逐请求日志 + 逐请求 metrics 字段" ;;
  deep)  LEVEL_DESC="steps + KV block 驻留 + 图调度统计 + MFU + 逐步投机解码指标" ;;
  kernel) LEVEL_DESC="deep + torch profiler + 逐层 NVTX（会自动关闭图模式）" ;;
esac

if [[ "$ENGINE_ARGS_MODE" == "vllm-cli" ]]; then
ARGS=(serve "$MODEL"
  --served-model-name "$SERVED_NAME"
  --host "$HOST" --port "$INTERNAL_PORT"
  --dtype "$DTYPE"
  --tensor-parallel-size "$TP"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
)
[[ "$EAGER" == "1" ]] && ARGS+=(--enforce-eager)

# 投机解码（可选；spec.decode 门禁与开头的能力检查双保险，防模板链中途覆盖）
if [[ -n "$DRAFT_MODEL" ]] && engine_has_cap spec.decode; then
  SPEC_DECODE_ON=1
  SPEC="{\"model\":\"$DRAFT_MODEL\",\"method\":\"$SPEC_METHOD\",\"num_speculative_tokens\":$SPEC_TOKENS"
  [[ -n "${SPEC_DTYPE:-}" ]] && SPEC="$SPEC,\"dtype\":\"$SPEC_DTYPE\""
  [[ -n "${SPEC_KV_DTYPE:-}" ]] && SPEC="$SPEC,\"kv_cache_dtype\":\"$SPEC_KV_DTYPE\""
  [[ -n "$SPEC_EXTRA_JSON" ]] && SPEC="$SPEC,$SPEC_EXTRA_JSON"
  SPEC="$SPEC}"
  ARGS+=(--speculative-config "$SPEC")
fi

if [[ "$LEVEL" != "off" ]]; then
  add_env "VLLM_LOGGING_LEVEL=$VLLM_LOGGING_LEVEL"
  add_env "VLLM_LOG_STATS_INTERVAL=$LOG_STATS_INTERVAL"
fi

if [[ "$LEVEL" == "steps" || "$LEVEL" == "deep" || "$LEVEL" == "kernel" ]]; then
  ARGS+=(
    --enable-logging-iteration-details
    --enable-log-requests
    --enable-per-request-metrics
    --enable-prompt-tokens-details
    --enable-force-include-usage
    --max-log-len 2048
    --show-hidden-metrics-for-version 0.5
  )
  add_env "VLLM_LOG_BATCHSIZE_INTERVAL=$LOG_BATCHSIZE_INTERVAL"
fi

if [[ "$LEVEL" == "deep" || "$LEVEL" == "kernel" ]]; then
  ARGS+=(
    --kv-cache-metrics
    --kv-cache-metrics-sample "${KV_CACHE_METRICS_SAMPLE:-0.05}"
    --cudagraph-metrics
    --enable-mfu-metrics
  )
  if [[ "$SPEC_DECODE_ON" == "1" ]]; then
    ARGS+=(--per-request-spec-decode-metrics "${PER_REQUEST_SPEC_METRICS:-detailed}")
  else
    warn "未配置 DRAFT_MODEL，跳过 --per-request-spec-decode-metrics（无投机解码时无数据）"
  fi
fi

if [[ "$LEVEL" == "kernel" ]]; then
  # torch profiler 落盘目录：docker 用容器内挂载点 /traces；host 直接写 run 目录（绝对路径）
  if runtime_is_host; then
    PROFILER_DIR="$TRACE_DIR"
  else
    PROFILER_DIR="/traces"
  fi
  PROFILER_JSON=$(printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":%s,"torch_profiler_with_flops":%s,"torch_profiler_record_shapes":%s,"torch_profiler_with_memory":%s,"torch_profiler_dump_cuda_time_total":true}' \
    "$PROFILER_DIR" "$PROFILER_WITH_STACK" "$PROFILER_WITH_FLOPS" "$PROFILER_RECORD_SHAPES" "$PROFILER_WITH_MEMORY")
  ARGS+=(--profiler-config "$PROFILER_JSON")
  # 把引擎内部的 record_function 范围打进 trace（CPU 侧阶段分布）
  add_env "VLLM_CUSTOM_SCOPES_FOR_PROFILING=$ENABLE_PROFILING_SCOPES"
  if [[ "$ENABLE_NVTX_SCOPES" == "1" ]]; then
    add_env "VLLM_NVTX_SCOPES_FOR_PROFILING=1"
    warn "已改用 NVTX 范围；注意 vLLM 内部是 if/elif，VLLM_CUSTOM_SCOPES_FOR_PROFILING 优先"
  fi
  if [[ "${ENABLE_LAYERWISE_NVTX:-1}" == "1" ]]; then
    ARGS+=(--enable-layerwise-nvtx-tracing)
    if [[ "$EAGER" != "1" ]]; then
      warn "--enable-layerwise-nvtx-tracing 与 CUDA/HIP Graph 不兼容，自动追加 --enforce-eager"
      ARGS+=(--enforce-eager)
      EAGER=1
    fi
  fi
  if [[ "$VENDOR" == "ascend" ]]; then
    warn "Ascend 的 torch profiler 支持为实验性（vllm-ascend 独立 fork），若端点不存在请用 deep/steps 档"
  fi
  log "profiler 已启用：启动后 POST /start_profile 开始采集，POST /stop_profile 落盘到 $TRACE_DIR"
fi
fi  # end ENGINE_ARGS_MODE=vllm-cli

# ---------------------------------------------------------------------------
# 可选：stat logger 插件（plugin_probe/）——把每个 engine step 的结构化统计写成 JSONL
# 它是"探测所有步骤"最深的一层：直接拿到引擎内部的 SchedulerStats / IterationStats
# （逐 step 批次、逐请求 queue/prefill/decode/e2e/TPOT、KV 驱逐、按位置接受数…）。
# ---------------------------------------------------------------------------
PLUGIN_SETUP=""
PROBE_LOG_PATH=""
if [[ "$PROBE_PLUGIN" == "1" ]] && engine_has_cap obs.step-log; then
  if [[ "$LEVEL" == "off" || "$LEVEL" == "basic" ]]; then
    warn "PROBE_PLUGIN=1 但档位是 $LEVEL：逐 step 明细需要 --enable-logging-iteration-details，建议用 steps/deep/kernel"
  fi
  if runtime_is_host; then
    # host：插件装到 run 目录（见下方安装步骤），JSONL 直接写宿主机 run 目录
    PROBE_LOG_PATH="$RUN_DIR/$PROBE_LOG_NAME"
  elif [[ "$MODE" == "exec" ]]; then
    PROBE_LOG_PATH="${EXEC_LOG_DIR:-${GAME_DIR:-/tmp}/vllm-probe}/$PROBE_LOG_NAME"
  else
    PROBE_LOG_PATH="/logs/$PROBE_LOG_NAME"
  fi
  add_env "VLLM_PROBE_LOG=$PROBE_LOG_PATH"
  if runtime_is_host; then
    log "stat logger 插件：开（host；JSONL → $PROBE_LOG_PATH）"
  else
    # 关键：在容器内 /tmp 复制一份再安装，避免容器里的 root 在宿主机工作区留下
    # root 属主的 egg-info / __pycache__（实测踩过这个坑）。
    PLUGIN_SETUP="python3 -c 'import vllm_probe_stat_logger' 2>/dev/null || "
    PLUGIN_SETUP+="{ rm -rf /tmp/vllm-probe-plugin && cp -r /opt/vllm-probe/plugin_probe /tmp/vllm-probe-plugin && "
    PLUGIN_SETUP+="pip install -q --no-deps --no-build-isolation -e /tmp/vllm-probe-plugin >/dev/null 2>&1 "
    PLUGIN_SETUP+="&& echo '[probe] stat logger 插件已安装(JSONL)'; } || "
    PLUGIN_SETUP+="echo '[probe] 警告: stat logger 插件安装失败（不影响推理，仅缺 JSONL）' >&2; "
    log "stat logger 插件：开（JSONL → $PROBE_LOG_PATH）"
  fi
else
  log "stat logger 插件：关（PROBE_PLUGIN=1 可开启；提供最细的逐 step/逐请求结构化数据）"
fi

# 用户自定义参数：
#   EXTRA_VLLM_ARGS     vLLM 引擎专用追加键（按空白切分）；custom 模式引擎忽略并打 ⚠️
#                       （键名含 vllm，误用大概率是模板叠错——设计.md §4.3）
#   EXTRA_ARGS_FILE     每行一个参数（引擎无关；JSON 参数请用后者）
#   EXTRA_ENGINE_ARGS   引擎无关追加键（末尾追加；对 vLLM 与 EXTRA_VLLM_ARGS 等效合并）
if [[ -n "$EXTRA_VLLM_ARGS" ]]; then
  if [[ "$ENGINE_ARGS_MODE" == "vllm-cli" ]]; then
    # shellcheck disable=SC2206
    ARGS+=($EXTRA_VLLM_ARGS)
  else
    warn "ENGINE=$ENGINE（$ENGINE_ARGS_MODE 模式）忽略 EXTRA_VLLM_ARGS（键名含 vllm，误用大概率是模板叠错）；追加参数请改用 EXTRA_ENGINE_ARGS"
  fi
fi
if [[ -n "${EXTRA_ARGS_FILE:-}" ]]; then
  [[ -f "$EXTRA_ARGS_FILE" ]] || die "EXTRA_ARGS_FILE 不存在：$EXTRA_ARGS_FILE"
  while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    ARGS+=("$line")
  done < "$EXTRA_ARGS_FILE"
fi
if [[ -n "$EXTRA_ENGINE_ARGS" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_ENGINE_ARGS)
fi

# ---------------------------------------------------------------------------
# 3. 环境变量（通用 + 厂商适配；docker/host 共用同一份 K=V）
# ---------------------------------------------------------------------------
if [[ -n "$VENDOR_DEVICE_VIS_ENV" ]]; then
  add_env "$VENDOR_DEVICE_VIS_ENV=$VISIBLE_DEVICES"
fi
if [[ -n "$VENDOR_ALLOC_CONF_NAME" ]]; then
  # 用户显式设置的分配器配置优先（如 PYTORCH_HIP_ALLOC_CONF=...），否则用厂商默认值
  add_env "$VENDOR_ALLOC_CONF_NAME=${!VENDOR_ALLOC_CONF_NAME:-$VENDOR_ALLOC_CONF_VALUE}"
fi
for kv in "${VENDOR_EXTRA_ENV[@]+"${VENDOR_EXTRA_ENV[@]}"}"; do
  [[ -n "$kv" ]] && add_env "$kv"
done
# vLLM 专属 env（VLLM_* / TOKENIZERS_PARALLELISM / TRITON 缓存）按 ENGINE_ARGS_MODE 收拢；
# PYTHONUNBUFFERED / PROMETHEUS_* / EXTRA_ENV 是引擎无关的通用段（设计.md §7.7）。
if [[ "$ENGINE_ARGS_MODE" == "vllm-cli" ]]; then
  add_env "VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
  add_env "TOKENIZERS_PARALLELISM=false"
fi
add_env "PYTHONUNBUFFERED=1"
if [[ "$ENGINE_ARGS_MODE" == "vllm-cli" ]]; then
  [[ -n "$VLLM_CACHE_ROOT" ]] && add_env "VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT"
  [[ -n "$TRITON_CACHE_DIR" ]] && add_env "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
  add_env "VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1}"
fi
if runtime_is_host; then
  add_env "PYTHONPATH=$(host_pythonpath)"
else
  add_env "PYTHONPATH=$(container_pythonpath)"
fi
# 引擎身份进启动环境（非 vLLM 引擎才写，便于事后核对产物归属/复现）：
# ENGINE=vllm（含显式设置）≡ 不设置，不加这一行——红线 1：vLLM 默认路径 dry-run
# 产物与施工前基线逐字节一致（docs/design/README.md §5）。
if [[ "$ENGINE" != "vllm" ]]; then
  add_env "ENGINE=$ENGINE"
fi
# custom 模式引擎：MODEL/SERVED_NAME 是 pack 名/注册名（不进命令行），export 进启动
# 环境便于复现与核对；vLLM 引擎的模型是 serve 位置参数，保持基线不导出这两个键。
if [[ "$ENGINE_ARGS_MODE" != "vllm-cli" ]]; then
  add_env "MODEL=$MODEL"
  add_env "SERVED_NAME=$SERVED_NAME"
fi
[[ -n "${PROMETHEUS_DISABLE_CREATED_SERIES:-}" ]] && add_env "PROMETHEUS_DISABLE_CREATED_SERIES=$PROMETHEUS_DISABLE_CREATED_SERIES"
if [[ -n "$EXTRA_ENV" ]]; then
  for kv in $EXTRA_ENV; do add_env "$kv"; done
fi

# ---------------------------------------------------------------------------
# 4. 组装启动命令
#    入口优先级（设计.md §7.3）：HOST_SERVE_ENTRYPOINT > ENGINE_ENTRYPOINT > 引擎默认
# ---------------------------------------------------------------------------
if runtime_is_host; then
  SERVE_ENTRYPOINT_ACTUAL="$(host_serve_entrypoint)"
else
  if [[ "$ENGINE_ARGS_MODE" == "vllm-cli" ]]; then
    SERVE_ENTRYPOINT_ACTUAL="${SERVE_ENTRYPOINT:-python3 -c 'import torch; from vllm.entrypoints.cli.main import main; main()'}"
  else
    # docker 侧非 vLLM 引擎已在 caps 门禁 die；此分支兜底给未来声明了
    # runtime.docker 的 custom 引擎（ENGINE_ENTRYPOINT 为空时才允许 SERVE_ENTRYPOINT）
    SERVE_ENTRYPOINT_ACTUAL="${SERVE_ENTRYPOINT:-${ENGINE_ENTRYPOINT:-}}"
  fi
fi
if [[ "$ENGINE_ARGS_MODE" != "vllm-cli" && -z "${HOST_SERVE_ENTRYPOINT:-}${SERVE_ENTRYPOINT:-}${ENGINE_ENTRYPOINT:-}" ]]; then
  die "ENGINE=$ENGINE（custom 模式）没有可用的启动入口：请在 engines/$ENGINE.sh 声明 ENGINE_ENTRYPOINT，或用 HOST_SERVE_ENTRYPOINT/SERVE_ENTRYPOINT 显式覆盖"
fi
CMD_STR="$SERVE_ENTRYPOINT_ACTUAL $(quote_argv "${ARGS[@]}")"

echo "$CMD_STR" > "$RUN_DIR/cmd.txt"
{
  echo "# 观测档位: $LEVEL ($LEVEL_DESC)"
  echo "# 运行时: $RUNTIME"
  echo "# 厂商: $VENDOR ($VENDOR_LABEL)"
  echo "# 生成时间: $(date -Is)"
  echo "$CMD_STR"
} >> "$RUN_DIR/args.txt"

# 容器内的工作目录：有源码 checkout 时进 /src，否则进工具包目录（一定存在）
INNER_WORKDIR="/src"
[[ -n "$SRC_DIR" ]] || INNER_WORKDIR="/opt/vllm-probe"
# 容器内启动前奏：source 厂商环境脚本（如 DTK/Ascend 必须）+ 固定 PYTHONPATH。
PRELUDE="$(vendor_prelude_shell)"
INNER_CMD="cd $INNER_WORKDIR && ${PRELUDE}export PYTHONPATH=$(container_pythonpath); ${PLUGIN_SETUP}exec $CMD_STR"

# host 模式的版本信息（best-effort；装不了 vllm/torch 时对应字段为 null）
# 注意：必须在与 serve_launch.sh 相同的环境里采集（先 source HOST_PRELUDE_FILES、
# 再注入 host PYTHONPATH），否则在需要 DTK/Ascend prelude 的机器上会永远拿到 null。
HOST_META_JSON=""
if runtime_is_host; then
  HOST_META_JSON="$(host_meta_json)"
  [[ -n "$HOST_META_JSON" ]] || HOST_META_JSON='{"python": null, "python_version": null, "vllm_version": null, "torch_version": null, "hostname": null, "kernel": null}'
fi

# 引擎版本（best-effort；ENGINE_VERSION_CMD 由引擎 profile 声明，vLLM 走 host_meta 的
# import 探测——host 段 vllm_version 字段保留，另旁增 engine_version）
ENGINE_VERSION_VALUE=""
if runtime_is_host && [[ -n "$ENGINE_VERSION_CMD" ]]; then
  # shellcheck disable=SC2086  # ENGINE_VERSION_CMD 按空白切分为命令 + 参数
  ENGINE_VERSION_VALUE="$(host_exec $ENGINE_VERSION_CMD 2>/dev/null | head -n 1 || true)"
fi

# run_meta.json（供报告读取；docker/host 都有，host 额外带 host 段）
RUN_META_DIR="$RUN_DIR" MODEL="$MODEL" SERVED_NAME="$SERVED_NAME" PORT="$PORT" TP="$TP" \
MAX_MODEL_LEN="$MAX_MODEL_LEN" MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" EAGER="$EAGER" \
PROBE_LEVEL="$LEVEL" LEVEL_DESC="$LEVEL_DESC" CONTAINER="$CONTAINER" IMAGE="$IMAGE" \
ENGINE_VALUE="$ENGINE" ENGINE_LABEL_VALUE="$ENGINE_LABEL" ENGINE_VERSION_VALUE="$ENGINE_VERSION_VALUE" \
RUNTIME="$RUNTIME" HOST_META_JSON="$HOST_META_JSON" STARTED_AT="$(date -Is)" CMD_STR="$CMD_STR" py - <<'PY'
import json, os
runtime = os.environ["RUNTIME"]
meta = {
    "started_at": os.environ["STARTED_AT"],
    "runtime": runtime,
    "image": os.environ["IMAGE"] if runtime == "docker" else "",
    "container": os.environ["CONTAINER"] if runtime == "docker" else "",
    "vendor": os.environ.get("VENDOR", ""),
    "vendor_label": os.environ.get("VENDOR_LABEL", ""),
    "model": os.environ["MODEL"],
    "served_name": os.environ["SERVED_NAME"],
    "port": int(os.environ["PORT"]),
    "tensor_parallel_size": int(os.environ["TP"]),
    "max_model_len": int(os.environ["MAX_MODEL_LEN"]),
    "max_num_batched_tokens": int(os.environ["MAX_NUM_BATCHED_TOKENS"]),
    "enforce_eager": os.environ["EAGER"] == "1",
    "probe_level": os.environ["PROBE_LEVEL"],
    "probe_level_desc": os.environ["LEVEL_DESC"],
    "command": os.environ["CMD_STR"],
}
# 引擎身份字段（T1.9）：非 vLLM 引擎才写——ENGINE=vllm（含未设置）时 run_meta 与
# 施工前基线逐字节一致（design/README.md §5 红线 1）；报告取值链会回落 ENGINE env/默认。
engine_id = os.environ.get("ENGINE_VALUE", "vllm")
if engine_id != "vllm":
    meta["engine"] = engine_id
    meta["engine_label"] = os.environ.get("ENGINE_LABEL_VALUE", "")
if runtime == "host":
    keys = ("python", "python_version", "vllm_version", "torch_version", "hostname", "kernel")
    try:
        meta["host"] = json.loads(os.environ.get("HOST_META_JSON") or "{}")
    except Exception:
        meta["host"] = {}
    if not isinstance(meta.get("host"), dict):
        meta["host"] = {}
    for k in keys:
        meta["host"].setdefault(k, None)
    # 引擎版本（ENGINE_VERSION_CMD 输出；空/未声明为 None）。vllm_version 字段保留原名。
    if engine_id != "vllm":
        meta["host"]["engine_version"] = os.environ.get("ENGINE_VERSION_VALUE") or None
open(os.path.join(os.environ["RUN_META_DIR"], "run_meta.json"), "w").write(
    json.dumps(meta, ensure_ascii=False, indent=2)
)
PY

log "运行目录：$RUN_DIR"
log "厂商：$VENDOR（$VENDOR_LABEL）"
log "观测档位：$LEVEL（$LEVEL_DESC）"

# ---------------------------------------------------------------------------
# 5. 启动
# ---------------------------------------------------------------------------
if runtime_is_host; then
  # ---- host（物理机裸跑）----
  LAUNCHER="$RUN_DIR/serve_launch.sh"
  WORKDIR="$PROJECT_ROOT"
  if [[ -n "$SRC_DIR" && -d "$SRC_DIR" ]]; then WORKDIR="$SRC_DIR"; fi
  HOST_PRELUDE="$(host_prelude_shell)"
  {
    echo '#!/usr/bin/env bash'
    echo '# 由 scripts/start_vllm.sh 生成：host（物理机裸跑）启动脚本'
    echo '# 环境变量来自与 docker 模式完全相同的 ENV_KVS，便于复现/对比。'
    # 引擎身份注记：便于事后核对产物归属（ENGINE=vllm 时不加，保持基线逐字节一致）
    if [[ "$ENGINE" != "vllm" ]]; then
      echo "# 引擎：$ENGINE（$ENGINE_LABEL）"
      echo "# 能力位：$ENGINE_CAPS"
    fi
    echo 'set -euo pipefail'
    printf 'cd %q\n' "$WORKDIR"
    if [[ -n "$HOST_PRELUDE" ]]; then
      # 厂商 env.sh（DTK/CANN/oneAPI）常用裸 ${VAR}，不兼容 set -u；
      # 只在 source prelude 期间临时关闭 nounset，之后恢复（否则 launcher 会立刻退出）。
      echo 'set +u'
      printf '%s\n' "$HOST_PRELUDE"
      echo 'set -u'
    else
      echo '# （HOST_PRELUDE_FILES 为空，跳过宿主机环境脚本）'
    fi
    for kv in "${ENV_KVS[@]+"${ENV_KVS[@]}"}"; do
      # K=V 按第一个 = 切分，值用 printf %q 保证含空格/引号也安全
      printf 'export %s=%q\n' "${kv%%=*}" "${kv#*=}"
    done
    printf 'exec %s %s\n' "$SERVE_ENTRYPOINT_ACTUAL" "$(quote_argv "${ARGS[@]}")"
  } > "$LAUNCHER"
  chmod +x "$LAUNCHER"

  if (( DRY_RUN )); then
    echo
    echo "# 运行时：host（物理机裸跑，不使用 docker）"
    echo "# 厂商：$VENDOR（$VENDOR_LABEL）"
    echo "# 宿主工作目录：$WORKDIR"
    echo "# 启动脚本：$LAUNCHER"
    echo
    echo "===== serve_launch.sh ====="
    cat "$LAUNCHER"
    echo "==========================="
    echo
    echo "# 实际启动方式："
    echo "setsid nohup bash $LAUNCHER > $LOG_FILE 2>&1 &"
    echo
    echo "# vLLM 实际执行的命令："
    echo "$CMD_STR"
    exit 0
  fi

  # 已有 host 实例在跑则拒绝启动（与 docker 的"容器已在运行"对齐）
  if host_instance_running "$RUN_DIR"; then
    die "host 实例已在运行（PID=$(host_instance_pid "$RUN_DIR" 2>/dev/null || echo ?)，run dir: $RUN_DIR）；先执行 scripts/stop_vllm.sh --run-dir $RUN_DIR"
  fi

  # host 预检：宿主依赖检查脚本由 scripts/host_check.sh 提供；不存在则跳过
  if [[ "${HOST_PRECHECK:-1}" == "1" ]] && (( ! SKIP_CHECK )); then
    if [[ -f "$SCRIPT_DIR/host_check.sh" ]]; then
      log "host 预检：bash scripts/host_check.sh --json --outdir $RUN_DIR"
      if ! bash "$SCRIPT_DIR/host_check.sh" --json --outdir "$RUN_DIR"; then
        die "物理机预检未通过（产物见 $RUN_DIR）；修复后重试，或用 --skip-check / HOST_PRECHECK=0 跳过"
      fi
      ok "host 预检通过"
    else
      warn "HOST_PRECHECK=1 但 scripts/host_check.sh 不存在，跳过预检"
    fi
  else
    log "跳过 host 预检（--skip-check 或 HOST_PRECHECK!=1）"
  fi

  # 端口占用检查：vLLM 自己会 bind $HOST:$PORT，提前失败更友好
  if ! "$(host_python)" - "$HOST" "$PORT" <<'PY'
import socket, sys

host, port = sys.argv[1], int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((host, port))
except OSError as exc:
    print(f"端口 {host}:{port} 已被占用：{exc}", file=sys.stderr)
    sys.exit(1)
finally:
    sock.close()
PY
  then
    die "端口 $HOST:$PORT 已被占用；请先停掉占用进程或换一个 PORT（.env 或 PORT=xxxx ...）"
  fi

  # stat logger 插件装到 run 目录（PYTHONPATH 已在 ENV_KVS 里指向 .probe_plugins）
  if [[ "$PROBE_PLUGIN" == "1" ]]; then
    mkdir -p "$RUN_DIR/.probe_plugins" "$RUN_DIR/.pip-tmp"
    # pip 从 pyproject 源码目录安装时会在源码树里生成 build/ 与 *.egg-info，先记录
    # 安装前的状态，安装后把「新出现」的清理掉，避免污染 git 工作区。
    PLUGIN_ARTIFACTS_BEFORE="$(ls -1d "$PROJECT_ROOT"/plugin_probe/build "$PROJECT_ROOT"/plugin_probe/*.egg-info 2>/dev/null || true)"
    log "安装 stat logger 插件到 $RUN_DIR/.probe_plugins"
    # TMPDIR 必须与 --target 同文件系统：默认 /tmp（tmpfs）与仓库跨设备时，
    # pip 构建 wheel 会报 OSError: Invalid cross-device link。
    if TMPDIR="$RUN_DIR/.pip-tmp" "$(host_python)" -m pip install -q --no-deps --no-build-isolation \
        --target "$RUN_DIR/.probe_plugins" "$PROJECT_ROOT/plugin_probe" >/dev/null 2>&1; then
      ok "stat logger 插件已安装（$RUN_DIR/.probe_plugins）"
      rm -rf "$RUN_DIR/.pip-tmp"   # 安装成功即清理临时目录；失败时保留便于排查
    else
      warn "stat logger 插件安装失败（不影响推理，仅缺 JSONL）"
    fi
    PLUGIN_ARTIFACTS_AFTER="$(ls -1d "$PROJECT_ROOT"/plugin_probe/build "$PROJECT_ROOT"/plugin_probe/*.egg-info 2>/dev/null || true)"
    for _art in $PLUGIN_ARTIFACTS_AFTER; do
      grep -qxF "$_art" <<< "$PLUGIN_ARTIFACTS_BEFORE" || rm -rf "$_art"
    done
  fi

  log "host 启动：$LAUNCHER（工作目录 $WORKDIR）"
  HOST_PID="$(host_start "$RUN_DIR" "$LOG_FILE" "$LAUNCHER")"
  ok "host 实例已启动：PID=$HOST_PID PGID=$(cat "$RUN_DIR/serve.pgid" 2>/dev/null || echo "$HOST_PID")"
  log "日志：$LOG_FILE"
else
  # ---- docker（原有行为）----
  if [[ "$MODE" == "run" ]]; then
    if container_exists "$CONTAINER"; then
      if container_running "$CONTAINER"; then
        die "容器 $CONTAINER 已在运行；先执行 scripts/stop_vllm.sh，或设置 CONTAINER=其它名字"
      fi
      log "移除已退出的同名容器 $CONTAINER"
      docker rm "$CONTAINER" >/dev/null
    fi

    DOCKER_ARGS=(
      -d --name "$CONTAINER"
      --security-opt label=disable
      --cap-add SYS_PTRACE
      --ipc host
      --shm-size "${SHM_SIZE:-16g}"
      -p "$PORT:$INTERNAL_PORT"
      --ulimit memlock=-1
      --ulimit stack=67108864
    )
    # 厂商设备直通（动态函数或静态数组；每行一个参数）
    while IFS= read -r arg; do
      [[ -n "$arg" ]] && DOCKER_ARGS+=("$arg")
    done < <(vendor_device_args)
    # 厂商额外挂载（宿主机路径存在才挂）
    while IFS= read -r spec; do
      [[ -n "$spec" ]] && DOCKER_ARGS+=(-v "$spec")
    done < <(vendor_mounts)
    # 用户额外挂载（EXTRA_MOUNTS，空格分隔的 host:container[:ro]）
    if [[ -n "$EXTRA_MOUNTS" ]]; then
      # shellcheck disable=SC2206
      for spec in $EXTRA_MOUNTS; do DOCKER_ARGS+=(-v "$spec"); done
    fi
    [[ -n "$SRC_DIR" ]] && DOCKER_ARGS+=(-v "$SRC_DIR":/src:ro)
    DOCKER_ARGS+=(
      -v "$PROJECT_ROOT":/opt/vllm-probe:ro
      -v "$RUN_DIR":/logs
      -v "$TRACE_DIR":/traces
    )
    [[ -n "$GAME_DIR" ]] && DOCKER_ARGS+=(-v "$GAME_DIR":"$GAME_DIR")
    # 共用的 K=V 在 docker 侧派生成 -e K=V
    for kv in "${ENV_KVS[@]+"${ENV_KVS[@]}"}"; do DOCKER_ARGS+=(-e "$kv"); done
    DOCKER_ARGS+=("$IMAGE" bash -lc "$INNER_CMD > /logs/serve.log 2>&1")

    if (( DRY_RUN )); then
      echo
      echo "# 厂商：$VENDOR（$VENDOR_LABEL）"
      echo "# 容器内实际执行的命令："
      echo "$INNER_CMD"
      echo
      echo "docker run \\"
      printf '  %s \\\n' "${DOCKER_ARGS[@]}"
      exit 0
    fi

    log "docker run 启动容器 $CONTAINER（镜像 $IMAGE）"
    docker run "${DOCKER_ARGS[@]}" >/dev/null
    log "容器已启动，日志：$LOG_FILE"

  elif [[ "$MODE" == "exec" ]]; then
    [[ -n "$EXEC_CONTAINER" ]] || die "MODE=exec 需要设置 EXEC_CONTAINER（要复用哪个容器）"
    container_running "$EXEC_CONTAINER" || die "容器 $EXEC_CONTAINER 未运行"
    EXEC_LOG_DIR="${EXEC_LOG_DIR:-${GAME_DIR:-/tmp}/vllm-probe}"
    docker exec "$EXEC_CONTAINER" mkdir -p "$EXEC_LOG_DIR"
    EXEC_LOG="$EXEC_LOG_DIR/serve-$(date +%H%M%S).log"
    DOCKER_EXEC_ENV_ARGS=()
    for kv in "${ENV_KVS[@]+"${ENV_KVS[@]}"}"; do DOCKER_EXEC_ENV_ARGS+=(-e "$kv"); done
    if (( DRY_RUN )); then
      echo "# 容器内实际执行的命令："
      echo "$INNER_CMD > $EXEC_LOG 2>&1"
      echo
      echo "docker exec -d ${DOCKER_EXEC_ENV_ARGS[*]:-} $EXEC_CONTAINER bash -lc \"...\""
      exit 0
    fi
    log "docker exec -d 在已有容器 $EXEC_CONTAINER 中启动（端口 $PORT 需与该容器映射一致）"
    docker exec -d "${DOCKER_EXEC_ENV_ARGS[@]+"${DOCKER_EXEC_ENV_ARGS[@]}"}" "$EXEC_CONTAINER" bash -lc "$INNER_CMD > $EXEC_LOG 2>&1"
    LOG_FILE="$EXEC_LOG"
    ln -sfn "$LOG_FILE" "$RUN_DIR/serve.log" 2>/dev/null || true
    log "日志：$LOG_FILE（已软链到 $RUN_DIR/serve.log）"
  else
    die "MODE 非法：$MODE（run|exec）"
  fi
fi

# ---------------------------------------------------------------------------
# 6. 等待就绪
# ---------------------------------------------------------------------------
if (( FOREGROUND )); then
  log "前台跟踪日志（Ctrl-C 结束跟踪，实例继续运行）"
  tail -f "$LOG_FILE"
  exit 0
fi

if (( DO_WAIT )); then
  log "等待引擎就绪（最多 ${WARMUP_TIMEOUT}s；大模型冷启动通常 5-15 分钟）"
  BASE="http://127.0.0.1:$PORT"
  # 就绪判定在 common.sh::wait_engine_ready（与 wait_ready.sh 共用，含 caps 协商：
  # 无 api.health 退化端口探测、无 /metrics 面跳过指标等待、ENGINE_METRICS_FETCH_SUFFIX）
  if wait_engine_ready "$BASE" "$WARMUP_TIMEOUT"; then
    ok "引擎就绪：$BASE"
  else
    warn "健康检查超时，最后 30 行日志："
    tail -30 "$LOG_FILE" 2>/dev/null || true
    die "引擎未在 ${WARMUP_TIMEOUT}s 内就绪"
  fi
  log "下一步：scripts/status.sh / scripts/collect_metrics.sh / scripts/probe_all.sh"
fi
