#!/usr/bin/env bash
# 在 docker 中启动一个"可观测的" vLLM 实例。
#
# 多厂商：容器设备直通 / 可见性环境变量 / 启动前 source 的环境脚本全部来自
# vendors/<id>.sh（VENDOR=auto 时自动探测，见 vendors/README.md）。
# 观测档位与厂商无关，见下。
#
# 用法：
#   scripts/start_vllm.sh                        # 按 .env 的 PROBE_LEVEL 启动
#   scripts/start_vllm.sh --level kernel         # 临时切到算子级观测档位
#   scripts/start_vllm.sh --dry-run              # 只打印将执行的命令，不启动
#   scripts/start_vllm.sh --exec                 # 复用已有容器（docker exec -d）
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

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

LEVEL=""
DRY_RUN=0
FOREGROUND=0
TAG="serve"
DO_WAIT=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --level) LEVEL="${2:?}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --foreground) FOREGROUND=1; shift ;;
    --tag) TAG="${2:?}"; shift 2 ;;
    --no-wait) DO_WAIT=0; shift ;;
    --run-dir) RUN_DIR="${2:?}"; shift 2 ;;
    --exec) MODE=exec; shift ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) die "未知参数：$1（-h 查看用法）" ;;
  esac
done
[[ -n "$LEVEL" ]] || LEVEL="$PROBE_LEVEL"

require_cmd docker
case "$LEVEL" in off|basic|steps|deep|kernel) ;; *) die "PROBE_LEVEL 非法：$LEVEL（off|basic|steps|deep|kernel）" ;; esac

[[ -n "$MODEL" ]] || die "MODEL 未设置：请在 .env 里指定模型目录（参考 .env.example）"
[[ -n "$IMAGE" ]] || die "IMAGE 未设置且厂商（$VENDOR）没有默认镜像：请在 .env 里指定 IMAGE。提示：$VENDOR_IMAGE_HINT"
[[ -n "$SERVED_NAME" ]] || SERVED_NAME="$(basename "$MODEL")"
if [[ -n "$DRAFT_MODEL" && -z "$SPEC_METHOD" ]]; then
  die "配置了 DRAFT_MODEL 但没有 SPEC_METHOD（如 dflash / mtp / eagle）：请在 .env 里设置"
fi

RUN_DIR="$(new_run_dir "$TAG")"
LOG_FILE="$RUN_DIR/serve.log"
TRACE_DIR="$RUN_DIR/traces"
mkdir -p "$TRACE_DIR" "$RUN_DIR"

# PORT 是宿主机端口（探针访问用）；INTERNAL_PORT 是容器内 vLLM 监听端口。
# MODE=run 时二者默认相同；配合 docker-compose.yml 时通常 INTERNAL_PORT=8000。
: "${INTERNAL_PORT:=$PORT}"

# ---------------------------------------------------------------------------
# 1. 服务参数
# ---------------------------------------------------------------------------
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

# 投机解码（可选）
SPEC_DECODE_ON=0
if [[ -n "$DRAFT_MODEL" ]]; then
  SPEC_DECODE_ON=1
  SPEC="{\"model\":\"$DRAFT_MODEL\",\"method\":\"$SPEC_METHOD\",\"num_speculative_tokens\":$SPEC_TOKENS"
  [[ -n "${SPEC_DTYPE:-}" ]] && SPEC="$SPEC,\"dtype\":\"$SPEC_DTYPE\""
  [[ -n "${SPEC_KV_DTYPE:-}" ]] && SPEC="$SPEC,\"kv_cache_dtype\":\"$SPEC_KV_DTYPE\""
  [[ -n "$SPEC_EXTRA_JSON" ]] && SPEC="$SPEC,$SPEC_EXTRA_JSON"
  SPEC="$SPEC}"
  ARGS+=(--speculative-config "$SPEC")
fi

# ---------------------------------------------------------------------------
# 2. 观测开关（按档位叠加；与厂商无关）
# ---------------------------------------------------------------------------
ENVS=()
add_env() { ENVS+=(-e "$1"); }

case "$LEVEL" in
  off)   LEVEL_DESC="关闭全部可选观测开关（对照基线）" ;;
  basic) LEVEL_DESC="默认周期 stats（VLLM_LOG_STATS_INTERVAL=${LOG_STATS_INTERVAL}s）+ /metrics" ;;
  steps) LEVEL_DESC="逐 iteration 明细 + 逐请求日志 + 逐请求 metrics 字段" ;;
  deep)  LEVEL_DESC="steps + KV block 驻留 + 图调度统计 + MFU + 逐步投机解码指标" ;;
  kernel) LEVEL_DESC="deep + torch profiler + 逐层 NVTX（会自动关闭图模式）" ;;
esac

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
  PROFILER_JSON=$(printf '{"profiler":"torch","torch_profiler_dir":"/traces","torch_profiler_with_stack":%s,"torch_profiler_with_flops":%s,"torch_profiler_record_shapes":%s,"torch_profiler_with_memory":%s,"torch_profiler_dump_cuda_time_total":true}' \
    "$PROFILER_WITH_STACK" "$PROFILER_WITH_FLOPS" "$PROFILER_RECORD_SHAPES" "$PROFILER_WITH_MEMORY")
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

# ---------------------------------------------------------------------------
# 可选：stat logger 插件（plugin_probe/）——把每个 engine step 的结构化统计写成 JSONL
# 它是"探测所有步骤"最深的一层：直接拿到引擎内部的 SchedulerStats / IterationStats
# （逐 step 批次、逐请求 queue/prefill/decode/e2e/TPOT、KV 驱逐、按位置接受数…）。
# ---------------------------------------------------------------------------
PLUGIN_SETUP=""
if [[ "$PROBE_PLUGIN" == "1" ]]; then
  if [[ "$LEVEL" == "off" || "$LEVEL" == "basic" ]]; then
    warn "PROBE_PLUGIN=1 但档位是 $LEVEL：逐 step 明细需要 --enable-logging-iteration-details，建议用 steps/deep/kernel"
  fi
  if [[ "$MODE" == "exec" ]]; then
    PROBE_LOG_CONTAINER="${EXEC_LOG_DIR:-${GAME_DIR:-/tmp}/vllm-probe}/$PROBE_LOG_NAME"
  else
    PROBE_LOG_CONTAINER="/logs/$PROBE_LOG_NAME"
  fi
  add_env "VLLM_PROBE_LOG=$PROBE_LOG_CONTAINER"
  # 关键：在容器内 /tmp 复制一份再安装，避免容器里的 root 在宿主机工作区留下
  # root 属主的 egg-info / __pycache__（实测踩过这个坑）。
  PLUGIN_SETUP="python3 -c 'import vllm_probe_stat_logger' 2>/dev/null || "
  PLUGIN_SETUP+="{ rm -rf /tmp/vllm-probe-plugin && cp -r /opt/vllm-probe/plugin_probe /tmp/vllm-probe-plugin && "
  PLUGIN_SETUP+="pip install -q --no-deps --no-build-isolation -e /tmp/vllm-probe-plugin >/dev/null 2>&1 "
  PLUGIN_SETUP+="&& echo '[probe] stat logger 插件已安装(JSONL)'; } || "
  PLUGIN_SETUP+="echo '[probe] 警告: stat logger 插件安装失败（不影响推理，仅缺 JSONL）' >&2; "
  log "stat logger 插件：开（JSONL → $PROBE_LOG_CONTAINER）"
else
  log "stat logger 插件：关（PROBE_PLUGIN=1 可开启；提供最细的逐 step/逐请求结构化数据）"
fi

# 用户自定义参数：EXTRA_VLLM_ARGS 按空白切分；EXTRA_ARGS_FILE 每行一个参数（JSON 参数请用后者）
if [[ -n "$EXTRA_VLLM_ARGS" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_VLLM_ARGS)
fi
if [[ -n "${EXTRA_ARGS_FILE:-}" ]]; then
  [[ -f "$EXTRA_ARGS_FILE" ]] || die "EXTRA_ARGS_FILE 不存在：$EXTRA_ARGS_FILE"
  while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    ARGS+=("$line")
  done < "$EXTRA_ARGS_FILE"
fi

# ---------------------------------------------------------------------------
# 3. 环境变量（通用 + 厂商适配）
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
add_env "VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
add_env "TOKENIZERS_PARALLELISM=false"
add_env "PYTHONUNBUFFERED=1"
[[ -n "$VLLM_CACHE_ROOT" ]] && add_env "VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT"
[[ -n "$TRITON_CACHE_DIR" ]] && add_env "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
add_env "VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1}"
add_env "PYTHONPATH=$(container_pythonpath)"
[[ -n "${PROMETHEUS_DISABLE_CREATED_SERIES:-}" ]] && add_env "PROMETHEUS_DISABLE_CREATED_SERIES=$PROMETHEUS_DISABLE_CREATED_SERIES"
if [[ -n "$EXTRA_ENV" ]]; then
  for kv in $EXTRA_ENV; do add_env "$kv"; done
fi

# ---------------------------------------------------------------------------
# 4. 组装容器内命令
# ---------------------------------------------------------------------------
SERVE_ENTRYPOINT="${SERVE_ENTRYPOINT:-python3 -c 'import torch; from vllm.entrypoints.cli.main import main; main()'}"
CMD_STR="$SERVE_ENTRYPOINT $(quote_argv "${ARGS[@]}")"

echo "$CMD_STR" > "$RUN_DIR/cmd.txt"
{
  echo "# 观测档位: $LEVEL ($LEVEL_DESC)"
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

# run_meta.json（供报告读取）
RUN_META_DIR="$RUN_DIR" MODEL="$MODEL" SERVED_NAME="$SERVED_NAME" PORT="$PORT" TP="$TP" \
MAX_MODEL_LEN="$MAX_MODEL_LEN" MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" EAGER="$EAGER" \
PROBE_LEVEL="$LEVEL" LEVEL_DESC="$LEVEL_DESC" CONTAINER="$CONTAINER" IMAGE="$IMAGE" \
STARTED_AT="$(date -Is)" CMD_STR="$CMD_STR" py - <<'PY'
import json, os
meta = {
    "started_at": os.environ["STARTED_AT"],
    "image": os.environ["IMAGE"],
    "container": os.environ["CONTAINER"],
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
  for env in "${ENVS[@]}"; do DOCKER_ARGS+=("$env"); done
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
  if (( DRY_RUN )); then
    echo "# 容器内实际执行的命令："
    echo "$INNER_CMD > $EXEC_LOG 2>&1"
    echo
    echo "docker exec -d ${ENVS[*]:-} $EXEC_CONTAINER bash -lc \"...\""
    exit 0
  fi
  log "docker exec -d 在已有容器 $EXEC_CONTAINER 中启动（端口 $PORT 需与该容器映射一致）"
  docker exec -d "${ENVS[@]+"${ENVS[@]}"}" "$EXEC_CONTAINER" bash -lc "$INNER_CMD > $EXEC_LOG 2>&1"
  LOG_FILE="$EXEC_LOG"
  ln -sfn "$LOG_FILE" "$RUN_DIR/serve.log" 2>/dev/null || true
  log "日志：$LOG_FILE（已软链到 $RUN_DIR/serve.log）"
else
  die "MODE 非法：$MODE（run|exec）"
fi

# ---------------------------------------------------------------------------
# 6. 等待就绪
# ---------------------------------------------------------------------------
if (( FOREGROUND )); then
  log "前台跟踪日志（Ctrl-C 结束跟踪，容器继续运行）"
  tail -f "$LOG_FILE"
  exit 0
fi

if (( DO_WAIT )); then
  log "等待引擎就绪（最多 ${WARMUP_TIMEOUT}s；大模型冷启动通常 5-15 分钟）"
  BASE="http://127.0.0.1:$PORT"
  if ! wait_http "$BASE/health" "$WARMUP_TIMEOUT" 200; then
    warn "健康检查超时，最后 30 行日志："
    tail -30 "$LOG_FILE" 2>/dev/null || true
    die "引擎未在 ${WARMUP_TIMEOUT}s 内就绪"
  fi
  # /health 为 200 后还要等引擎指标出现（就绪指标由厂商 profile 定义）
  waited=0
  while (( waited < 120 )); do
    if curl -s -m 5 "$BASE/metrics" 2>/dev/null | grep -qE "$VENDOR_READY_METRIC"; then break; fi
    sleep 5; waited=$((waited + 5))
  done
  ok "引擎就绪：$BASE （耗时约 ${waited}s 的指标等待）"
  log "下一步：scripts/status.sh / scripts/collect_metrics.sh / scripts/probe_all.sh"
fi
