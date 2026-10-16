#!/usr/bin/env bash
# 公共函数库：配置加载、厂商适配层、日志、docker/路径辅助。
# 其它脚本通过 `source "$(dirname "$0")/lib/common.sh"` 引入。
#
# 约定：
#   * 所有可调参数来自 <项目根>/.env（见 .env.example），也可用环境变量临时覆盖；
#   * 厂商适配来自 vendors/<id>.sh（VENDOR=auto 时自动探测，见 vendors/README.md）；
#   * RUN_DIR 未指定时自动创建 runs/<时间戳>/，所有产物落在里面；
#   * 本文件在 source 时会加载 .env 并完成厂商探测（无其它副作用）。

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT

# ---------------------------------------------------------------------------
# 加载 .env（显式环境变量优先：`VENDOR=nvidia make up` 这类前缀覆盖必须生效）
# ---------------------------------------------------------------------------
_dump_env_after_source() {
  set -a
  # shellcheck disable=SC1090
  . "$1" >/dev/null 2>&1 || exit 0
  local name
  for name in $(compgen -e); do
    printf '%s\0%s\0' "$name" "${!name}"
  done
}

load_env_file() {
  local file="$1" name value
  local -A pre=()
  for name in $(compgen -e); do pre["$name"]=1; done
  while IFS= read -r -d '' name; do
    IFS= read -r -d '' value || break
    [[ -n "${pre[$name]:-}" ]] && continue
    export "$name=$value"
  done < <(bash -c "$(declare -f _dump_env_after_source); _dump_env_after_source \"\$1\"" _ "$file")
}

ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
  load_env_file "$ENV_FILE"
fi

# ---------------------------------------------------------------------------
# 默认值（.env 没写时生效；这里只放与厂商无关的通用默认值，
# 站点特定路径/镜像请在 .env 中配置 —— 见 .env.example）
# ---------------------------------------------------------------------------
: "${VENDOR:=auto}"                    # auto | generic | nvidia | amd | hygon | intel | ascend | 自定义
: "${IMAGE:=}"                         # 空 → 用厂商 profile 的 VENDOR_DEFAULT_IMAGE 兜底
: "${CONTAINER:=vllm-probe}"
: "${MODE:=run}"                       # run=新建容器, exec=复用已有容器
: "${EXEC_CONTAINER:=}"                # MODE=exec 时必填
: "${SRC_DIR:=}"                       # vLLM 源码 checkout（挂到容器 /src；留空则不挂）
: "${MODEL:=}"                         # 模型目录（必填，启动时校验）
: "${SERVED_NAME:=}"                   # 留空 → 取 MODEL 的 basename
: "${HOST:=0.0.0.0}"
: "${PORT:=8100}"
: "${TP:=1}"
: "${DTYPE:=float16}"
: "${GPU_MEM_UTIL:=0.85}"
: "${MAX_MODEL_LEN:=32768}"
: "${MAX_NUM_BATCHED_TOKENS:=8192}"
: "${MAX_NUM_SEQS:=64}"
: "${EAGER:=0}"
: "${VISIBLE_DEVICES:=0}"
: "${DRAFT_MODEL:=}"                   # 留空 = 不启用投机解码
: "${SPEC_METHOD:=}"
: "${SPEC_TOKENS:=7}"
: "${SPEC_EXTRA_JSON:=}"
: "${PROBE_LEVEL:=steps}"              # off | basic | steps | deep | kernel
: "${RUNS_DIR:=$PROJECT_ROOT/runs}"
: "${RUN_DIR:=}"
: "${LOG_STATS_INTERVAL:=5}"
: "${LOG_BATCHSIZE_INTERVAL:=100}"
: "${VLLM_LOGGING_LEVEL:=INFO}"
: "${WARMUP_TIMEOUT:=900}"
: "${EXTRA_VLLM_ARGS:=}"
: "${EXTRA_ARGS_FILE:=}"
: "${EXTRA_ENV:=}"
: "${EXTRA_MOUNTS:=}"                  # 额外挂载（空格分隔的 host:container[:ro] 列表）
: "${GAME_DIR:=}"                      # 共享数据目录挂载（可选，站点配置，如 /mnt/game）
: "${PROMPT_TOKENS:=512}"
: "${MAX_TOKENS:=128}"
: "${CONCURRENCY:=1}"
: "${REPEAT:=3}"
: "${SHARED_PREFIX_TOKENS:=0}"
: "${PROBE_ENDPOINT:=chat}"
: "${GPU_SAMPLE_INTERVAL:=1}"
: "${PROFILER_WITH_STACK:=true}"
: "${PROFILER_WITH_FLOPS:=true}"
: "${PROFILER_RECORD_SHAPES:=true}"
: "${PROFILER_WITH_MEMORY:=true}"
# kernel 档：让引擎内部的 record_function 范围出现在 torch trace 里
# （schedule: allocate_slots、gpu_model_runner: forward/sample/postprocess、
#   llm_engine step: * 等），报告的 "Top CPU 侧范围" 表就读这些数据。
: "${ENABLE_PROFILING_SCOPES:=1}"
# 想改用 NVTX 范围（给 rocprof/nsys 用）而不是 torch record_function 时设为 1；
# 二者在 vLLM 内部是 if/elif，CUSTOM 优先。
: "${ENABLE_NVTX_SCOPES:=0}"
: "${ENABLE_LAYERWISE_NVTX:=1}"
# stat logger 插件：把每个 engine step 的结构化统计写成 JSONL（见 plugin_probe/）
: "${PROBE_PLUGIN:=0}"
: "${PROBE_LOG_NAME:=probe_steps.jsonl}"
# Triton 相关（站点可选；留空则不注入/不挂 PYTHONPATH）
: "${TRITON_INSTALL_DIR:=}"
: "${TRITON_CACHE_DIR:=}"
: "${VLLM_CACHE_ROOT:=}"
# 容器内最终 PYTHONPATH（留空 → 由容器内脚本自动拼装，见 container_pythonpath）
: "${PYTHONPATH_IN_CONTAINER:=}"
: "${SHM_SIZE:=16g}"

export IMAGE CONTAINER MODE EXEC_CONTAINER SRC_DIR MODEL SERVED_NAME HOST PORT TP DTYPE
export GPU_MEM_UTIL MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS EAGER VISIBLE_DEVICES
export DRAFT_MODEL SPEC_METHOD SPEC_TOKENS PROBE_LEVEL RUNS_DIR WARMUP_TIMEOUT

# ---------------------------------------------------------------------------
# 厂商适配层（vendors/<id>.sh ↔ probe/vendors.py）
# ---------------------------------------------------------------------------
VENDORS_DIR="$PROJECT_ROOT/vendors"
export VENDORS_DIR

list_vendors() {
  local f base
  for f in "$VENDORS_DIR"/*.sh; do
    base="$(basename "$f" .sh)"
    [[ "$base" == _* ]] && continue
    printf '%s\n' "$base"
  done
}

vendor_profile_path() {
  printf '%s/%s.sh' "$VENDORS_DIR" "$1"
}

# 在子 shell 中读取 profile 的某个变量（无副作用；用于自动探测元数据）
vendor_meta() {
  local id="$1" key="$2"
  bash -c 'source "$1" >/dev/null 2>&1 || exit 1; printf "%s" "${!2:-}"' _ \
    "$(vendor_profile_path "$id")" "$key"
}

vendor_auto_matches() {
  local id="$1" smi dev candidate
  smi="$(vendor_meta "$id" VENDOR_DETECT_SMI)"
  dev="$(vendor_meta "$id" VENDOR_DETECT_DEVICE)"
  [[ -n "$smi" || -n "$dev" ]] || return 1
  for candidate in $smi; do
    command -v "$candidate" >/dev/null 2>&1 && return 0
  done
  if [[ -n "$dev" ]]; then
    # shellcheck disable=SC2086
    compgen -G $dev >/dev/null 2>&1 && return 0
  fi
  return 1
}

load_vendor_profile() {
  local id="$1"
  # shellcheck disable=SC1090
  source "$(vendor_profile_path "$id")"
  VENDOR="$id"
  : "${VENDOR_LABEL:=$id}"
  : "${VENDOR_SMI_CMD:=}"
  : "${VENDOR_SMI_FALLBACKS:=}"
  : "${VENDOR_TRACE_PACK:=base}"
  : "${VENDOR_READY_METRIC:=^vllm:cache_config_info}"
  : "${VENDOR_DEVICE_VIS_ENV:=}"
  : "${VENDOR_ALLOC_CONF_NAME:=}"
  : "${VENDOR_ALLOC_CONF_VALUE:=expandable_segments:True}"
  : "${VENDOR_DEFAULT_IMAGE:=}"
  : "${VENDOR_IMAGE_HINT:=}"
  : "${VENDOR_EXTRA_ENV_STR:=}"
  if ! declare -p VENDOR_DOCKER_DEVICE_ARGS >/dev/null 2>&1; then
    VENDOR_DOCKER_DEVICE_ARGS=()
  fi
  if ! declare -p VENDOR_DOCKER_MOUNTS >/dev/null 2>&1; then
    VENDOR_DOCKER_MOUNTS=()
  fi
  if ! declare -p VENDOR_PRELUDE_FILES >/dev/null 2>&1; then
    VENDOR_PRELUDE_FILES=()
  fi
  if ! declare -p VENDOR_EXTRA_ENV >/dev/null 2>&1; then
    VENDOR_EXTRA_ENV=()
  fi
  # 镜像兜底：.env 未设置 IMAGE 时用厂商 profile 的默认镜像
  if [[ -z "$IMAGE" && -n "$VENDOR_DEFAULT_IMAGE" ]]; then
    IMAGE="$VENDOR_DEFAULT_IMAGE"
  fi
  export VENDOR VENDOR_LABEL VENDOR_SMI_CMD VENDOR_SMI_FALLBACKS VENDOR_TRACE_PACK
  export VENDOR_READY_METRIC VENDOR_DEVICE_VIS_ENV VENDOR_ALLOC_CONF_NAME VENDOR_ALLOC_CONF_VALUE
  export VENDOR_DEFAULT_IMAGE VENDOR_IMAGE_HINT IMAGE
}

detect_vendor() {
  local requested="${VENDOR:-auto}" id="" candidate
  if [[ "$requested" != "auto" ]]; then
    [[ -f "$(vendor_profile_path "$requested")" ]] \
      || die "未知厂商 VENDOR=$requested（可用：$(list_vendors | paste -sd' ' -)）"
    id="$requested"
  else
    for candidate in ascend nvidia hygon amd intel; do
      if vendor_auto_matches "$candidate"; then id="$candidate"; break; fi
    done
    [[ -n "$id" ]] || id="generic"
  fi
  load_vendor_profile "$id"
}

# 设备直通参数：优先动态函数，否则用静态数组；每行一个参数
vendor_device_args() {
  if declare -F vendor_dynamic_docker_args >/dev/null 2>&1; then
    vendor_dynamic_docker_args
  else
    local arg
    for arg in "${VENDOR_DOCKER_DEVICE_ARGS[@]+"${VENDOR_DOCKER_DEVICE_ARGS[@]}"}"; do
      printf '%s\n' "$arg"
    done
  fi
}

# 额外挂载：每行一个 host:container[:ro]，仅在宿主机路径存在时输出
vendor_mounts() {
  local spec host
  for spec in "${VENDOR_DOCKER_MOUNTS[@]+"${VENDOR_DOCKER_MOUNTS[@]}"}"; do
    [[ -n "$spec" ]] || continue
    host="${spec%%:*}"
    if [[ -e "$host" ]]; then
      printf '%s\n' "$spec"
    else
      warn "厂商挂载源不存在，已跳过：$spec"
    fi
  done
}

# 容器内 PYTHONPATH：优先 .env 的显式配置，否则按 SRC_DIR/TRITON_INSTALL_DIR 自动拼装
container_pythonpath() {
  if [[ -n "${PYTHONPATH_IN_CONTAINER:-}" ]]; then
    printf '%s' "$PYTHONPATH_IN_CONTAINER"
    return 0
  fi
  local p="/opt/vllm-probe"
  [[ -n "${SRC_DIR:-}" ]] && p="/src:$p"
  [[ -n "${TRITON_INSTALL_DIR:-}" ]] && p="$p:$TRITON_INSTALL_DIR"
  printf '%s' "$p"
}

# 在容器内启动服务前的 source 命令串（厂商 profile 的 VENDOR_PRELUDE_FILES）
vendor_prelude_shell() {
  local out="" f
  for f in "${VENDOR_PRELUDE_FILES[@]+"${VENDOR_PRELUDE_FILES[@]}"}"; do
    [[ -n "$f" ]] || continue
    out+="[ -f $f ] && . $f >/dev/null 2>&1; "
  done
  printf '%s' "$out"
}

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_INFO=$'\033[36m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_OK=$'\033[32m'
else
  C_RESET=""; C_INFO=""; C_WARN=""; C_ERR=""; C_OK=""
fi

log()  { printf '%s[%s]%s %s\n' "$C_INFO" "$(date +%H:%M:%S)" "$C_RESET" "$*"; }
ok()   { printf '%s[%s] ✅ %s%s\n' "$C_OK" "$(date +%H:%M:%S)" "$*" "$C_RESET"; }
warn() { printf '%s[%s] ⚠️  %s%s\n' "$C_WARN" "$(date +%H:%M:%S)" "$*" "$C_RESET" >&2; }
die()  { printf '%s[%s] ❌ %s%s\n' "$C_ERR" "$(date +%H:%M:%S)" "$*" "$C_RESET" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "缺少命令：$1"
}

# 厂商探测在日志/工具函数就绪之后执行（detect_vendor 会在参数非法时调用 die）
detect_vendor

# ---------------------------------------------------------------------------
# Python 探针
# ---------------------------------------------------------------------------
py() {
  PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 "$@"
}

# ---------------------------------------------------------------------------
# 运行目录
# ---------------------------------------------------------------------------
new_run_dir() {
  local tag="${1:-run}"
  if [[ -n "$RUN_DIR" ]]; then
    mkdir -p "$RUN_DIR"
    echo "$RUN_DIR"
    return 0
  fi
  local dir="$RUNS_DIR/$(date +%Y%m%d-%H%M%S)-$tag"
  mkdir -p "$dir"
  echo "$dir"
}

latest_run_dir() {
  if [[ -n "$RUN_DIR" ]]; then echo "$RUN_DIR"; return 0; fi
  local latest
  latest="$(ls -1d "$RUNS_DIR"/*/ 2>/dev/null | sort | tail -1 || true)"
  [[ -n "$latest" ]] || die "找不到任何运行目录（$RUNS_DIR），请先执行 scripts/start_vllm.sh"
  echo "${latest%/}"
}

# ---------------------------------------------------------------------------
# docker 辅助
# ---------------------------------------------------------------------------
container_exists() {
  docker inspect "$1" >/dev/null 2>&1
}

container_running() {
  [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null || echo false)" == "true" ]]
}

container_python() {
  local name="$1"; shift
  docker exec "$name" bash -lc "$(vendor_prelude_shell)export PYTHONPATH=$(container_pythonpath); python3 \"\$@\"" -- "$@"
}

# 在后台把 `docker logs -f` 落到文件，便于逐 step 解析（返回 pid 文件路径）
start_log_follower() {
  local container="$1" logfile="$2"
  mkdir -p "$(dirname "$logfile")"
  : > "$logfile"
  nohup docker logs -f --since 1m "$container" >> "$logfile" 2>&1 &
  local pid=$!
  echo "$pid" > "$logfile.pid"
  echo "$logfile"
}

stop_log_follower() {
  local logfile="$1"
  local pidfile="$logfile.pid"
  if [[ -f "$pidfile" ]]; then
    local pid
    pid="$(cat "$pidfile")"
    kill "$pid" 2>/dev/null || true
    rm -f "$pidfile"
  fi
}

wait_http() {
  local url="$1" timeout="${2:-300}" expect="${3:-200}"
  local waited=0
  while (( waited < timeout )); do
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$url" 2>/dev/null || echo 000)"
    if [[ "$code" == "$expect" ]]; then
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
    if (( waited % 30 == 0 )); then log "等待 $url ... 已 ${waited}s（HTTP $code）"; fi
  done
  return 1
}

# 解析 shell 数组为可读命令行（写入 cmd.txt 用）
quote_argv() {
  local out=""
  local arg
  for arg in "$@"; do
    out+="$(printf '%q' "$arg") "
  done
  printf '%s' "${out% }"
}
