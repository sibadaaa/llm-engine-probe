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

# --- 运行时：docker（默认）| host（物理机裸跑，无容器） ---
# 对 docker 模式无影响；host 相关默认值见下方「运行时抽象」一节。
: "${RUNTIME:=docker}"                 # docker | host（别名 container→docker、bare/native→host）
: "${HOST_PYTHON:=python3}"            # host 模式使用的宿主机 python 解释器
: "${HOST_SERVE_ENTRYPOINT:=}"         # 留空 → 用 $HOST_PYTHON 执行 vLLM 的 CLI 入口
: "${HOST_PRECHECK:=1}"                # 1 = host 启动前跑 scripts/host_check.sh（--skip-check 可跳过）
: "${HOST_PRELUDE_FILES:=}"            # host 启动前要 source 的宿主机环境脚本（空格分隔）
: "${HOST_STOP_TIMEOUT:=60}"           # host 停止时 SIGTERM 后等待秒数，超时 SIGKILL
: "${HOST_VLLM_PATTERN:=vllm.entrypoints.cli.main|vllm\.v1\.engine|EngineCore|APIServer}"
: "${PYSPY_BIN:=py-spy}"               # host 模式栈采样使用的 py-spy 可执行文件
# 1 = 停止后跳过显存释放等待（测试/CI/同机已有其它显存占用时很有用）
: "${PROBE_SKIP_VRAM_WAIT:=0}"

export IMAGE CONTAINER MODE EXEC_CONTAINER SRC_DIR MODEL SERVED_NAME HOST PORT TP DTYPE
export GPU_MEM_UTIL MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS EAGER VISIBLE_DEVICES
export DRAFT_MODEL SPEC_METHOD SPEC_TOKENS PROBE_LEVEL RUNS_DIR WARMUP_TIMEOUT
export RUNTIME HOST_PYTHON HOST_SERVE_ENTRYPOINT HOST_PRECHECK HOST_PRELUDE_FILES
export HOST_STOP_TIMEOUT HOST_VLLM_PATTERN PYSPY_BIN PROBE_SKIP_VRAM_WAIT

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

# ---------------------------------------------------------------------------
# 运行时抽象：docker（默认）| host（物理机裸跑）
#   RUNTIME=docker : 通过 docker run / docker exec 跑（原有行为）
#   RUNTIME=host   : 直接在物理机上起 vLLM（不经过容器）
# 别名：container→docker、bare/native→host；大小写不敏感。
# 本文件 source 时会立即 normalize 并 export；非法值直接 die。
# ---------------------------------------------------------------------------
runtime_normalize() {
  local r="${1:-${RUNTIME:-docker}}"
  r="$(printf '%s' "$r" | tr '[:upper:]' '[:lower:]')"
  case "$r" in
    ""|docker|container) printf 'docker' ;;
    host|bare|native)    printf 'host' ;;
    *) die "RUNTIME 非法：$r（可用：docker|host；别名 container→docker、bare/native→host）" ;;
  esac
}

# 注意：两个判定函数每次都读「当前」$RUNTIME（不缓存），便于脚本 source 后再改。
runtime_is_host()   { [[ "${RUNTIME:-docker}" == "host" ]]; }
runtime_is_docker() { [[ "${RUNTIME:-docker}" == "docker" ]]; }

# 校验当前运行时所需的宿主依赖（只应在脚本里调用；source 本文件时不执行，
# 因此没有 docker 的物理机也能 source 并跑 host 模式）。
runtime_require() {
  if runtime_is_host; then
    local py
    py="$(host_python)"
    if ! command -v "$py" >/dev/null 2>&1 && [[ ! -x "$py" ]]; then
      die "RUNTIME=host 需要可执行的 HOST_PYTHON：$py（在 .env 里设置 HOST_PYTHON，或把 python3 放进 PATH）"
    fi
    require_cmd curl
  else
    require_cmd docker
  fi
}

RUNTIME="$(runtime_normalize "${RUNTIME:-docker}")"
export RUNTIME

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

# 最新运行目录：优先 $RUN_DIR，否则 RUNS_DIR 下最新目录。
# 找不到时 return 1（**不要 die**）：本函数几乎都用在
#   `RUN_DIR="$(latest_run_dir 2>/dev/null || new_run_dir <tag>)"`
# 或 `if ! RUN_DIR="$(latest_run_dir 2>/dev/null)"; then ...` 里，
# 若在这里 die(exit) 只会退出命令替换的子 shell，调用方的 `||` 分支反而走不到，
# set -e 下脚本会静默退出（全新机器还没有 runs/ 目录时必现）。
latest_run_dir() {
  if [[ -n "$RUN_DIR" ]]; then echo "$RUN_DIR"; return 0; fi
  local latest
  latest="$(ls -1d "$RUNS_DIR"/*/ 2>/dev/null | sort | tail -1 || true)"
  [[ -n "$latest" ]] || return 1
  echo "${latest%/}"
}

# ---------------------------------------------------------------------------
# host（物理机裸跑）运行时辅助
#
# 产物约定（<RUN_DIR> 下）：
#   serve.pid        进程组组长 PID（setsid 起的会话 leader）
#   serve.pgid       进程组 ID（正常情况下与 serve.pid 相同）
#   serve_launch.sh 生成的启动脚本（cd → source host prelude → export 环境 → exec vLLM）
#   serve.log        stdout/stderr
# ---------------------------------------------------------------------------
host_python() {
  printf '%s' "${HOST_PYTHON:-python3}"
}

# host 模式的 vLLM 启动入口（与容器内 SERVE_ENTRYPOINT 等价，只是换成宿主机 python）
host_serve_entrypoint() {
  if [[ -n "${HOST_SERVE_ENTRYPOINT:-}" ]]; then
    printf '%s' "$HOST_SERVE_ENTRYPOINT"
  else
    printf "%s -c 'import torch; from vllm.entrypoints.cli.main import main; main()'" "$(host_python)"
  fi
}

# host 模式的 PYTHONPATH：源码 checkout + 工具包根（+ triton 安装目录 + 插件目录）
host_pythonpath() {
  local -a parts=()
  if [[ -n "${SRC_DIR:-}" ]]; then parts+=("$SRC_DIR"); fi
  parts+=("$PROJECT_ROOT")
  if [[ -n "${TRITON_INSTALL_DIR:-}" ]]; then parts+=("$TRITON_INSTALL_DIR"); fi
  if [[ "${PROBE_PLUGIN:-0}" == "1" && -n "${RUN_DIR:-}" ]]; then
    parts+=("$RUN_DIR/.probe_plugins")
  fi
  local IFS=':'
  printf '%s' "${parts[*]}"
}

# host 模式启动前 source 的宿主机环境脚本（空格分隔；不存在的文件静默跳过）
host_prelude_shell() {
  local out="" f q
  for f in ${HOST_PRELUDE_FILES:-}; do
    [[ -n "$f" ]] || continue
    q="$(printf '%q' "$f")"
    out+="if [ -f $q ]; then . $q >/dev/null 2>&1 || true; fi; "
  done
  printf '%s' "$out"
}

# 解析要操作的 run dir：显式参数 → $RUN_DIR → 最新 runs/* 目录；找不到返回空（不 die）
_host_resolve_run_dir() {
  local run_dir="${1:-${RUN_DIR:-}}"
  if [[ -z "$run_dir" ]]; then
    run_dir="$(ls -1d "$RUNS_DIR"/*/ 2>/dev/null | sort | tail -1 || true)"
    run_dir="${run_dir%/}"
  fi
  printf '%s' "$run_dir"
}

# host 实例 PID（读 <run_dir>/serve.pid；不存在/非法返回 1）
host_instance_pid() {
  local run_dir pid
  run_dir="$(_host_resolve_run_dir "${1:-}")"
  [[ -n "$run_dir" && -f "$run_dir/serve.pid" ]] || return 1
  pid="$(cat "$run_dir/serve.pid" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$pid"
}

# host 实例进程组 ID（优先 serve.pgid，缺失时回退 serve.pid）
host_instance_pgid() {
  local run_dir pgid pid
  run_dir="$(_host_resolve_run_dir "${1:-}")"
  [[ -n "$run_dir" ]] || return 1
  pgid=""
  if [[ -f "$run_dir/serve.pgid" ]]; then
    pgid="$(cat "$run_dir/serve.pgid" 2>/dev/null || true)"
  fi
  if [[ ! "$pgid" =~ ^[0-9]+$ ]]; then
    pid="$(host_instance_pid "$run_dir" 2>/dev/null || true)"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    pgid="$pid"
  fi
  printf '%s\n' "$pgid"
}

# host 实例是否在运行（只认 serve.pid，不会像 pgrep 那样误伤现网进程）
host_instance_running() {
  local pid
  pid="$(host_instance_pid "${1:-}" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

# host_start <run_dir> <log_file> <launcher>
# 用 setsid+nohup 在后台起 launcher，stdout/stderr 全量重定向到 log_file；
# 真实会话 leader PID 写入 serve.pid/serve.pgid（stop 时可按 -PGID 整组终止）。
host_start() {
  local run_dir="$1" log_file="$2" launcher="$3"
  [[ -n "$run_dir" && -n "$log_file" && -n "$launcher" ]] || die "host_start 参数不全：<run_dir> <log_file> <launcher>"
  command -v setsid >/dev/null 2>&1 || die "RUNTIME=host 需要 setsid 命令（util-linux）"
  mkdir -p "$run_dir" "$(dirname "$log_file")"
  rm -f "$run_dir/serve.pid" "$run_dir/serve.pgid"
  # setsid 在少数情况下会 fork；不直接记 $!，而是让会话 leader 自己把 $$ 落盘，
  # 再 exec launcher（exec 保持 PID，PGID == 该 PID）。
  setsid nohup bash -c 'printf "%s\n" "$$" > "$1/serve.pid"; exec bash "$2"' _ \
    "$run_dir" "$launcher" > "$log_file" 2>&1 < /dev/null &
  local waited pid=""
  for waited in $(seq 1 50); do
    if [[ -s "$run_dir/serve.pid" ]]; then
      pid="$(cat "$run_dir/serve.pid" 2>/dev/null || true)"
      if [[ "$pid" =~ ^[0-9]+$ ]]; then break; fi
    fi
    sleep 0.1
  done
  if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
    die "host 后台启动失败：$launcher（见日志 $log_file）"
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    die "host launcher 启动后立即退出（PID=$pid）；日志：$log_file"
  fi
  printf '%s\n' "$pid" > "$run_dir/serve.pgid"
  printf '%s\n' "$pid"
}

# host_stop [--run-dir D] [--timeout N] [--force]
# 先 SIGTERM 整个进程组，等待 timeout 秒仍存活则 SIGKILL；--force 直接 SIGKILL。
# 返回：0=已停止/本来就没在跑；1=没有 pid 记录；2=进程组仍在（需要人工确认）。
host_stop() {
  local run_dir="${RUN_DIR:-}" timeout="${HOST_STOP_TIMEOUT:-60}" force=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-dir) run_dir="${2:?}"; shift 2 ;;
      --timeout) timeout="${2:?}"; shift 2 ;;
      --force) force=1; shift ;;
      *) die "host_stop 未知参数：$1" ;;
    esac
  done
  run_dir="$(_host_resolve_run_dir "$run_dir")"
  local pid="" pgid="" target=""
  pid="$(host_instance_pid "$run_dir" 2>/dev/null || true)"
  pgid="$(host_instance_pgid "$run_dir" 2>/dev/null || true)"
  target="$pgid"
  [[ -n "$target" ]] || target="$pid"
  if [[ -z "$pid" && -z "$pgid" && -z "$run_dir" ]]; then
    warn "host 实例无 pid 记录（未指定 / 找不到运行目录）"
    return 1
  fi
  if [[ -z "$pid" && -z "$pgid" ]]; then
    warn "host 实例无 pid 记录：$run_dir/serve.pid（可能未用 host 模式启动）"
    return 1
  fi

  local alive=0
  if [[ -n "$target" ]] && kill -0 -- "-$target" 2>/dev/null; then
    alive=1
  elif [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    alive=1
  fi
  if (( ! alive )); then
    log "host 实例已不在运行（PID=${pid:-?} PGID=${target:-?}）"
    return 0
  fi

  if (( force )); then
    warn "host_stop --force：跳过 SIGTERM，直接 SIGKILL 进程组 PGID=$target"
  else
    log "终止 host 进程组 PGID=$target（SIGTERM，最多等待 ${timeout}s）"
    kill -TERM -- "-$target" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    local waited=0 still=1
    while (( waited < timeout )); do
      if ! kill -0 -- "-$target" 2>/dev/null; then still=0; break; fi
      sleep 1
      waited=$((waited + 1))
    done
    if (( still )); then
      warn "进程组 PGID=$target 在 ${timeout}s 内未退出，发送 SIGKILL"
    fi
  fi
  if kill -0 -- "-$target" 2>/dev/null; then
    kill -KILL -- "-$target" 2>/dev/null || true
    sleep 1
  fi
  if kill -0 -- "-$target" 2>/dev/null; then
    warn "host 进程组 PGID=$target 仍存活，请手动检查：ps -o pid,pgid,args -g $target"
    return 2
  fi
  ok "host 实例已停止（PGID=$target；pid 记录保留在 $run_dir）"
  return 0
}

# host_stop_pattern [pattern]
# 按进程名强杀宿主机上所有匹配 vLLM 的进程（等价容器 --kill-exec 的语义，慎用：
# 会误伤其它实验，包括不是本工具启动的 vLLM）。
host_stop_pattern() {
  local pattern="${1:-$HOST_VLLM_PATTERN}"
  local -a pids=() live=() remain=()
  mapfile -t pids < <(pgrep -f "$pattern" 2>/dev/null || true)
  local p
  for p in "${pids[@]+"${pids[@]}"}"; do
    [[ -n "$p" ]] || continue
    if [[ "$p" == "$$" || "$p" == "$PPID" ]]; then continue; fi
    live+=("$p")
  done
  if [[ ${#live[@]} -eq 0 ]]; then
    warn "本机没有匹配 vLLM 的进程（pattern: $pattern）"
    return 0
  fi
  warn "准备终止本机所有匹配 vLLM 的进程（pattern: $pattern）：${live[*]}"
  kill -TERM "${live[@]}" 2>/dev/null || true
  local waited=0 any=0
  for waited in $(seq 1 10); do
    any=0
    for p in "${live[@]}"; do
      if kill -0 "$p" 2>/dev/null; then any=1; fi
    done
    if (( ! any )); then break; fi
    sleep 1
  done
  for p in "${live[@]}"; do
    if kill -0 "$p" 2>/dev/null; then remain+=("$p"); fi
  done
  if [[ ${#remain[@]} -gt 0 ]]; then
    warn "以下进程未响应 SIGTERM，发送 SIGKILL：${remain[*]}"
    kill -KILL "${remain[@]}" 2>/dev/null || true
  fi
  ok "已按 pattern 终止 vLLM 进程（pattern: $pattern）"
  return 0
}

# host 上匹配 vLLM 的 PID（每行一个；无进程时输出为空并返回 0）
host_vllm_pids() {
  local p
  while IFS= read -r p; do
    [[ -n "$p" ]] || continue
    if [[ "$p" == "$$" || "$p" == "$PPID" ]]; then continue; fi
    printf '%s\n' "$p"
  done < <(pgrep -f "$HOST_VLLM_PATTERN" 2>/dev/null || true)
  return 0
}

# ---------------------------------------------------------------------------
# 跨运行时实例辅助（status/watch_steps/pyspy/probe_all 用）
# ---------------------------------------------------------------------------
# 当前实例的运行目录：优先 $RUN_DIR，否则 RUNS_DIR 下最新目录；找不到输出空并返回 1。
instance_run_dir() {
  local d
  d="$(_host_resolve_run_dir "${1:-}")"
  [[ -n "$d" ]] || return 1
  printf '%s\n' "$d"
}

# 当前实例是否在运行（host 看 serve.pid，docker 看容器状态）
instance_running() {
  if runtime_is_host; then
    host_instance_running "${1:-}"
  else
    container_running "$CONTAINER"
  fi
}

# 一行描述当前实例；有存活实例返回 0，否则返回 1（仍打印状态描述）。
instance_describe() {
  if runtime_is_host; then
    local pid pgid
    pid="$(host_instance_pid "${1:-}" 2>/dev/null || true)"
    pgid="$(host_instance_pgid "${1:-}" 2>/dev/null || true)"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      printf 'host（PID=%s PGID=%s）' "$pid" "${pgid:-$pid}"
      return 0
    fi
    printf 'host（未运行）'
    return 1
  fi
  if container_running "$CONTAINER"; then
    printf 'docker 容器=%s（运行中）' "$CONTAINER"
    return 0
  fi
  if container_exists "$CONTAINER"; then
    printf 'docker 容器=%s（已停止）' "$CONTAINER"
  else
    printf 'docker（无容器 %s）' "$CONTAINER"
  fi
  return 1
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
