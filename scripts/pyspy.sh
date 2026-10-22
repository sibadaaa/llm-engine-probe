#!/usr/bin/env bash
# Python 栈采样：用 py-spy 抓火焰图 / 栈快照。
# docker 模式在容器内执行 py-spy；host 模式（RUNTIME=host）直接对宿主机上的 vLLM 进程执行。
# 用途：当 GPU 时间线出现空洞、或 step 时间抖动（双稳态）时，确认 CPU 侧卡在哪里。
#
# 用法：
#   scripts/pyspy.sh --mode dump                 # 所有 vllm 进程的栈快照
#   scripts/pyspy.sh --mode flame --duration 30  # 30 秒火焰图（SVG + speedscope JSON）
#   scripts/pyspy.sh --mode flame --pid 12345    # 指定 pid
#   scripts/pyspy.sh --mode flame --subprocesses # 跟随子进程（含 worker）
#   RUNTIME=host scripts/pyspy.sh --mode dump    # 物理机模式（需本机已装 py-spy）
#
# 产物：一律落在 <RUN_DIR>/pyspy/（docker 模式的 /logs 就是该目录的挂载点）。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

PYSPY_MODE="${PYSPY_MODE:-dump}"
DURATION="${PYSPY_DURATION:-30}"
RATE="${PYSPY_RATE:-100}"
TARGET_PID=""
SUBPROCESSES=0
NATIVE=0
IDLE=0
CONTAINER_GIVEN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) PYSPY_MODE="${2:?}"; shift 2 ;;
    --duration) DURATION="${2:?}"; shift 2 ;;
    --rate) RATE="${2:?}"; shift 2 ;;
    --pid) TARGET_PID="${2:?}"; shift 2 ;;
    --subprocesses) SUBPROCESSES=1; shift ;;
    --native) NATIVE=1; shift ;;
    --idle) IDLE=1; shift ;;
    --run-dir) RUN_DIR="${2:?}"; shift 2 ;;
    --container) CONTAINER="${2:?}"; CONTAINER_GIVEN=1; shift 2 ;;
    --runtime)
      # runtime_normalize 只打印规范化结果（container→docker、bare/native→host），非法值 die
      RUNTIME="$(runtime_normalize "${2:?}")"
      export RUNTIME
      shift 2
      ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done
case "$PYSPY_MODE" in dump|flame|top) ;; *) die "未知 --mode：$PYSPY_MODE（dump|flame|top）" ;; esac

# 引擎组件：加载 profile（进程查找正则走 engine_process_pattern 优先级链）
engine_meta

PYSPY_BIN="${PYSPY_BIN:-py-spy}"
# latest_run_dir 找不到目录时返回 1（不 die），据此回落去新建一个运行目录
if ! RUN_DIR="$(latest_run_dir 2>/dev/null)"; then
  RUN_DIR="$(new_run_dir pyspy)"
fi
mkdir -p "$RUN_DIR/pyspy"
PROC_FILE="$RUN_DIR/pyspy/processes.txt"

FLAGS=()
(( SUBPROCESSES )) && FLAGS+=(--subprocesses)
(( NATIVE )) && FLAGS+=(--native)
(( IDLE )) && FLAGS+=(--idle)
TS="$(date +%H%M%S)"

PIDS=()
FAILED=0

# ---------------------------------------------------------------------------
# 宿主机（物理机）分支：直接对 host_vllm_pids 找到的进程调用 py-spy
# ---------------------------------------------------------------------------
dump_host() {
  local pid out
  for pid in "${PIDS[@]}"; do
    out="$RUN_DIR/pyspy/pyspy_dump_${pid}_${TS}.txt"
    log "py-spy dump --pid $pid -> $out"
    if ! "$PYSPY_BIN" dump --pid "$pid" ${FLAGS[@]+"${FLAGS[@]}"} > "$out" 2>&1; then
      echo "py-spy dump 失败（检查 ptrace 权限：cat /proc/sys/kernel/yama/ptrace_scope）" >> "$out"
      warn "py-spy dump 失败：pid=$pid（详见 $out）"
      FAILED=1
    fi
    sed -n '1,25p' "$out"
    echo "  ..."
  done
  return 0
}

flame_host() {
  local pid json svg
  for pid in "${PIDS[@]}"; do
    json="$RUN_DIR/pyspy/pyspy_flame_${pid}_${TS}.json"
    svg="$RUN_DIR/pyspy/pyspy_flame_${pid}_${TS}.svg"
    log "py-spy record --pid $pid --duration $DURATION --rate $RATE -> $svg（+ speedscope $json）"
    if ! "$PYSPY_BIN" record --pid "$pid" --duration "$DURATION" --rate "$RATE" \
          --format speedscope --output "$json" ${FLAGS[@]+"${FLAGS[@]}"} >/dev/null 2>&1; then
      warn "py-spy record（speedscope）失败：pid=$pid（检查 ptrace 权限）"
      FAILED=1
    fi
    if ! "$PYSPY_BIN" record --pid "$pid" --duration "$DURATION" --rate "$RATE" \
          --output "$svg" ${FLAGS[@]+"${FLAGS[@]}"} 2>&1 | tail -3; then
      warn "py-spy record（svg）失败：pid=$pid（检查 ptrace 权限）"
      FAILED=1
    fi
  done
  return 0
}

top_host() {
  log "py-spy top（交互，Ctrl-C 结束）"
  "$PYSPY_BIN" top --pid "${PIDS[0]}" ${FLAGS[@]+"${FLAGS[@]}"}
}

if runtime_is_host; then
  runtime_require
  if (( CONTAINER_GIVEN )); then
    warn "--container 仅对 docker 模式生效；RUNTIME=host 已忽略（$CONTAINER）"
  fi

  # 注意：不要调用 host_stop_pattern——它是「按 pattern 强杀进程」的动作用函数，
  # 这里只需要展示/匹配 pattern 本身，用 engine_process_pattern（优先级链）。
  log "查找宿主机上的引擎进程（pattern：$(engine_process_pattern)）"
  {
    printf '%-8s %-10s %s\n' "PID" "ELAPSED" "ARGS"
    while IFS= read -r pid; do
      [[ -n "$pid" ]] || continue
      ps -o pid=,etime=,args= -p "$pid" 2>/dev/null | sed 's/^[[:space:]]*//' || true
    done < <(host_vllm_pids)
  } > "$PROC_FILE"
  cut -c1-160 "$PROC_FILE"

  if [[ -n "$TARGET_PID" ]]; then
    PIDS=("$TARGET_PID")
  else
    mapfile -t PIDS < <(awk 'NR > 1 && $1 ~ /^[0-9]+$/ {print $1}' "$PROC_FILE" | sort -u | head -8)
  fi
  [[ ${#PIDS[@]} -gt 0 ]] || die "未找到宿主机上的 vLLM 进程；宿主机是否在跑 vLLM？或权限不足（py-spy 需要 ptrace 权限：cat /proc/sys/kernel/yama/ptrace_scope）"

  require_cmd "$PYSPY_BIN"
  case "$PYSPY_MODE" in
    dump) dump_host ;;
    flame) flame_host ;;
    top) top_host ;;
  esac

  ok "产物：$RUN_DIR/pyspy/"
  (( FAILED )) && warn "部分 py-spy 采样失败，详见上方日志（常见原因是 ptrace 权限或进程已退出）"
  ls -la "$RUN_DIR/pyspy" | tail -10
  exit 0
fi

# ---------------------------------------------------------------------------
# docker 分支：保持原有行为（在容器内执行，产物经 /logs 落到 <RUN_DIR>）
# ---------------------------------------------------------------------------
runtime_require
container_running "$CONTAINER" || die "容器 $CONTAINER 未运行"

log "查找容器内的引擎进程"
docker exec "$CONTAINER" bash -lc "ps -eo pid,etime,args | grep -E '$(engine_process_pattern)' | grep -v grep" > "$PROC_FILE" || true
cut -c1-160 "$PROC_FILE"

if [[ -n "$TARGET_PID" ]]; then
  PIDS=("$TARGET_PID")
else
  mapfile -t PIDS < <(awk '{print $1}' "$PROC_FILE" 2>/dev/null | sort -u | head -8)
fi
[[ ${#PIDS[@]} -gt 0 ]] || die "未找到目标进程；容器内是否在跑 vLLM？"

case "$PYSPY_MODE" in
  dump)
    for pid in "${PIDS[@]}"; do
      out="/logs/pyspy/pyspy_dump_${pid}_${TS}.txt"
      log "py-spy dump --pid $pid -> $out"
      docker exec "$CONTAINER" bash -lc "py-spy dump --pid $pid ${FLAGS[*]:-} ${NATIVE:+} > '$out' 2>&1 || echo 'py-spy dump 失败（可能需要 --cap-add SYS_PTRACE）' >> '$out'"
      docker exec "$CONTAINER" bash -lc "sed -n '1,25p' '$out'"
      echo "  ..."
    done
    ;;
  flame)
    for pid in "${PIDS[@]}"; do
      out="/logs/pyspy/pyspy_flame_${pid}_${TS}.svg"
      log "py-spy record --pid $pid --duration $DURATION --rate $RATE -> $out"
      docker exec "$CONTAINER" bash -lc \
        "py-spy record --pid $pid --duration $DURATION --rate $RATE --format speedscope --output '${out%.svg}.json' ${FLAGS[*]:-} >/dev/null 2>&1; \
         py-spy record --pid $pid --duration $DURATION --rate $RATE --output '$out' ${FLAGS[*]:-} 2>&1 | tail -3"
    done
    ;;
  top)
    log "py-spy top（交互，Ctrl-C 结束）"
    docker exec -it "$CONTAINER" bash -lc "py-spy top --pid ${PIDS[0]} ${FLAGS[*]:-}"
    ;;
esac

ok "产物：$RUN_DIR/pyspy/"
ls -la "$RUN_DIR/pyspy" | tail -10
