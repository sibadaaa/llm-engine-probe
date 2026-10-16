#!/usr/bin/env bash
# Python 栈采样：在容器内用 py-spy 抓火焰图 / 栈快照。
# 用途：当 GPU 时间线出现空洞、或 step 时间抖动（双稳态）时，确认 CPU 侧卡在哪里。
#
# 用法：
#   scripts/pyspy.sh --mode dump                 # 所有 vllm 进程的栈快照
#   scripts/pyspy.sh --mode flame --duration 30  # 30 秒火焰图（SVG）
#   scripts/pyspy.sh --mode flame --pid 12345    # 指定 pid
#   scripts/pyspy.sh --mode flame --subprocesses # 跟随子进程（含 worker）
#
# 产物：<RUN_DIR>/pyspy_*.svg / pyspy_dump_*.txt

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
    --container) CONTAINER="${2:?}"; shift 2 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

require_cmd docker
container_running "$CONTAINER" || die "容器 $CONTAINER 未运行"

RUN_DIR="$(latest_run_dir 2>/dev/null || new_run_dir pyspy)"
mkdir -p "$RUN_DIR/pyspy"

log "查找容器内的 vLLM 进程"
docker exec "$CONTAINER" bash -lc "ps -eo pid,etime,args | grep -E 'vllm|EngineCore|APIServer' | grep -v grep" > "$RUN_DIR/pyspy/processes.txt" || true
cat "$RUN_DIR/pyspy/processes.txt" | cut -c1-160

PIDS=()
if [[ -n "$TARGET_PID" ]]; then
  PIDS=("$TARGET_PID")
else
  mapfile -t PIDS < <(awk '{print $1}' "$RUN_DIR/pyspy/processes.txt" 2>/dev/null | sort -u | head -8)
fi
[[ ${#PIDS[@]} -gt 0 ]] || die "未找到目标进程；容器内是否在跑 vLLM？"

FLAGS=()
(( SUBPROCESSES )) && FLAGS+=(--subprocesses)
(( NATIVE )) && FLAGS+=(--native)
(( IDLE )) && FLAGS+=(--idle)

TS="$(date +%H%M%S)"
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
  *) die "未知 --mode：$PYSPY_MODE（dump|flame|top）" ;;
esac

ok "产物：$RUN_DIR/pyspy/"
ls -la "$RUN_DIR/pyspy" | tail -10
