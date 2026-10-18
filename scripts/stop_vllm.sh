#!/usr/bin/env bash
# 停止探测实例（docker 或 host 物理机裸跑，由 RUNTIME 决定；也可 --runtime 临时指定）。
#
# 用法：
#   scripts/stop_vllm.sh              # docker：停止容器 PID 1（SIGTERM，60s 超时）
#                                     # host  ：SIGTERM 整个进程组（HOST_STOP_TIMEOUT 秒后 SIGKILL）
#   scripts/stop_vllm.sh --rm         # docker：停止并删除容器；host：无需删除（保留 pid 记录）
#   scripts/stop_vllm.sh --kill-exec  # 按进程名强杀（docker=容器内；host=本机 pgrep pattern）
#   scripts/stop_vllm.sh --run-dir D  # host：指定要停的 run 目录（默认 $RUN_DIR/最新 runs/*）
#   scripts/stop_vllm.sh --timeout N  # host：SIGTERM 后等待秒数（默认 HOST_STOP_TIMEOUT=60）
#   scripts/stop_vllm.sh --no-vram-wait   # 停止后跳过显存释放等待（等价 PROBE_SKIP_VRAM_WAIT=1）
#   scripts/stop_vllm.sh --runtime host   # 临时指定运行时（docker|host，别名可用）
#
# 注意：--kill-exec 会杀掉所有匹配 vllm 的进程（host 下是本机全机！），请先确认不会误伤其它实验。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

REMOVE=0
KILL_EXEC=0
STOP_TIMEOUT="${HOST_STOP_TIMEOUT:-60}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rm) REMOVE=1; shift ;;
    --kill-exec) KILL_EXEC=1; shift ;;
    --run-dir) RUN_DIR="${2:?}"; shift 2 ;;
    --timeout) STOP_TIMEOUT="${2:?}"; shift 2 ;;
    --no-vram-wait) PROBE_SKIP_VRAM_WAIT=1; shift ;;
    --host) RUNTIME=host; shift ;;
    --docker) RUNTIME=docker; shift ;;
    --runtime) RUNTIME="${2:?}"; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

RUNTIME="$(runtime_normalize "${RUNTIME:-docker}")"
export RUNTIME
runtime_require

# 等待显存释放（docker --kill-exec 与 host 停止共用；无 SMI 的机器直接跳过）
# PROBE_SKIP_VRAM_WAIT=1 / --no-vram-wait 时跳过（测试/CI/同机已有其它显存占用）。
wait_vram_release() {
  if [[ "${PROBE_SKIP_VRAM_WAIT:-0}" == "1" ]]; then
    log "跳过显存释放等待（PROBE_SKIP_VRAM_WAIT=1）"
    return 0
  fi
  log "等待显存释放（${VENDOR_SMI_CMD:-厂商 SMI} 采样）"
  local vram_ok=0 local_sum i
  for i in $(seq 1 20); do
    sleep 3
    local_sum="$(py -m probe telemetry --vram-sum 2>/dev/null || echo none)"
    if [[ "$local_sum" == "none" ]]; then
      warn "无法从本机 SMI 读取显存占用（厂商 $VENDOR），跳过等待"
      vram_ok=1
      break
    fi
    log "  VRAM 合计 ${local_sum}%"
    if awk -v v="$local_sum" 'BEGIN { exit !(v < 20) }'; then ok "显存已释放"; vram_ok=1; break; fi
  done
  (( vram_ok )) || warn "20 次采样后显存仍未见明显回落，请手动用本机 SMI（${VENDOR_SMI_CMD:-如 hy-smi/nvidia-smi}）确认"
}

# ---------------------------------------------------------------------------
# host（物理机裸跑）：按 serve.pgid/serve.pid 终止整个进程组
# ---------------------------------------------------------------------------
if runtime_is_host; then
  if (( KILL_EXEC )); then
    warn "RUNTIME=host：--kill-exec 等价于按 pattern 杀本机所有匹配 vLLM 的进程（会误伤其它实验，请确认）"
    host_stop_pattern
    wait_vram_release
    exit 0
  fi

  HOST_RUN_DIR="$(instance_run_dir 2>/dev/null || true)"
  if [[ -z "$HOST_RUN_DIR" ]]; then
    warn "找不到运行目录（$RUNS_DIR）；若实例用 --run-dir 启动，请用 --run-dir 指定"
    exit 0
  fi

  host_rc=0
  WAS_RUNNING=0
  if host_instance_running "$HOST_RUN_DIR"; then WAS_RUNNING=1; fi
  host_stop --run-dir "$HOST_RUN_DIR" --timeout "$STOP_TIMEOUT" || host_rc=$?
  case "$host_rc" in
    0)
      ok "host 实例已停止（run dir: $HOST_RUN_DIR）"
      ;;
    1)
      # host_stop 已在内部 warn「无 pid 记录」，这里按契约视为正常退出
      log "没有可停止的 host 实例（run dir: $HOST_RUN_DIR）"
      exit 0
      ;;
    *)
      warn "host 实例可能仍在运行，请手动确认（run dir: $HOST_RUN_DIR）"
      exit 1
      ;;
  esac

  # 日志快照
  stop_log_follower "$HOST_RUN_DIR/serve.log"
  if [[ -f "$HOST_RUN_DIR/serve.log" ]]; then
    log "日志快照：$HOST_RUN_DIR/serve.log（$(wc -l < "$HOST_RUN_DIR/serve.log") 行）"
  fi

  if (( REMOVE )); then
    log "RUNTIME=host：--rm 无需删除容器；保留 pid 记录与 run dir（$HOST_RUN_DIR）"
    log "如需彻底清理：rm -rf $HOST_RUN_DIR"
  fi

  # 只有「本次确实停掉了一个活实例」才等显存释放；本来就没在跑则跳过，避免无谓等待
  if (( WAS_RUNNING )); then
    wait_vram_release
  else
    log "跳过显存释放等待（停止前 host 实例已不在运行）"
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
# docker（原有行为）
# ---------------------------------------------------------------------------
if (( KILL_EXEC )); then
  container_running "$EXEC_CONTAINER" || die "容器 $EXEC_CONTAINER 未运行"
  pattern='vllm.entrypoints.cli.main|vllm\.v1\.engine|s19_|serve_'
  log "在容器 $EXEC_CONTAINER 内终止 vLLM 进程（pattern: $pattern）"
  docker exec "$EXEC_CONTAINER" bash -lc "
    for p in \$(ps -eo pid,args | grep -E '$pattern' | grep -v grep | awk '{print \$1}'); do
      echo \"  kill -TERM \$p\"; kill -TERM \$p 2>/dev/null || true
    done
    sleep 8
    for p in \$(ps -eo pid,args | grep -E '$pattern' | grep -v grep | awk '{print \$1}'); do
      echo \"  kill -9 \$p\"; kill -9 \$p 2>/dev/null || true
    done
  "
  wait_vram_release
  exit 0
fi

if [[ "$MODE" == "exec" ]]; then
  log "MODE=exec：请使用 --kill-exec 终止已有容器内的进程"
  exit 0
fi

if ! container_exists "$CONTAINER"; then
  warn "容器 $CONTAINER 不存在"
  exit 0
fi
if ! container_running "$CONTAINER"; then
  log "容器 $CONTAINER 已停止"
else
  log "停止容器 $CONTAINER（等待引擎优雅退出，最多 60s）"
  docker stop -t 60 "$CONTAINER" >/dev/null || warn "docker stop 返回非零，继续"
  ok "容器已停止"
fi

# 收集最后的日志快照
RUN_DIR_LATEST="$(latest_run_dir 2>/dev/null || true)"
if [[ -n "${RUN_DIR_LATEST:-}" ]]; then
  stop_log_follower "$RUN_DIR_LATEST/serve.log"
  if [[ -f "$RUN_DIR_LATEST/serve.log" ]]; then
    log "日志快照：$RUN_DIR_LATEST/serve.log（$(wc -l < "$RUN_DIR_LATEST/serve.log") 行）"
  fi
fi

if (( REMOVE )); then
  docker rm "$CONTAINER" >/dev/null && ok "已删除容器 $CONTAINER"
fi
