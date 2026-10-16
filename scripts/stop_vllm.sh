#!/usr/bin/env bash
# 停止探测实例。
#
# 用法：
#   scripts/stop_vllm.sh              # 停止容器的 PID 1（优雅 SIGTERM，60s 超时）
#   scripts/stop_vllm.sh --rm         # 停止并删除容器
#   scripts/stop_vllm.sh --kill-exec  # 在已有容器里按进程名强杀（MODE=exec 场景）
#
# 注意：--kill-exec 会杀掉容器内所有匹配 vllm 的进程，请先确认不会误伤其它实验。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

REMOVE=0
KILL_EXEC=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rm) REMOVE=1; shift ;;
    --kill-exec) KILL_EXEC=1; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

require_cmd docker

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
  log "等待显存释放（${VENDOR_SMI_CMD:-厂商 SMI} 采样）"
  vram_ok=0
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
