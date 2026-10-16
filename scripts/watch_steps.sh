#!/usr/bin/env bash
# 实时观察引擎逐 step 行为（读取运行目录里的 serve.log，或直接跟 docker logs）。
#
# 用法：
#   scripts/watch_steps.sh                      # 跟随最新运行目录的日志，实时打印 step
#   scripts/watch_steps.sh --tail 2000 --live   # 只回放最后 2000 条事件后退出
#   scripts/watch_steps.sh --summary            # 只输出统计摘要（不跟随）
#   scripts/watch_steps.sh --docker             # 改用 docker logs -f
#
# 依赖启动时打开 `--enable-logging-iteration-details`（PROBE_LEVEL>=steps）。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

MODE_SRC="file"
TAIL=""
SUMMARY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --file) LOG_FILE="${2:?}"; shift 2 ;;
    --run-dir) RUN_DIR="${2:?}"; shift 2 ;;
    --tail) TAIL="${2:?}"; shift 2 ;;
    --summary) SUMMARY=1; shift ;;
    --docker) MODE_SRC="docker"; shift ;;
    --container) CONTAINER="${2:?}"; shift 2 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

RUN_DIR="$(latest_run_dir)"
LOG_FILE="${LOG_FILE:-$RUN_DIR/serve.log}"

if [[ "$MODE_SRC" == "docker" ]]; then
  container_running "$CONTAINER" || die "容器 $CONTAINER 未运行"
  log "跟随 docker logs -f $CONTAINER（Ctrl-C 结束）"
  docker logs -f --since 1m "$CONTAINER" 2>&1 | py -m probe logs --stdin --live
  exit 0
fi

[[ -f "$LOG_FILE" ]] || die "日志不存在：$LOG_FILE（先执行 scripts/start_vllm.sh）"

if (( SUMMARY )); then
  log "汇总 $LOG_FILE"
  py -m probe logs --file "$LOG_FILE" $([[ -n "$TAIL" ]] && echo "--tail $TAIL") \
    --jsonl "$RUN_DIR/steps.jsonl" --summary "$RUN_DIR/steps_summary.json"
  exit 0
fi

if [[ -n "$TAIL" ]]; then
  log "回放 $LOG_FILE 的最后 $TAIL 条事件"
  py -m probe logs --file "$LOG_FILE" --tail "$TAIL" --live --live-tail "$TAIL"
  exit 0
fi

log "实时跟随 $LOG_FILE（Ctrl-C 结束；同时会在结束时汇总）"
set +e
py -m probe logs --file "$LOG_FILE" --follow --live --jsonl "$RUN_DIR/steps_live.jsonl"
set -e
log "已停止跟随；生成汇总"
py -m probe logs --file "$LOG_FILE" --jsonl "$RUN_DIR/steps.jsonl" --summary "$RUN_DIR/steps_summary.json"
