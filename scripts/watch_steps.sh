#!/usr/bin/env bash
# 实时观察引擎逐 step 行为（默认读取运行目录里的 serve.log，docker 模式也可直接跟 docker logs）。
#
# 用法：
#   scripts/watch_steps.sh                      # 跟随最新运行目录的日志，实时打印 step
#   scripts/watch_steps.sh --tail 2000 --live   # 只回放最后 2000 条事件后退出
#   scripts/watch_steps.sh --summary            # 只输出统计摘要（不跟随）
#   scripts/watch_steps.sh --docker             # 改用 docker logs -f（仅 RUNTIME=docker）
#   RUNTIME=host scripts/watch_steps.sh         # 物理机模式：默认文件模式（<RUN_DIR>/serve.log）
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
    --runtime)
      # runtime_normalize 只打印规范化结果（container→docker、bare/native→host），非法值 die
      RUNTIME="$(runtime_normalize "${2:?}")"
      export RUNTIME
      shift 2
      ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

# --docker 走容器日志，物理机模式没有容器：直接给出可操作提示，别让用户等半天。
if [[ "$MODE_SRC" == "docker" ]] && runtime_is_host; then
  _rd="$(instance_run_dir 2>/dev/null || true)"
  [[ -n "$_rd" ]] || _rd="$RUNS_DIR/<最新运行目录>"
  die "RUNTIME=host 请用默认文件模式（$_rd/serve.log）；物理机模式没有容器日志可跟"
fi

# latest_run_dir 找不到目录时返回 1（不是 die），这里给出可操作的提示
if ! RUN_DIR="$(latest_run_dir 2>/dev/null)"; then
  die "找不到运行目录（$RUNS_DIR）；先 scripts/start_vllm.sh，或用 --run-dir 指定已有运行目录"
fi
LOG_FILE="${LOG_FILE:-$RUN_DIR/serve.log}"

if [[ "$MODE_SRC" == "docker" ]]; then
  runtime_require
  container_running "$CONTAINER" || die "容器 $CONTAINER 未运行"
  log "跟随 docker logs -f $CONTAINER（Ctrl-C 结束）"
  docker logs -f --since 1m "$CONTAINER" 2>&1 | py -m probe logs --stdin --live
  exit 0
fi

[[ -f "$LOG_FILE" ]] || die "日志不存在：$LOG_FILE（先执行 scripts/start_vllm.sh；host 模式确认已写 serve.log）"

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
