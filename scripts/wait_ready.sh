#!/usr/bin/env bash
# 等待实例就绪（/health 200 且 /metrics 出现就绪指标）。
# 引擎感知（F3）：就绪判定统一走 common.sh::wait_engine_ready（与 start_vllm.sh 共用）——
# 按引擎 caps 协商：无 api.health 的引擎退化端口探测、无 /metrics 面跳过指标等待、
# 抓取带 ENGINE_METRICS_FETCH_SUFFIX、就绪正则按 ENGINE_READY_METRIC > VENDOR_READY_METRIC。
#
# 用法：scripts/wait_ready.sh [--port 8100] [--timeout 900] [--quiet]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

# 引擎组件：加载 profile（未知 ENGINE die；ENGINE_* 供 wait_engine_ready 使用）
engine_meta

TIMEOUT="$WARMUP_TIMEOUT"
QUIET=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="${2:?}"; shift 2 ;;
    --timeout) TIMEOUT="${2:?}"; shift 2 ;;
    --quiet) QUIET=1; shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

BASE="http://127.0.0.1:$PORT"
started=$(date +%s)
(( QUIET )) || log "等待 $BASE 就绪（最多 ${TIMEOUT}s；引擎=$ENGINE，厂商=$VENDOR）"

if wait_engine_ready "$BASE" "$TIMEOUT"; then
  elapsed=$(( $(date +%s) - started ))
  (( QUIET )) || ok "引擎就绪，耗时 ${elapsed}s（引擎=$ENGINE，厂商=$VENDOR，就绪指标=$(engine_ready_metric)）"
  exit 0
fi
(( QUIET )) || warn "超时（${TIMEOUT}s），实例未就绪（/health 或端口连通性探测未通过）"
exit 1
