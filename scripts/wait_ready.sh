#!/usr/bin/env bash
# 等待 vLLM 实例就绪（/health 200 且 /metrics 出现 vllm:* 指标）。
#
# 用法：scripts/wait_ready.sh [--port 8100] [--timeout 900] [--quiet]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

TIMEOUT="$WARMUP_TIMEOUT"
QUIET=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="${2:?}"; shift 2 ;;
    --timeout) TIMEOUT="${2:?}"; shift 2 ;;
    --quiet) QUIET=1; shift ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

BASE="http://127.0.0.1:$PORT"
started=$(date +%s)
(( QUIET )) || log "等待 $BASE 就绪（最多 ${TIMEOUT}s）"

while :; do
  code="$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$BASE/health" 2>/dev/null || echo 000)"
  if [[ "$code" == "200" ]]; then
    if curl -s -m 5 "$BASE/metrics" 2>/dev/null | grep -qE "$VENDOR_READY_METRIC"; then
      elapsed=$(( $(date +%s) - started ))
      (( QUIET )) || ok "引擎就绪，耗时 ${elapsed}s（厂商=$VENDOR，就绪指标=${VENDOR_READY_METRIC}）"
      exit 0
    fi
  fi
  now=$(date +%s)
  if (( now - started > TIMEOUT )); then
    (( QUIET )) || warn "超时（${TIMEOUT}s），/health=$code"
    exit 1
  fi
  sleep 5
done
