#!/usr/bin/env bash
# 请求级探测：在发压前后各抓一次 /metrics，并对每个请求逐 chunk 打点。
#
# 用法：
#   scripts/probe_http.sh --prompt-tokens 512 --max-tokens 128 --concurrency 1 --repeat 5
#   scripts/probe_http.sh --shared-prefix-tokens 2048     # 探测前缀缓存命中
#   scripts/probe_http.sh --no-metrics                    # 跳过 /metrics 前后快照
#
# 产物（都在 <RUN_DIR>）：
#   http_requests.jsonl   逐请求明细（含服务端返回的 metrics 字段）
#   http_summary.json     分位数汇总
#   metrics_before.json / metrics_after.json / metrics_diff.json

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

WITH_METRICS=1
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-metrics) WITH_METRICS=0; shift ;;
    --run-dir) RUN_DIR="${2:?}"; shift 2 ;;
    --port) PORT="${2:?}"; shift 2 ;;
    --prompt-tokens) PROMPT_TOKENS="${2:?}"; shift 2 ;;
    --max-tokens) MAX_TOKENS="${2:?}"; shift 2 ;;
    --concurrency) CONCURRENCY="${2:?}"; shift 2 ;;
    --repeat) REPEAT="${2:?}"; shift 2 ;;
    --shared-prefix-tokens) SHARED_PREFIX_TOKENS="${2:?}"; shift 2 ;;
    --endpoint) EXTRA+=(--endpoint "${2:?}"); shift 2 ;;
    --no-stream) EXTRA+=(--no-stream); shift ;;
    --warmup) EXTRA+=(--warmup "${2:?}"); shift 2 ;;
    --extra) EXTRA+=("${2:?}"); shift 2 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

RUN_DIR="$(latest_run_dir 2>/dev/null || new_run_dir http)"
mkdir -p "$RUN_DIR"
log "运行目录：$RUN_DIR"

if (( WITH_METRICS )); then
  log "抓取基线指标 metrics_before"
  py -m probe metrics --port "$PORT" --outdir "$RUN_DIR" --tag metrics_before >/dev/null
fi

START_EPOCH=$(date +%s)
log "开始请求级探测：prompt≈${PROMPT_TOKENS} max_tokens=${MAX_TOKENS} 并发=${CONCURRENCY}×${REPEAT}"
set +e
py -m probe http \
  --port "$PORT" --outdir "$RUN_DIR" \
  --prompt-tokens "$PROMPT_TOKENS" --max-tokens "$MAX_TOKENS" \
  --concurrency "$CONCURRENCY" --repeat "$REPEAT" \
  --shared-prefix-tokens "$SHARED_PREFIX_TOKENS" \
  "${EXTRA[@]}"
RC=$?
set -e
ELAPSED=$(( $(date +%s) - START_EPOCH ))

if (( WITH_METRICS )); then
  log "抓取结束指标 metrics_after（窗口 ${ELAPSED}s）"
  py -m probe metrics --port "$PORT" --outdir "$RUN_DIR" --tag metrics_after >/dev/null
  py -m probe metrics --outdir "$RUN_DIR" \
    --diff "$RUN_DIR/metrics_before.json" "$RUN_DIR/metrics_after.json" --elapsed "$ELAPSED"
fi

if (( RC != 0 )); then
  warn "部分请求失败（退出码 $RC），请看 $RUN_DIR/http_summary.json 里的 errors"
fi
ok "请求级探测完成：$RUN_DIR/http_summary.json"
