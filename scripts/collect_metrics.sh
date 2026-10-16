#!/usr/bin/env bash
# 采集 /metrics 快照，并支持前后两次快照做差。
#
# 用法：
#   scripts/collect_metrics.sh                       # 存到 <RUN_DIR>/metrics_<tag>.json
#   scripts/collect_metrics.sh --tag before
#   scripts/collect_metrics.sh --diff before after --elapsed 60
#   scripts/collect_metrics.sh --watch 5             # 每 5 秒打印一次阶段表（Ctrl-C 结束）

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

TAG="metrics"
DIFF=0
BEFORE=""; AFTER=""; ELAPSED=""
WATCH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="${2:?}"; shift 2 ;;
    --diff) DIFF=1; BEFORE="${2:?}"; AFTER="${3:?}"; shift 3 ;;
    --elapsed) ELAPSED="${2:?}"; shift 2 ;;
    --watch) WATCH="${2:?}"; shift 2 ;;
    --run-dir) RUN_DIR="${2:?}"; shift 2 ;;
    --port) PORT="${2:?}"; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done
RUN_DIR="$(latest_run_dir 2>/dev/null || new_run_dir metrics)"
mkdir -p "$RUN_DIR"

if (( WATCH )); then
  log "每 ${WATCH}s 刷新阶段表（Ctrl-C 结束）"
  while :; do
    clear
    date "+%F %T"
    py -m probe metrics --port "$PORT" --outdir "$RUN_DIR" --tag "metrics_watch" 2>/dev/null || warn "抓取失败"
    sleep "$WATCH"
  done
fi

if (( DIFF )); then
  [[ -f "$BEFORE" ]] || die "找不到快照：$BEFORE"
  [[ -f "$AFTER" ]] || die "找不到快照：$AFTER"
  [[ -n "$ELAPSED" ]] || ELAPSED="$(
    py - "$BEFORE" "$AFTER" <<'PY'
import json, sys
a = json.load(open(sys.argv[1])).get("meta", {})
b = json.load(open(sys.argv[2])).get("meta", {})
print(max(1.0, float(b.get("fetched_at", 0)) - float(a.get("fetched_at", 0))))
PY
  )"
  log "做差：$BEFORE → $AFTER（窗口 ${ELAPSED}s）"
  py -m probe metrics --outdir "$RUN_DIR" --diff "$BEFORE" "$AFTER" --elapsed "$ELAPSED" --print
  exit 0
fi

log "采集指标快照 tag=$TAG（端口 $PORT）"
py -m probe metrics --port "$PORT" --outdir "$RUN_DIR" --tag "$TAG"
ok "快照已保存到 $RUN_DIR/$TAG.json（同时生成 $TAG.tsv）"
