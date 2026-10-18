#!/usr/bin/env bash
# 生成中文 Markdown 报告（把运行目录里的所有产物汇总成一份文档）。
#
# 用法：
#   scripts/report.sh                     # 用最近一次运行目录
#   scripts/report.sh --run-dir runs/xxx
#   scripts/report.sh --open              # 生成后用 less 打开

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

OPEN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="${2:?}"; shift 2 ;;
    --open) OPEN=1; shift ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

# latest_run_dir 找不到目录时返回 1（不是 die），这里给出可操作的提示
if ! RUN_DIR="$(latest_run_dir 2>/dev/null)"; then
  die "找不到运行目录（$RUNS_DIR）；先 scripts/start_vllm.sh，或用 --run-dir 指定已有运行目录"
fi
log "汇总目录：$RUN_DIR"
py -m probe report --outdir "$RUN_DIR"
if (( OPEN )); then
  less "$RUN_DIR/REPORT.md"
fi
