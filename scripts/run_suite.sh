#!/usr/bin/env bash
# 标准化对比测试套件：固定 8 场景矩阵 → suite_results.json（可跨厂商对比）。
#
# 用法：
#   scripts/run_suite.sh                       # 跑全部场景（写入最新运行目录）
#   scripts/run_suite.sh --scenarios s02,s05   # 只跑指定场景（子串匹配）
#   scripts/run_suite.sh --list                # 只列出矩阵
#   scripts/run_suite.sh --run-dir runs/x      # 指定产物目录
#
# 之后用 `python3 -m probe compare --a A --b B` 或 `make compare A=... B=...`
# 对两次（例如不同厂商/不同机器/不同引擎参数）的结果并排对比。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

PORT="${PORT:-8100}"
WANT_LIST=0
for arg in "$@"; do
  [[ "$arg" == "--list" ]] && WANT_LIST=1
done
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="${2:?}"; shift 2 ;;
    --run-dir) RUN_DIR="${2:?}"; shift 2 ;;
    -h|--help) sed -n '2,13p' "$0" && py -m probe suite --help 2>/dev/null | head -20; exit 0 ;;
    *) break ;;  # 其余参数透传给 probe suite
  esac
done

BASE="http://127.0.0.1:$PORT"
if (( ! WANT_LIST )); then
  code="$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$BASE/health" || echo 000)"
  [[ "$code" == "200" ]] || die "服务未就绪（$BASE/health = $code）；先 scripts/start_vllm.sh"
fi

RUN_DIR="$(latest_run_dir 2>/dev/null || new_run_dir suite)"
mkdir -p "$RUN_DIR"
log "运行目录：$RUN_DIR（厂商=$VENDOR）"
py -m probe suite --port "$PORT" --outdir "$RUN_DIR" "$@"
