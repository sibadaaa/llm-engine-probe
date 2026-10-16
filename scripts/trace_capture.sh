#!/usr/bin/env bash
# 算子级探测：用 vLLM 自带的 torch profiler 抓一份 Chrome trace。
#
# 前置条件：实例启动时带了 --profiler-config（PROBE_LEVEL=kernel），
#           此时 /start_profile、/stop_profile 才会被注册。
#
# 流程：POST /start_profile → 发一段负载 → POST /stop_profile → 等待落盘 → 分析
#
# 用法：
#   scripts/trace_capture.sh                          # 默认压一段 decode 负载
#   scripts/trace_capture.sh --prompt-tokens 4096 --max-tokens 64 --concurrency 8
#   scripts/trace_capture.sh --skip-load              # 只开关 profiler，自己手动打流量
#   scripts/trace_capture.sh --analyze-only           # 只分析已有 trace

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

SKIP_LOAD=0
ANALYZE_ONLY=0
WAIT_TRACE="${WAIT_TRACE:-180}"
KEY_AVG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-load) SKIP_LOAD=1; shift ;;
    --analyze-only) ANALYZE_ONLY=1; shift ;;
    --run-dir) RUN_DIR="${2:?}"; shift 2 ;;
    --port) PORT="${2:?}"; shift 2 ;;
    --prompt-tokens) PROMPT_TOKENS="${2:?}"; shift 2 ;;
    --max-tokens) MAX_TOKENS="${2:?}"; shift 2 ;;
    --concurrency) CONCURRENCY="${2:?}"; shift 2 ;;
    --repeat) REPEAT="${2:?}"; shift 2 ;;
    --wait) WAIT_TRACE="${2:?}"; shift 2 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

RUN_DIR="$(latest_run_dir 2>/dev/null || new_run_dir trace)"
TRACE_DIR="$RUN_DIR/traces"
mkdir -p "$TRACE_DIR"
BASE="http://127.0.0.1:$PORT"

post() {
  local path="$1"
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' -m 60 -X POST "$BASE$path" || echo 000)"
  echo "$code"
}

if (( ! ANALYZE_ONLY )); then
  code="$(post /start_profile)"
  [[ "$code" == "200" ]] || die "/start_profile 返回 $code；实例需要用 PROBE_LEVEL=kernel 启动（--profiler-config）"
  ok "profiler 已启动"

  if (( SKIP_LOAD )); then
    log "跳过自动负载，请手动打流量，然后执行：curl -X POST $BASE/stop_profile"
    log "完成后重新运行：scripts/trace_capture.sh --analyze-only"
    exit 0
  fi

  log "开始负载（profiler 正在记录）：prompt≈$PROMPT_TOKENS max_tokens=$MAX_TOKENS 并发=$CONCURRENCY×$REPEAT"
  py -m probe http --port "$PORT" --outdir "$RUN_DIR" \
    --prompt-tokens "$PROMPT_TOKENS" --max-tokens "$MAX_TOKENS" \
    --concurrency "$CONCURRENCY" --repeat "$REPEAT" --warmup 0 || warn "负载阶段有失败请求"

  code="$(post /stop_profile)"
  [[ "$code" == "200" ]] || die "/stop_profile 返回 $code"
  ok "profiler 已停止，等待 trace 落盘（最多 ${WAIT_TRACE}s）"
fi

# ---------------------------------------------------------------------------
# 等待 trace 文件出现
# ---------------------------------------------------------------------------
waited=0
while :; do
  count=$(find "$TRACE_DIR" -maxdepth 1 -name '*.pt.trace.json*' 2>/dev/null | wc -l)
  if (( count > 0 )); then break; fi
  if (( waited >= WAIT_TRACE )); then
    warn "等待 ${WAIT_TRACE}s 仍未见 trace 文件，目录内容："
    ls -la "$TRACE_DIR" || true
    die "trace 未生成；检查启动参数 --profiler-config 的 torch_profiler_dir 是否指向 /traces"
  fi
  sleep 5; waited=$((waited + 5))
done
ok "落盘完成（${waited}s）"

PRIMARY="$(find "$TRACE_DIR" -maxdepth 1 -name 'rank0*.pt.trace.json*' | sort | head -1)"
[[ -n "$PRIMARY" ]] || PRIMARY="$(find "$TRACE_DIR" -maxdepth 1 -name '*.pt.trace.json*' | sort | head -1)"
KEY_AVG="$(find "$TRACE_DIR" -maxdepth 1 -name 'profiler_out_*.txt' | sort | head -1)"

# 记录产物清单
TRACE_DIR="$TRACE_DIR" RUN_DIR="$RUN_DIR" py - <<'PY'
import json, os, pathlib
trace_dir = pathlib.Path(os.environ["TRACE_DIR"])
files = sorted(str(p) for p in trace_dir.glob("*"))
out = {
    "files": [{"name": pathlib.Path(f).name, "size_mb": round(pathlib.Path(f).stat().st_size / 2**20, 2)} for f in files],
    "count": len(files),
}
pathlib.Path(os.environ["RUN_DIR"], "profiler_files.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"trace 目录 {trace_dir} 共 {len(files)} 个文件")
for item in out["files"]:
    print(f"  {item['name']}  {item['size_mb']} MB")
PY

log "分析主 trace：$PRIMARY"
py -m probe trace --outdir "$RUN_DIR" --path "$PRIMARY" \
  $([[ -n "$KEY_AVG" ]] && echo "--key-averages $KEY_AVG") --top 25

ok "算子级探测完成：$RUN_DIR/trace_summary.json"
log "提示：用 Chrome/Perfetto 打开 $PRIMARY 可看完整时间线"
