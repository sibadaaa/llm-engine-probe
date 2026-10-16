#!/usr/bin/env bash
# 一键全链路探测：自检 → 指标基线 → 请求级 → 指标差分 → step 级 → (算子级) → 报告。
#
# 用法：
#   scripts/probe_all.sh                        # 完整跑一遍（按 .env 的档位）
#   scripts/probe_all.sh --skip-trace           # 跳过 torch profiler（档位非 kernel 时本来也会跳过）
#   scripts/probe_all.sh --with-gpu-sample      # 请求级探测期间同时采样 GPU 遥测（各厂商 SMI）
#   scripts/probe_all.sh --with-pyspy           # 结束后抓一份 py-spy 栈快照
#   scripts/probe_all.sh --run-dir /path/to/run # 复用已有运行目录
#
# 建议：先 scripts/start_vllm.sh，再本脚本；最后看 <RUN_DIR>/REPORT.md。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

SKIP_HTTP=0
SKIP_TRACE=0
SKIP_LOGS=0
WITH_GPU_SAMPLE=0
WITH_PYSPY=0
WITH_ENV=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-http) SKIP_HTTP=1; shift ;;
    --skip-trace) SKIP_TRACE=1; shift ;;
    --skip-logs) SKIP_LOGS=1; shift ;;
    --with-gpu-sample) WITH_GPU_SAMPLE=1; shift ;;
    --with-pyspy) WITH_PYSPY=1; shift ;;
    --no-env) WITH_ENV=0; shift ;;
    --run-dir) RUN_DIR="${2:?}"; shift 2 ;;
    --port) PORT="${2:?}"; shift 2 ;;
    --prompt-tokens) PROMPT_TOKENS="${2:?}"; shift 2 ;;
    --max-tokens) MAX_TOKENS="${2:?}"; shift 2 ;;
    --concurrency) CONCURRENCY="${2:?}"; shift 2 ;;
    --repeat) REPEAT="${2:?}"; shift 2 ;;
    --level) PROBE_LEVEL="${2:?}"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

RUN_DIR="$(latest_run_dir 2>/dev/null || new_run_dir probe)"
mkdir -p "$RUN_DIR"

# 如果是直接对"已在运行的实例"做探测（没经过 start_vllm.sh），补一份最小 run_meta.json，
# 这样报告的开头不会是一片"—"。
if [[ ! -f "$RUN_DIR/run_meta.json" ]]; then
  RUN_META_DIR="$RUN_DIR" PORT="$PORT" PROBE_LEVEL="$PROBE_LEVEL" py - <<'PY'
import json, os, time
meta = {
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "container": os.environ.get("CONTAINER", ""),
    "image": os.environ.get("IMAGE", ""),
    "port": int(os.environ["PORT"]),
    "probe_level": os.environ.get("PROBE_LEVEL", ""),
    "note": "由 probe_all.sh 生成（未经过 start_vllm.sh，模型/参数等字段请以实际进程为准）",
}
path = os.path.join(os.environ["RUN_META_DIR"], "run_meta.json")
open(path, "w").write(json.dumps(meta, ensure_ascii=False, indent=2))
PY
fi

log "==================== 全链路探测开始 ===================="
log "运行目录：$RUN_DIR"
log "目标：http://127.0.0.1:$PORT  档位：$PROBE_LEVEL"

# 0) 前置检查
BASE="http://127.0.0.1:$PORT"
code="$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$BASE/health" || echo 000)"
[[ "$code" == "200" ]] || die "服务未就绪（$BASE/health = $code）；先 scripts/start_vllm.sh 或 scripts/wait_ready.sh"

# 1) 开关自检
if (( WITH_ENV )); then
  log "[1/6] 探测开关自检"
  py -m probe env --port "$PORT" --outdir "$RUN_DIR" || warn "自检有未通过项（详见 $RUN_DIR/env_check.json）"
fi

# 2) 可选：GPU 遥测后台采样（各厂商 SMI）
GPU_SAMPLE_PID=""
if (( WITH_GPU_SAMPLE )); then
  # 采样窗口按"预期压测时长"估算，避免采太短或太久
  GPU_SAMPLE_WINDOW=$(( 60 + CONCURRENCY * REPEAT * 5 ))
  log "[2/6] 后台采样 GPU 遥测（${VENDOR_SMI_CMD:-自动}，约 ${GPU_SAMPLE_WINDOW}s，覆盖整个探测窗口）"
  bash "$SCRIPT_DIR/gpu_sample.sh" --run-dir "$RUN_DIR" --duration "$GPU_SAMPLE_WINDOW" --interval "$GPU_SAMPLE_INTERVAL" \
    > "$RUN_DIR/gpu_sample.log" 2>&1 &
  GPU_SAMPLE_PID=$!
else
  log "[2/6] 跳过 GPU 遥测采样（加 --with-gpu-sample 启用）"
fi

# 3) 请求级 + 指标差分
if (( SKIP_HTTP )); then
  log "[3/6] 跳过请求级探测"
else
  log "[3/6] 请求级探测（prompt≈$PROMPT_TOKENS max_tokens=$MAX_TOKENS 并发=$CONCURRENCY×$REPEAT）"
  bash "$SCRIPT_DIR/probe_http.sh" --run-dir "$RUN_DIR" --port "$PORT" \
    --prompt-tokens "$PROMPT_TOKENS" --max-tokens "$MAX_TOKENS" \
    --concurrency "$CONCURRENCY" --repeat "$REPEAT" \
    --shared-prefix-tokens "$SHARED_PREFIX_TOKENS" || warn "请求级探测有失败请求"
fi

if [[ -n "$GPU_SAMPLE_PID" ]]; then
  kill "$GPU_SAMPLE_PID" 2>/dev/null || true
  wait "$GPU_SAMPLE_PID" 2>/dev/null || true
  log "GPU 遥测采样已停止（$RUN_DIR/gpu_telem.jsonl）"
fi

# 4) step 级（日志）
if (( SKIP_LOGS )); then
  log "[4/6] 跳过 step 级日志解析"
else
  log "[4/6] step 级日志解析（--enable-logging-iteration-details）"
  if [[ -f "$RUN_DIR/serve.log" ]]; then
    py -m probe logs --file "$RUN_DIR/serve.log" \
      --jsonl "$RUN_DIR/steps.jsonl" --summary "$RUN_DIR/steps_summary.json" \
      || warn "日志解析失败（可能没有 iteration 明细；档位需 >= steps）"
  else
    warn "找不到 $RUN_DIR/serve.log；若用 docker logs 收集，请先落盘日志"
  fi
fi

# 4b) stat logger 插件（PROBE_PLUGIN=1 时由插件写出的结构化 step JSONL）
PLUGIN_JSONL=""
for candidate in "$RUN_DIR/probe_steps.jsonl" "${GAME_DIR:-/tmp}/vllm-probe/probe_steps.jsonl"; do
  if [[ -s "$candidate" ]]; then PLUGIN_JSONL="$candidate"; break; fi
done
if [[ -n "$PLUGIN_JSONL" ]]; then
  log "[4b/6] stat logger 插件数据（$(wc -l < "$PLUGIN_JSONL") 行）"
  [[ "$PLUGIN_JSONL" == "$RUN_DIR/probe_steps.jsonl" ]] || cp -f "$PLUGIN_JSONL" "$RUN_DIR/probe_steps.jsonl"
  py -m probe plugin --outdir "$RUN_DIR" --file "$RUN_DIR/probe_steps.jsonl" || warn "插件数据解析失败"
elif [[ "${PROBE_PLUGIN:-0}" == "1" ]]; then
  warn "PROBE_PLUGIN=1 但找不到 probe_steps.jsonl（应写在 /logs/probe_steps.jsonl）"
fi

# 5) 算子级（torch profiler）
if (( SKIP_TRACE )); then
  log "[5/6] 跳过算子级 trace"
elif [[ "$PROBE_LEVEL" == "kernel" ]]; then
  log "[5/6] 算子级探测（/start_profile → 负载 → /stop_profile）"
  bash "$SCRIPT_DIR/trace_capture.sh" --run-dir "$RUN_DIR" --port "$PORT" \
    --prompt-tokens "$PROMPT_TOKENS" --max-tokens "$MAX_TOKENS" \
    --concurrency "$CONCURRENCY" --repeat 1 \
    || warn "算子级探测失败（确认实例是 PROBE_LEVEL=kernel 启动的）"
else
  log "[5/6] 跳过算子级 trace（当前档位 $PROBE_LEVEL != kernel）"
fi

# 6) 可选 py-spy + 报告
if (( WITH_PYSPY )); then
  log "[6/6] py-spy 栈快照"
  bash "$SCRIPT_DIR/pyspy.sh" --mode dump --run-dir "$RUN_DIR" || warn "py-spy 失败"
else
  log "[6/6] 跳过 py-spy（加 --with-pyspy 启用）"
fi

log "生成报告"
py -m probe report --outdir "$RUN_DIR"

ok "==================== 全链路探测完成 ===================="
echo
echo "  报告：$RUN_DIR/REPORT.md"
echo "  逐请求：$RUN_DIR/http_requests.jsonl"
echo "  step 事件：$RUN_DIR/steps.jsonl"
echo "  报告里的『自动归因与优化建议』一节会直接给出下一步该调什么。"
