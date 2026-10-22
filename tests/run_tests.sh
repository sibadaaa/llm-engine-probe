#!/usr/bin/env bash
# 测试入口：
#   1) 单元测试（真实日志/指标样例 + 合成 trace + mock 服务）
#   2) 脚本语法检查（bash -n）
#   3) 对本地 mock vLLM 服务跑一遍完整链路（无需 GPU/Docker）
#   4) 物理机（RUNTIME=host）模式自测（假入口，无需 GPU/vLLM/Docker）
#   5) 物理机启动模板自测（hosts/*/*.env：语法/键名/叠加/dry-run，无需 GPU/vLLM/Docker）
#
# 用法：bash tests/run_tests.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PASS=0
FAIL=0
step() { printf '\n\033[36m==== %s ====\033[0m\n' "$*"; }
check() {
  if [[ "$1" == "0" ]]; then printf '  \033[32m✅ %s\033[0m\n' "$2"; PASS=$((PASS+1));
  else printf '  \033[31m❌ %s\033[0m\n' "$2"; FAIL=$((FAIL+1)); fi
}

# ---------------------------------------------------------------------------
step "1/5 单元测试"
if python3 -m unittest discover -s tests -v; then check 0 "unittest 全部通过"; else check 1 "unittest 有失败"; fi

# ---------------------------------------------------------------------------
step "2/5 脚本语法检查（bash -n）"
SYNTAX_OK=0
for script in scripts/*.sh scripts/lib/*.sh engines/*.sh vendors/*.sh tests/run_tests.sh tests/host_mode_selftest.sh tests/host_templates_selftest.sh; do
  if bash -n "$script" 2>/tmp/bashn.err; then
    printf '  ok   %s\n' "$script"
  else
    printf '  FAIL %s\n' "$script"; cat /tmp/bashn.err; SYNTAX_OK=1
  fi
done
check $SYNTAX_OK "全部脚本语法正确"

# ---------------------------------------------------------------------------
step "3/5 mock 服务端到端（无 GPU 依赖）"
PORT="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
RUN="/tmp/probe-e2e-$$"
mkdir -p "$RUN/traces"
BASE="http://127.0.0.1:$PORT"

MOCK_TRACE_DIR="$RUN/traces" MOCK_CACHE_HIT=1 python3 tests/mock_vllm_server.py --port "$PORT" \
  > "$RUN/mock.log" 2>&1 &
MOCK_PID=$!
cleanup() { kill "$MOCK_PID" 2>/dev/null || true; }
trap cleanup EXIT

ready=1
for _ in $(seq 1 40); do
  if curl -sf -m 2 "$BASE/health" >/dev/null 2>&1; then ready=0; break; fi
  sleep 0.25
done
check $ready "mock 服务已就绪（:${PORT}）"
[[ $ready == 0 ]] || { cat "$RUN/mock.log"; exit 1; }

# --- 自检
python3 -m probe env --base-url "$BASE" --outdir "$RUN" > "$RUN/env.out" 2>&1
check $? "probe env 自检"
grep -q "逐请求指标 metrics 字段" "$RUN/env.out" && check 0 "自检识别到 --enable-per-request-metrics" || check 1 "自检未识别逐请求指标"

# --- 厂商适配层
python3 -m probe vendor > "$RUN/vendor.out" 2>&1
check $? "probe vendor（厂商检测）"
grep -q "trace 规则包" "$RUN/vendor.out" && check 0 "厂商信息包含 trace 规则包" || check 1 "厂商信息缺 trace 规则包"

# --- GPU 遥测（各厂商 SMI 抽象；无 SMI 的机器应输出 none 而不是报错）
VRAM_SUM="$(python3 -m probe telemetry --vram-sum 2>/dev/null || echo none)"
if [[ "$VRAM_SUM" == "none" || "$VRAM_SUM" =~ ^[0-9.]+$ ]]; then
  check 0 "probe telemetry --vram-sum（输出：$VRAM_SUM）"
else
  check 1 "probe telemetry --vram-sum 输出异常：$VRAM_SUM"
fi
python3 -m probe telemetry --parse samples/smi/nvidia-smi.txt --backend nvidia-smi > "$RUN/telem_parse.out" 2>&1
check $? "nvidia-smi 样例解析"
grep -q "NVIDIA H100 PCIe" "$RUN/telem_parse.out" && check 0 "解析出正确显卡名" || check 1 "解析缺显卡名"
python3 -m probe telemetry --parse samples/smi/npu-smi.txt --backend npu-smi > /dev/null 2>&1
check $? "npu-smi 样例解析"
python3 -m probe telemetry --duration 1 --interval 0.5 --outdir "$RUN" --no-raw > "$RUN/telem.out" 2>&1
check $? "GPU 遥测采样（1s）"
[[ -f "$RUN/gpu_telem.jsonl" ]] && check 0 "gpu_telem.jsonl 已生成" || check 1 "缺少 gpu_telem.jsonl"

# --- 指标快照 + 差分
python3 -m probe metrics --base-url "$BASE" --outdir "$RUN" --tag metrics_before > "$RUN/m1.out" 2>&1
check $? "metrics 快照"
python3 -m probe http --base-url "$BASE" --outdir "$RUN" --prompt-tokens 128 --max-tokens 16 \
  --concurrency 2 --repeat 2 > "$RUN/http.out" 2>&1
check $? "请求级探测（并发 2×2）"
grep -q "服务端  queue" "$RUN/http.out" && check 0 "逐请求服务端指标已聚合" || check 1 "缺少服务端逐请求指标"
python3 -m probe metrics --base-url "$BASE" --outdir "$RUN" --tag metrics_after >/dev/null 2>&1
python3 -m probe metrics --outdir "$RUN" --diff "$RUN/metrics_before.json" "$RUN/metrics_after.json" \
  --elapsed 5 > "$RUN/diff.out" 2>&1
check $? "指标差分"
python3 - "$RUN/metrics_diff.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["counters"]["vllm:prompt_tokens_total"] > 0, "prompt_tokens 未增长"
assert d["rates"]["vllm:generation_tokens_total"] > 0, "generation 速率异常"
print("  差分计数：", {k: v for k, v in list(d["counters"].items())[:4]})
PY
check $? "差分结果数值合理"

# --- 算子级（mock 的 /start_profile → 负载 → /stop_profile 会写出 trace）
PROBE_LEVEL=kernel RUN_DIR="$RUN" PORT="$PORT" bash scripts/trace_capture.sh \
  --run-dir "$RUN" --port "$PORT" --prompt-tokens 128 --max-tokens 16 --concurrency 1 --repeat 1 \
  > "$RUN/trace.out" 2>&1
check $? "trace_capture.sh 全流程"
[[ -f "$RUN/trace_summary.json" ]] && check 0 "trace_summary.json 已生成" || check 1 "缺少 trace_summary.json"

# --- step 级（用真实日志样例，模拟 serve.log）
cp samples/serve.log "$RUN/serve.log"
python3 -m probe logs --file "$RUN/serve.log" --jsonl "$RUN/steps.jsonl" \
  --summary "$RUN/steps_summary.json" > "$RUN/steps.out" 2>&1
check $? "step 级日志解析"
grep -q "decode ms / token" "$RUN/steps.out" || grep -q "解码" "$RUN/steps.out" && check 0 "step 摘要包含解码口径" || check 1 "step 摘要缺字段"

# --- stat logger 插件的 JSONL（无 GPU：用测试夹具构造与真实字段同名的记录）
python3 - "$RUN/probe_steps.jsonl" <<'PY'
import json, sys
sys.path.insert(0, "tests")
sys.path.insert(0, ".")
from test_plugin_records import _load_records_module, fake_stats

records_mod = _load_records_module()
stats = fake_stats()
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write(json.dumps({"schema": 1, "kind": "header", "vllm_version": "0.18.1+das"}) + "\n")
    for i in range(6):
        fh.write(json.dumps(records_mod.build_record(
            stats.scheduler_stats, stats.iteration_stats, step=i, ts=1000.0 + i)) + "\n")
print("已生成", sys.argv[1])
PY
check $? "生成插件 JSONL 夹具"
python3 -m probe plugin --outdir "$RUN" --file "$RUN/probe_steps.jsonl" > "$RUN/plugin.out" 2>&1
check $? "probe plugin 汇总（CLI）"
grep -q "逐请求" "$RUN/plugin.out" && check 0 "插件汇总包含逐请求服务端生命周期" || check 1 "插件汇总缺逐请求字段"
grep -q "投机解码" "$RUN/plugin.out" && check 0 "插件汇总包含投机解码接受率" || check 1 "插件汇总缺投机解码"

# --- 标准化对比套件（只跑 2 个场景加速）
python3 -m probe suite --base-url "$BASE" --outdir "$RUN" --scenarios s02,s06 > "$RUN/suite.out" 2>&1
check $? "probe suite（标准化套件，s02+s06）"
[[ -f "$RUN/suite_results.json" ]] && check 0 "suite_results.json 已生成" || check 1 "缺少 suite_results.json"
python3 - "$RUN/suite_results.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["kind"] == "vllm-probe-suite", "kind 不对"
s = d["scenarios"]["s02_chat_short_c1"]
assert s["ok"], f"s02 未成功: {s.get('error')}"
assert s["ttft_ms"]["p50"] is not None, "缺 TTFT"
assert s["engine"]["generation_tokens_per_s"] > 0, "引擎差分异常"
print("  套件 schema：", list(d["scenarios"]))
PY
check $? "套件结果 schema 合理"
python3 -m probe compare --a "$RUN/suite_results.json" --b "$RUN/suite_results.json" \
  --out "$RUN/COMPARE.md" > "$RUN/compare.out" 2>&1
check $? "probe compare（套件对比）"
grep -q "标准化套件对比" "$RUN/COMPARE.md" && check 0 "对比报告已生成" || check 1 "对比报告缺内容"

# --- 一键编排脚本（复用同一运行目录；档位 steps 时不重复跑 trace）
PROBE_LEVEL=steps bash scripts/probe_all.sh --run-dir "$RUN" --port "$PORT" \
  --prompt-tokens 128 --max-tokens 16 --concurrency 1 --repeat 1 --skip-trace \
  > "$RUN/all.out" 2>&1
check $? "probe_all.sh 一键编排"
[[ -f "$RUN/REPORT.md" ]] && check 0 "REPORT.md 已生成" || check 1 "缺少 REPORT.md"
grep -q "自动归因与优化建议" "$RUN/REPORT.md" && check 0 "报告包含自动归因章节" || check 1 "报告缺自动归因"
grep -q "stat logger 插件" "$RUN/REPORT.md" && check 0 "报告包含插件结构化数据一节" || check 1 "报告缺插件一节"

# ---------------------------------------------------------------------------
# 3b. strata flavor 端到端（F3 T3.7：无硬件的引擎能力协商回归）
# ---------------------------------------------------------------------------
step "3b/5 strata flavor 端到端（引擎协商）"
SPORT="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
SBASE="http://127.0.0.1:$SPORT"
SOUT="$RUN/strata"
mkdir -p "$SOUT"
MOCK_TRACE_DIR="$RUN/traces" python3 tests/mock_vllm_server.py --port "$SPORT" --flavor strata \
  > "$RUN/mock-strata.log" 2>&1 &
MOCK_SPID=$!
cleanup() { kill "$MOCK_PID" "$MOCK_SPID" 2>/dev/null || true; }
sready=1
for _ in $(seq 1 40); do
  if curl -sf -m 2 "$SBASE/health" >/dev/null 2>&1; then sready=0; break; fi
  sleep 0.25
done
check $sready "strata mock 服务已就绪（:$SPORT）"

# env：engine 字段落产物 + 可选指标表为 strata 集（无 cache_config_info 专列）
ENGINE=strata python3 -m probe env --base-url "$SBASE" --outdir "$SOUT" --no-request \
  > "$SOUT/env.out" 2>&1
check $? "probe env（ENGINE=strata）"
python3 - "$SOUT/env_check.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["engine"] == "strata", f"engine 字段错误: {d.get('engine')}"
names = [c["name"] for c in d["checks"] if c["name"].startswith("指标 ")]
assert any("time_to_first_token_seconds" in n for n in names), f"strata 指标表缺 TTFT: {names}"
assert not any("kv_block_lifetime" in n for n in names), "strata 不应有 kv_block 指标行"
assert not any("cache_config_info" in n for n in names), "strata 不应有 cache_config_info 指标行"
print("  env_check engine=strata；指标表为 strata 集")
PY
check $? "env_check.json 引擎字段与 strata 指标表"

# http：reasoning_content 前缀块被计数（TTFT 出数）、逐请求 metrics 显式降级、engine 落产物
ENGINE=strata python3 -m probe http --base-url "$SBASE" --outdir "$SOUT" \
  --prompt-tokens 128 --max-tokens 16 --concurrency 2 --repeat 2 > "$SOUT/http.out" 2>&1
check $? "probe http（ENGINE=strata）"
python3 - "$SOUT/http_summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["config"]["engine"] == "strata", "http_summary 配置缺 engine=strata"
assert d["ttft_ms"]["p50"] is not None, "TTFT 缺失（reasoning_content 应被 delta_content_keys 计数）"
assert d["server_side"]["available"] is False, "strata 不应有逐请求服务端指标"
assert "obs.per-request-metrics" in d["server_side"]["hint"], "降级 hint 应说明能力位"
assert d["spec_decode"]["available"] is False
print("  http_summary：TTFT 出数、server_side/spec 降级注记、engine=strata")
PY
check $? "http_summary.json 引擎协商"

# metrics：?format=prometheus 后缀抓取命中 + 阶段表缺项行（5 个生命周期直方图缺失）
ENGINE=strata python3 -m probe metrics --base-url "$SBASE" --outdir "$SOUT" --tag metrics_before \
  > "$SOUT/m1.out" 2>&1
check $? "strata metrics 快照（suffix 抓取）"
# metrics_after 带 --engine-extras（F3 T3.8）：engine_extra 段落产物，报告可读
ENGINE=strata python3 -m probe metrics --base-url "$SBASE" --outdir "$SOUT" --tag metrics_after \
  --engine-extras > "$SOUT/m2.out" 2>&1
check $? "strata metrics 快照 2（--engine-extras）"
python3 - "$SOUT/metrics_after.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
fams = ((d.get("meta") or {}).get("engine_extra") or {}).get("families") or {}
assert "strata:last_hit_rate" in fams, f"engine_extra 缺 strata:last_hit_rate: {list(fams)}"
assert "strata:live_tok_s" in fams
print("  engine_extra 段已采集 strata:* 指标")
PY
check $? "engine_extra 段落产物（--engine-extras）"
ENGINE=strata python3 -m probe metrics --outdir "$SOUT" --diff "$SOUT/metrics_before.json" \
  "$SOUT/metrics_after.json" --elapsed 5 > "$SOUT/diff.out" 2>&1
check $? "strata 指标差分"
grep -q "该指标未导出" "$SOUT/m1.out" && check 0 "阶段表缺项行 available=False（queue/prefill/decode 等）" \
  || check 1 "阶段表缺项行缺失"
python3 - "$RUN/metrics_before.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert "engine_extra" not in (d.get("meta") or {}), "vLLM 链路不应出现 engine_extra 段"
print("  vLLM 快照无 engine_extra（默认关闭 schema 不变）")
PY
check $? "vLLM 快照 schema 不变（无 engine_extra）"

# suite + report：engine 上下文、报告引擎行与能力降级注记、diagnose 引擎化
ENGINE=strata python3 -m probe suite --base-url "$SBASE" --outdir "$SOUT" --scenarios s02,s06 \
  > "$SOUT/suite.out" 2>&1
check $? "probe suite（ENGINE=strata）"
python3 - "$SOUT/suite_results.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["kind"] == "vllm-probe-suite", "kind 契约不能改"
assert d["engine"]["engine"] == "strata", f"套件上下文缺 engine: {d['engine'].get('engine')}"
print("  suite_results：kind 不变、上下文带 engine=strata")
PY
check $? "suite_results.json 引擎上下文"
ENGINE=strata python3 -m probe report --outdir "$SOUT" > "$SOUT/report.out" 2>&1
check $? "probe report（ENGINE=strata）"
grep -q "引擎：Strata" "$SOUT/REPORT.md" && check 0 "REPORT.md 含引擎行" || check 1 "REPORT.md 缺引擎行"
grep -q "不支持逐 step 日志/插件观测" "$SOUT/REPORT.md" \
  && check 0 "REPORT.md 含逐 step 降级注记" || check 1 "REPORT.md 缺逐 step 注记"
grep -q "kernel 归因不适用" "$SOUT/REPORT.md" \
  && check 0 "REPORT.md 含 kernel 降级注记" || check 1 "REPORT.md 缺 kernel 注记"
grep -q "专家缓存命中率（引擎侧）" "$SOUT/REPORT.md" \
  && check 0 "REPORT.md §4 含引擎侧命中率（engine_extra）" || check 1 "REPORT.md 缺引擎侧命中率行"
if grep -qE '\-\-max-num-seqs|\-\-max-num-batched-tokens|\-\-enable-prefix-caching|\-\-enforce-eager|\-\-speculative-config' "$SOUT/REPORT.md"; then
  check 1 "diagnose 不应含 vLLM flag 建议（引擎化失败）"
else
  check 0 "diagnose 无 vLLM flag 建议（已替换为引擎注记）"
fi

# ---------------------------------------------------------------------------
step "4/5 物理机（host）模式自测"
if bash tests/host_mode_selftest.sh; then check 0 "host 模式自测全部通过"; else check 1 "host 模式自测有失败"; fi

# ---------------------------------------------------------------------------
step "5/5 物理机启动模板自测（hosts/*/*.env）"
if bash tests/host_templates_selftest.sh; then check 0 "host 模板自测全部通过"; else check 1 "host 模板自测有失败"; fi

echo
echo "产物示例：$RUN/REPORT.md"
echo "（保留该目录便于人工检查；不需要时 rm -rf $RUN）"
echo
printf '\033[36m==== 结果：%d 项通过 / %d 项失败 ====\033[0m\n' "$PASS" "$FAIL"
[[ "$FAIL" == "0" ]]
