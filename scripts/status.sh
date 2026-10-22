#!/usr/bin/env bash
# 一眼看清当前探测实例的状态：运行方式、端点、关键指标、最近一次运行目录。
#
# 用法：scripts/status.sh [--port 8100] [--run-dir DIR] [--runtime docker|host]
#   RUNTIME=host scripts/status.sh    # 物理机（裸机）模式：显示宿主机上的 vLLM 进程

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

# 引擎组件：加载 profile（ENGINE != vllm 时版本探测改走 ENGINE_VERSION_CMD；未知 ENGINE die）
engine_meta

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="${2:?}"; shift 2 ;;
    --run-dir) RUN_DIR="${2:?}"; shift 2 ;;
    --runtime)
      # runtime_normalize 只打印规范化结果（container→docker、bare/native→host），非法值 die
      RUNTIME="$(runtime_normalize "${2:?}")"
      export RUNTIME
      shift 2
      ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

BASE="http://127.0.0.1:$PORT"

# 实例带 --api-key 启动时，端点探测要带 Authorization（未设置则不加）
AUTH_ARGS=()
if [[ -n "${VLLM_API_KEY:-}" ]]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${VLLM_API_KEY}")
fi

echo "===================== 运行方式 ====================="
echo "运行时：$RUNTIME"
echo "厂商：$VENDOR（$VENDOR_LABEL）  SMI：${VENDOR_SMI_CMD:-无}"
# 实例概况交给 common.sh 的 instance_describe（一行描述，有存活实例返回 0）；
# 拿不到实例时给一句人类可读的提示，不要让脚本崩掉。
DESCRIBE_RC=0
instance_describe || DESCRIBE_RC=$?
echo  # instance_describe 用 printf 输出、不带换行
if runtime_is_host; then
  HOST_VLLM_PIDS="$(host_vllm_pids 2>/dev/null || true)"
  if (( DESCRIBE_RC != 0 )); then
    # start_vllm.sh --host 会写 serve.pid；没有时可能是外部方式启动的实例，
    # 这里按 HOST_VLLM_PATTERN 只读匹配，提示"外部实例"而不是误报"未运行"。
    if [[ -n "$HOST_VLLM_PIDS" ]]; then
      echo "（外部实例：非本工具启动、未记录 serve.pid；按进程名匹配到 $(printf '%s\n' "$HOST_VLLM_PIDS" | wc -l | tr -d ' ') 个 vLLM 进程）"
    else
      echo "（宿主机没有正在运行的 vLLM 实例；先执行 scripts/start_vllm.sh --host）"
    fi
  fi
  HOST_PY="$(host_python 2>/dev/null || echo python3)"
  echo "  HOST_PYTHON：$HOST_PY"
  # 版本探测与真实启动同环境（source prelude + host PYTHONPATH），避免假失败。
  # 引擎身份（T1.9）：ENGINE != vllm 时改走引擎 profile 的 ENGINE_VERSION_CMD（无则注记跳过）；
  # ENGINE=vllm 保持原 import vllm 探测不变。
  if [[ "$ENGINE" == "vllm" ]]; then
    echo "  vLLM 版本：$(host_exec "$HOST_PY" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo '未知（HOST_PYTHON 环境里 import vllm 失败；检查 HOST_PRELUDE_FILES / SRC_DIR）')"
  elif [[ -n "$ENGINE_VERSION_CMD" ]]; then
    # shellcheck disable=SC2086  # ENGINE_VERSION_CMD 按空白切分为命令 + 参数
    ENGINE_VERSION_OUT="$(host_exec $ENGINE_VERSION_CMD 2>/dev/null | head -n 1 || true)"
    echo "  引擎版本（ENGINE=$ENGINE，via $ENGINE_VERSION_CMD）：${ENGINE_VERSION_OUT:-未知（ENGINE_VERSION_CMD 未取到输出；见 docs/design/strata-engine/施工图.md T2.5 真机核对清单）}"
  else
    echo "  引擎版本：未知（ENGINE=$ENGINE 未声明 ENGINE_VERSION_CMD；引擎预检可改走 /props 元数据）"
  fi
  # 进程表（pid/etime/args）：优先看 start_vllm.sh --host 记录的本实例（serve.pid），
  # 没有本工具启动的实例时，只读地列出宿主机上按 pattern 能匹配到的 vLLM 进程（诊断用）。
  if host_instance_running 2>/dev/null; then
    INSTANCE_PID="$(host_instance_pid 2>/dev/null || true)"
    echo "  --- 本实例进程（serve.pid） ---"
    printf '  %-9s %-10s %s\n' "PID" "ELAPSED" "ARGS"
    ps -o pid=,etime=,args= -p "$INSTANCE_PID" 2>/dev/null | cut -c1-200 | sed 's/^[[:space:]]*/  /' || true
  elif [[ -n "$HOST_VLLM_PIDS" ]]; then
    echo "  --- 宿主机上匹配引擎的进程（pgrep -f \"$(engine_process_pattern)\"） ---"
    printf '  %-9s %-10s %s\n' "PID" "ELAPSED" "ARGS"
    while IFS= read -r _pid; do
      [[ -n "$_pid" ]] || continue
      ps -o pid=,etime=,args= -p "$_pid" 2>/dev/null | cut -c1-200 | sed 's/^[[:space:]]*/  /' || true
    done <<< "$HOST_VLLM_PIDS"
  fi
fi

echo
echo "===================== 端点 ====================="
# 注意：curl -w '%{http_code}' 在连接失败时自己就会输出 000，
# 这里不能再 `|| echo 000`（会变成 000000），只用 `|| true` 防止 set -e 退出。
printf '%-28s %s\n' "/health" "$(curl -s -o /dev/null -w '%{http_code}' -m 5 "${AUTH_ARGS[@]+"${AUTH_ARGS[@]}"}" "$BASE/health" 2>/dev/null || true)"
printf '%-28s %s\n' "/v1/models" "$(curl -s -o /dev/null -w '%{http_code}' -m 5 "${AUTH_ARGS[@]+"${AUTH_ARGS[@]}"}" "$BASE/v1/models" 2>/dev/null || true)"
printf '%-28s %s\n' "/metrics" "$(curl -s -o /dev/null -w '%{http_code}' -m 10 "${AUTH_ARGS[@]+"${AUTH_ARGS[@]}"}" "$BASE/metrics${ENGINE_METRICS_FETCH_SUFFIX:-}" 2>/dev/null || true)"
# 只读探活：用 GET 而不是 POST /start_profile（对生产实例绝无副作用）。
# 404=未配置 profiler；405/200/400=路由存在；其它按原样显示。
printf '%-28s %s\n' "/start_profile(profiler，只读探活)" \
  "$(curl -s -o /dev/null -w '%{http_code}' -m 5 "${AUTH_ARGS[@]+"${AUTH_ARGS[@]}"}" "$BASE/start_profile" 2>/dev/null || true)"

echo
echo "===================== 关键指标 ====================="
# /metrics 抓取带引擎 profile 的后缀（F3 T3.6；ENGINE=vllm 时为空串，行为不变）
if curl -s -m 10 "${AUTH_ARGS[@]+"${AUTH_ARGS[@]}"}" "$BASE/metrics${ENGINE_METRICS_FETCH_SUFFIX:-}" > /tmp/.vllm_probe_metrics 2>/dev/null; then
  py - <<'PY'
import sys
from probe.metrics_probe import parse_metrics_text, engine_state
snap = parse_metrics_text(open("/tmp/.vllm_probe_metrics").read())
st = engine_state(snap)
def g(v): return "—" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
print(f"  running={g(st['num_requests_running'])}  waiting={g(st['num_requests_waiting'])}"
      f"  等待原因={st['num_requests_waiting_by_reason']}")
print(f"  KV 使用率={g((st['kv_cache_usage_perc'] or 0) * 100)}%  抢占={g(st['num_preemptions_total'])}")
rate = st["prefix_cache_hit_rate"]
print(f"  前缀缓存命中率={'—' if rate is None else f'{rate*100:.1f}%'}"
      f"  已缓存 prompt token={g(st['prompt_tokens_cached_total'])}")
print(f"  prompt_tokens={g(st['prompt_tokens_total'])}  generation_tokens={g(st['generation_tokens_total'])}")
sd = st["spec_decode"]
if sd["drafts"]:
    print(f"  投机解码: 平均接受长度={g(sd['mean_acceptance_length'])} 草稿接受率="
          f"{g((sd['draft_acceptance_rate'] or 0) * 100)}% 按位置={sd['per_position_accepted']}")
it = st["iteration_tokens"]
if it:
    print(f"  每 step token 数: 均值={g(it['tokens_per_step'])} p50={g(it['p50_tokens_per_step'])} steps={g(it['steps'])}")
info = snap.family("vllm:cache_config_info")
if info and info.series:
    lb = info.series[0].labels
    print(f"  配置: prefix_caching={lb.get('enable_prefix_caching')} "
          f"block_size={lb.get('block_size')} kv_cache_size_tokens={lb.get('kv_cache_size_tokens')}")
PY
else
  echo "  （无法读取 /metrics）"
fi

echo
echo "===================== 日志 ====================="
# 注意：latest_run_dir 找不到目录时返回 1（不 die），放进 if 条件里优雅降级即可。
LATEST=""
if ! LATEST="$(latest_run_dir 2>/dev/null)"; then
  LATEST=""
fi
LOG_FILE=""
if [[ -n "$LATEST" && -f "$LATEST/serve.log" ]]; then
  LOG_FILE="$LATEST/serve.log"
fi
# 外部启动实例：日志通常不在 RUN_DIR/serve.log，依次看 run_meta["serve_log"] 与 PROBE_LOG_FILE
if [[ -z "$LOG_FILE" && -n "$LATEST" && -f "$LATEST/run_meta.json" ]]; then
  LOG_FILE="$(py - "$LATEST/run_meta.json" 2>/dev/null <<'PY'
import json, os, sys
try:
    meta = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(0)
path = str(meta.get("serve_log") or "").strip()
if path and os.path.isfile(path):
    print(path)
PY
)"
fi
if [[ -z "$LOG_FILE" && -n "${PROBE_LOG_FILE:-}" && -f "${PROBE_LOG_FILE}" ]]; then
  LOG_FILE="$PROBE_LOG_FILE"
fi
if [[ -n "$LOG_FILE" && -f "$LOG_FILE" ]]; then
  echo "  日志文件：$LOG_FILE"
  echo "  日志行数：$(wc -l < "$LOG_FILE")"
  echo "  --- 最近 5 条 step/统计 ---"
  py -m probe logs --file "$LOG_FILE" --live --live-tail 5 2>/dev/null | sed 's/^/  /' || true
else
  echo "  （未找到运行目录日志；外部实例可设置 PROBE_LOG_FILE 或在 run_meta.json 里写 serve_log）"
fi
