#!/usr/bin/env bash
# 一眼看清当前探测实例的状态：运行方式、端点、关键指标、最近一次运行目录。
#
# 用法：scripts/status.sh [--port 8100] [--run-dir DIR] [--runtime docker|host]
#   RUNTIME=host scripts/status.sh    # 物理机（裸机）模式：显示宿主机上的 vLLM 进程

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

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

echo "===================== 运行方式 ====================="
echo "运行时：$RUNTIME"
echo "厂商：$VENDOR（$VENDOR_LABEL）  SMI：${VENDOR_SMI_CMD:-无}"
# 实例概况交给 common.sh 的 instance_describe（一行描述，有存活实例返回 0）；
# 拿不到实例时给一句人类可读的提示，不要让脚本崩掉。
DESCRIBE_RC=0
instance_describe || DESCRIBE_RC=$?
echo  # instance_describe 用 printf 输出、不带换行
if (( DESCRIBE_RC != 0 )) && runtime_is_host; then
  # docker 模式下 instance_describe 已经说明了容器状态，这里只给 host 模式补充启动指引
  echo "（宿主机没有正在运行的 vLLM 实例；先执行 scripts/start_vllm.sh --host）"
fi
if runtime_is_host; then
  HOST_PY="$(host_python 2>/dev/null || echo python3)"
  echo "  HOST_PYTHON：$HOST_PY"
  echo "  vLLM 版本：$("$HOST_PY" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo '未知（当前 HOST_PYTHON 环境里 import vllm 失败）')"
  # 进程表（pid/etime/args）：优先看 start_vllm.sh --host 记录的本实例（serve.pid），
  # 没有本工具启动的实例时，只读地列出宿主机上按 pattern 能匹配到的 vLLM 进程（诊断用）。
  if host_instance_running 2>/dev/null; then
    INSTANCE_PID="$(host_instance_pid 2>/dev/null || true)"
    echo "  --- 本实例进程（serve.pid） ---"
    printf '  %-9s %-10s %s\n' "PID" "ELAPSED" "ARGS"
    ps -o pid=,etime=,args= -p "$INSTANCE_PID" 2>/dev/null | cut -c1-200 | sed 's/^[[:space:]]*/  /' || true
  else
    HOST_VLLM_PIDS="$(host_vllm_pids 2>/dev/null || true)"
    if [[ -n "$HOST_VLLM_PIDS" ]]; then
      echo "  --- 宿主机上匹配 vLLM 的进程（pgrep -f \"${HOST_VLLM_PATTERN:-…}\"） ---"
      printf '  %-9s %-10s %s\n' "PID" "ELAPSED" "ARGS"
      while IFS= read -r _pid; do
        [[ -n "$_pid" ]] || continue
        ps -o pid=,etime=,args= -p "$_pid" 2>/dev/null | cut -c1-200 | sed 's/^[[:space:]]*/  /' || true
      done <<< "$HOST_VLLM_PIDS"
    fi
  fi
fi

echo
echo "===================== 端点 ====================="
# 注意：curl -w '%{http_code}' 在连接失败时自己就会输出 000，
# 这里不能再 `|| echo 000`（会变成 000000），只用 `|| true` 防止 set -e 退出。
printf '%-28s %s\n' "/health" "$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$BASE/health" 2>/dev/null || true)"
printf '%-28s %s\n' "/v1/models" "$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$BASE/v1/models" 2>/dev/null || true)"
printf '%-28s %s\n' "/metrics" "$(curl -s -o /dev/null -w '%{http_code}' -m 10 "$BASE/metrics" 2>/dev/null || true)"
printf '%-28s %s\n' "/start_profile(profiler)" \
  "$(curl -s -o /dev/null -w '%{http_code}' -m 5 -X POST "$BASE/start_profile" 2>/dev/null || true)"

echo
echo "===================== 关键指标 ====================="
if curl -s -m 10 "$BASE/metrics" > /tmp/.vllm_probe_metrics 2>/dev/null; then
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
if [[ -n "$LATEST" && -f "$LATEST/serve.log" ]]; then
  echo "  运行目录：$LATEST"
  echo "  日志行数：$(wc -l < "$LATEST/serve.log")"
  echo "  --- 最近 5 条 step/统计 ---"
  py -m probe logs --file "$LATEST/serve.log" --live --live-tail 5 2>/dev/null | sed 's/^/  /' || true
else
  echo "  （未找到运行目录日志）"
fi
