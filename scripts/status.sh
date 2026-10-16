#!/usr/bin/env bash
# 一眼看清当前探测实例的状态：容器、端点、关键指标、最近一次运行目录。
#
# 用法：scripts/status.sh [--port 8100] [--run-dir DIR]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="${2:?}"; shift 2 ;;
    --run-dir) RUN_DIR="${2:?}"; shift 2 ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

BASE="http://127.0.0.1:$PORT"

echo "===================== 容器 ====================="
echo "厂商：$VENDOR（$VENDOR_LABEL）  SMI：${VENDOR_SMI_CMD:-无}"
if container_exists "$CONTAINER"; then
  docker ps -a --filter "name=^/${CONTAINER}$" \
    --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
else
  echo "（容器 $CONTAINER 不存在；MODE=${MODE}）"
fi

echo
echo "===================== 端点 ====================="
printf '%-28s %s\n' "/health" "$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$BASE/health" || echo 000)"
printf '%-28s %s\n' "/v1/models" "$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$BASE/v1/models" || echo 000)"
printf '%-28s %s\n' "/metrics" "$(curl -s -o /dev/null -w '%{http_code}' -m 10 "$BASE/metrics" || echo 000)"
printf '%-28s %s\n' "/start_profile(profiler)" \
  "$(curl -s -o /dev/null -w '%{http_code}' -m 5 -X POST "$BASE/start_profile" 2>/dev/null || echo 000)"

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
LATEST="$(latest_run_dir 2>/dev/null || true)"
if [[ -n "${LATEST:-}" && -f "$LATEST/serve.log" ]]; then
  echo "  运行目录：$LATEST"
  echo "  日志行数：$(wc -l < "$LATEST/serve.log")"
  echo "  --- 最近 5 条 step/统计 ---"
  py -m probe logs --file "$LATEST/serve.log" --live --live-tail 5 2>/dev/null | sed 's/^/  /' || true
else
  echo "  （未找到运行目录日志）"
fi
