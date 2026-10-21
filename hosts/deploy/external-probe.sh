#!/usr/bin/env bash
# ============================================================================
# hosts/deploy/external-probe.sh — 只读探测「不是本工具启动」的 vLLM 实例
#
# 适用：systemd / 手工 / 其它框架启动的实例（没有 <RUN_DIR>/serve.pid）。
# 原则：**只探测，绝不 start/stop**。本脚本只设置 PROBE_LOG_FILE/PORT/RUN_DIR，
#       然后 exec 到 hosts/vllm-host.sh probe（内部是 scripts/probe_all.sh
#       --runtime host --port <PORT>），不会调用 up/down。
#
# 用法：
#   hosts/deploy/external-probe.sh <serve.log> [模板...]
#   hosts/deploy/external-probe.sh -h
#
# 示例：
#   hosts/deploy/external-probe.sh /var/log/vllm/serve.log
#   PORT=8200 hosts/deploy/external-probe.sh ~/vllm.log vendor/hygon
#   RUN_DIR=runs/live-host hosts/deploy/external-probe.sh /data/serve.log scenario/tp4-prod
#
# 日志会被复制成 <RUN_DIR>/serve.log 再做 step 级解析；实例带 --api-key 时
# 用 VLLM_API_KEY=... 传入（产物里的 key 会打码，但启动类产物请自行保护权限）。
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

err() { printf '❌ %s\n' "$*" >&2; }

usage() {
  cat <<'EOF'
用法：hosts/deploy/external-probe.sh <serve.log> [模板...]

只读探测已经在运行的 vLLM 实例（systemd/手工启动，不是本工具起的）。
绝不 start/stop；只调用 hosts/vllm-host.sh probe。

参数：
  <serve.log>     外部实例的日志文件（必需；会复制为 <RUN_DIR>/serve.log 做 step 解析）
  [模板...]       可选模板链，如 vendor/hygon scenario/tp4-prod；用于补厂商/场景变量

环境变量：
  PORT            外部实例端口，默认 8000（显式设置会覆盖 .env/模板里的 PORT）
  RUN_DIR         运行目录；不设时自动用 runs/external-probe-<时间戳>，
                  避免把日志复制进已有运行目录、覆盖别人的 serve.log
  RUNS_DIR        RUN_DIR 的根目录（默认 <仓库>/runs）
  VLLM_API_KEY    实例带 --api-key 时填同一个 key
  PROBE_LEVEL     探测档位（off|basic|steps|deep|kernel；默认沿用 .env）

示例：
  hosts/deploy/external-probe.sh /var/log/vllm/serve.log
  PORT=8200 hosts/deploy/external-probe.sh ~/vllm.log vendor/hygon
  RUN_DIR=runs/live-host hosts/deploy/external-probe.sh /data/serve.log

提示：外部实例没有 serve.pid，scripts/stop_vllm.sh 不会停它；请用它自己的方式停止。
EOF
}

if [[ $# -eq 0 ]]; then
  usage >&2
  err "缺少日志文件参数（用法：hosts/deploy/external-probe.sh <serve.log> [模板...]）"
  exit 2
fi
case "$1" in
  -h|--help) usage; exit 0 ;;
  -*) err "未知选项：$1（第一个参数必须是外部实例的日志文件；-h 查看用法）"; exit 2 ;;
esac

LOG_FILE="$1"
shift

if [[ -z "$LOG_FILE" ]]; then
  err "日志文件路径为空"
  exit 2
fi
if [[ ! -e "$LOG_FILE" ]]; then
  err "日志文件不存在：$LOG_FILE"
  exit 2
fi
if [[ ! -f "$LOG_FILE" ]]; then
  err "不是普通文件（目录/设备？）：$LOG_FILE"
  exit 2
fi
if [[ ! -r "$LOG_FILE" ]]; then
  err "日志文件不可读：$LOG_FILE"
  exit 2
fi

for arg in "$@"; do
  case "$arg" in
    -*) err "未知选项：$arg（模板写成 vendor/hygon 这类名字；探测参数请直接用 scripts/probe_all.sh）"; exit 2 ;;
  esac
done

# 探测报告按物理机（host）标注；不改实例、不启动任何东西。
export RUNTIME=host
export PROBE_LOG_FILE="$LOG_FILE"

# 默认 8000：外部实例最常见的端口；显式 PORT=... 优先。
: "${PORT:=8000}"
export PORT

# 默认给外部实例一个独立运行目录：probe_all.sh 会把外部日志复制成
# <RUN_DIR>/serve.log，复用旧目录可能覆盖上一次的产物。
if [[ -z "${RUN_DIR:-}" ]]; then
  RUNS_BASE="${RUNS_DIR:-$ROOT/runs}"
  RUN_DIR="$RUNS_BASE/external-probe-$(date +%Y%m%d-%H%M%S)"
  export RUN_DIR
fi

printf '只读探测外部实例：log=%s port=%s run_dir=%s\n' "$LOG_FILE" "$PORT" "$RUN_DIR" >&2

# exec：退出码/信号原样透传；只进 probe 分支，不存在 up/down 路径。
exec bash "$ROOT/hosts/vllm-host.sh" probe "$@"
