#!/usr/bin/env bash
# 采样 GPU 遥测（显存/功耗/温度/利用率），用于和 step 时间/直方图对齐解释抖动。
#
# 多厂商：具体 SMI 命令由厂商 profile 决定（hygon=hy-smi / nvidia=nvidia-smi /
# amd=amd-smi|rocm-smi / ascend=npu-smi / intel=xpu-smi），解析在
# probe/telemetry.py 中完成并归一化成统一 JSONL。
#
# 用法：
#   scripts/gpu_sample.sh --duration 60           # 采样 60 秒
#   scripts/gpu_sample.sh --duration 300 --interval 1
#   scripts/gpu_sample.sh --once --json           # 采一次（调试）
#
# 产物：
#   <RUN_DIR>/gpu_telem.jsonl   每行一次采样（含各卡明细与求和）
#   <RUN_DIR>/gpu_telem_raw.txt 原始 SMI 输出快照（每行的卡片数与时间戳）
#
# 说明：部分平台的 SMI 不提供真实的利用率（如 hy-smi 的 DCU% 常为 N/A），
#       GPU 侧真实忙碌度请用 torch profiler 的时间线。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

DURATION="${GPU_SAMPLE_DURATION:-60}"
ONCE=0
JSON_FLAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration) DURATION="${2:?}"; shift 2 ;;
    --interval) GPU_SAMPLE_INTERVAL="${2:?}"; shift 2 ;;
    --once) ONCE=1; shift ;;
    --json) JSON_FLAG="--json"; shift ;;
    --run-dir) RUN_DIR="${2:?}"; shift 2 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

RUN_DIR="$(latest_run_dir 2>/dev/null || new_run_dir gpu-sample)"
mkdir -p "$RUN_DIR"

if (( ONCE )); then
  py -m probe telemetry --once ${JSON_FLAG:+$JSON_FLAG}
  exit $?
fi

log "采样 GPU 遥测（厂商=$VENDOR，SMI=${VENDOR_SMI_CMD:-自动}）：每 ${GPU_SAMPLE_INTERVAL}s 一次，共 ${DURATION}s -> $RUN_DIR/gpu_telem.jsonl"
py -m probe telemetry --duration "$DURATION" --interval "$GPU_SAMPLE_INTERVAL" \
  --outdir "$RUN_DIR" --file "$RUN_DIR/gpu_telem.jsonl" --raw-file "$RUN_DIR/gpu_telem_raw.txt"
ok "遥测采样完成"
