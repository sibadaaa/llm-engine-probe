#!/usr/bin/env bash
# 物理机（host）模式环境预检：起 vLLM 之前先体检这台机器缺什么。
#
# 用法：
#   bash scripts/host_check.sh                     # 人读输出 + 末尾汇总
#   bash scripts/host_check.sh --json --outdir "$RUN_DIR"
#   bash scripts/host_check.sh --start             # 启动前快速预检（只跑 fail 级核心项）
#   bash scripts/host_check.sh --quick             # 跳过耗时项
#   bash scripts/host_check.sh --skip-vllm-import --skip-gpu
#
# 说明：
#   * 本脚本 source scripts/lib/common.sh，从而复用 .env 的 MODEL/PORT/VENDOR/
#     HOST_PYTHON/HOST_PRELUDE_FILES 等配置；不依赖 docker（host 模式不需要）。
#   * 真正的检查逻辑在 `python3 -m probe hostcheck`（probe/hostcheck.py）。
#   * 退出码：存在 fail 级检查项 → 1，否则 0（warn/info 不影响）。
#     start_vllm.sh 预检约定：非零退出 = 有 fail 级问题。
#   * --json 时额外写 <outdir>/host_check.json；outdir 默认
#     $PROBE_OUTDIR，其次 $RUN_DIR，都没有则由 Python 侧落到 runs/latest。
#
# shellcheck shell=bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "$ROOT/scripts/lib/common.sh"

usage() {
  cat <<'EOF'
物理机（host）模式环境预检（python3 -m probe hostcheck 的 shell 包装）

用法：
  bash scripts/host_check.sh [选项]

选项：
  --json                 额外写 <outdir>/host_check.json
  --outdir DIR           产物目录（默认 $PROBE_OUTDIR，其次 $RUN_DIR，再默认 runs/latest）
  --quick                跳过耗时项（import vllm/torch、GPU SMI 采样）
  --start                启动前快速预检：只跑 fail 级核心项（隐含 --quick）
  --skip-vllm-import     跳过 import vllm 检查
  --skip-gpu             跳过 GPU SMI 采样
  -h, --help             显示本帮助

环境变量（来自 .env 或显式覆盖）：
  HOST_PYTHON           宿主 Python 解释器（默认 python3）
  MODEL                 模型目录
  PORT                  服务端口（默认 8100）
  VENDOR                厂商（auto|generic|nvidia|amd|hygon|intel|ascend）
  HOST_PRELUDE_FILES    厂商 prelude 脚本（空格分隔，缺文件只 warn）
  PROBE_OUTDIR          产物目录（--outdir 缺省时使用）

退出码：0 = 无 fail 级项；1 = 有 fail 级项（warn/info 不影响退出码）。
EOF
}

json=0
outdir=""
extra=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      json=1
      shift
      ;;
    --outdir)
      [[ $# -ge 2 && -n "${2:-}" ]] || { echo "❌ --outdir 需要一个目录参数" >&2; exit 2; }
      outdir="$2"
      shift 2
      ;;
    --outdir=*)
      outdir="${1#*=}"
      [[ -n "$outdir" ]] || { echo "❌ --outdir 不能为空" >&2; exit 2; }
      shift
      ;;
    --quick|--start|--skip-vllm-import|--skip-gpu)
      extra+=("$1")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "❌ 未知参数：$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# 宿主解释器：新版 common.sh 用 host_python() 解析；旧版/缺失时退回 HOST_PYTHON。
host_py="${HOST_PYTHON:-python3}"
if declare -F host_python >/dev/null 2>&1; then
  _resolved="$(host_python 2>/dev/null || true)"
  [[ -n "$_resolved" ]] && host_py="$_resolved"
  unset _resolved
fi

# 让宿主解释器按 start_vllm.sh 相同的 PYTHONPATH 看到 vLLM（源码安装场景）。
if declare -F host_pythonpath >/dev/null 2>&1; then
  _hpp="$(host_pythonpath 2>/dev/null || true)"
  if [[ -n "$_hpp" ]]; then
    export PYTHONPATH="$_hpp${PYTHONPATH:+:$PYTHONPATH}"
  fi
  unset _hpp
fi

# py 由 common.sh 提供（PYTHONPATH 前缀项目根，保证 `-m probe` 可用）；缺失时兜底。
if ! declare -F py >/dev/null 2>&1; then
  py() { PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 "$@"; }
fi

# 输出目录：显式 --outdir > PROBE_OUTDIR > RUN_DIR；都为空则交给 Python（runs/latest）。
if [[ -z "$outdir" ]]; then
  outdir="${PROBE_OUTDIR:-}"
fi
if [[ -z "$outdir" ]]; then
  outdir="${RUN_DIR:-}"
fi

py_args=(
  -m probe hostcheck
  --host-python "$host_py"
  --model "${MODEL:-}"
  --port "${PORT:-8100}"
  --prelude "${HOST_PRELUDE_FILES:-}"
)
[[ -n "$outdir" ]] && py_args+=(--outdir "$outdir")
[[ "$json" == "1" ]] && py_args+=(--json)
py_args+=(${extra[@]+"${extra[@]}"})

if py "${py_args[@]}"; then
  exit 0
else
  exit $?
fi
