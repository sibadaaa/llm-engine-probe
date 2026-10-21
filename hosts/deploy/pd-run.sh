#!/usr/bin/env bash
# ============================================================================
# hosts/deploy/pd-run.sh — PD 分离（prefill / decode 两个实例）实验性启动骨架
#
# 设计原则：默认**拒绝启动**。不带闸门变量运行时只打印「需要补齐的变量 +
# KV connector 要求 + 两个实例的命令预览」，然后退出 2；方便先复核再动手。
#
# 用法：
#   hosts/deploy/pd-run.sh                 # 预览并退出 2（默认）
#   PD_ENABLE=1 KV_CONNECTOR=NixlConnector hosts/deploy/pd-run.sh
#                                          # 显式闸门 → 依次启动 prefill、decode
#   hosts/deploy/pd-run.sh -h              # 帮助，退出 0
#
# 内部变量（刻意放在本脚本，不写进 .env 模板）：
#   PD_ENABLE       默认 0；必须显式设 1
#   KV_CONNECTOR    KV connector 名称（如 NixlConnector/P2pConnector/MooncakeConnector）
#   PREFILL_PORT    prefill 端口，默认 8200（显式覆盖 deploy/pd-prefill.env 的 PORT）
#   DECODE_PORT     decode  端口，默认 8201（显式覆盖 deploy/pd-decode.env 的 PORT）
#   PROXY_ADDR      PD 前置代理/路由地址，仅用于提示；本脚本不启动代理
#   PD_RUNS_BASE    运行目录根，默认 <仓库>/runs
#   PD_PREFILL_RUN_DIR / PD_DECODE_RUN_DIR
#                   两个实例的运行目录，默认 <PD_RUNS_BASE>/pd-prefill、pd-decode
#
# 只调用 hosts/vllm-host.sh up（不绕过模板层、不直接执行 vLLM）。KV transfer
# 参数由本脚本按角色生成：prefill=kv_producer、decode=kv_consumer；如果
# EXTRA_VLLM_ARGS 或 EXTRA_ARGS_FILE 里已经写了 --kv-transfer-config，则不再自动追加
# （此时 prefill/decode 的 kv_role 由你自己保证）。
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

say()  { printf '%s\n' "$*"; }
warn() { printf '⚠️  %s\n' "$*" >&2; }
err()  { printf '❌ %s\n' "$*" >&2; }

usage() {
  cat <<'EOF'
用法：hosts/deploy/pd-run.sh [-h]

PD 分离实验性骨架：默认只打印预览并退出 2；只有同时设置
  PD_ENABLE=1
  KV_CONNECTOR=<connector 名称>
才会依次执行：
  hosts/vllm-host.sh up deploy/pd-prefill
  hosts/vllm-host.sh up deploy/pd-decode

可用环境变量：
  PD_ENABLE=1                       显式闸门（必需）
  KV_CONNECTOR=NixlConnector        KV connector 名称（必需），用于生成 --kv-transfer-config
  PREFILL_PORT=8200                 prefill 端口（默认 8200）
  DECODE_PORT=8201                  decode 端口（默认 8201）
  PROXY_ADDR=127.0.0.1:8100         PD 前置代理地址（仅提示；本脚本不启动代理）
  PD_RUNS_BASE=<仓库>/runs          运行目录根
  PD_PREFILL_RUN_DIR / PD_DECODE_RUN_DIR
                                    自定义两个实例的运行目录
  EXTRA_VLLM_ARGS / EXTRA_ARGS_FILE
                                    额外 vLLM 参数；已含 --kv-transfer-config 时不再自动追加
  EXTRA_ENV                         connector 需要的额外环境变量（K=V 列表）

示例：
  hosts/deploy/pd-run.sh
  PD_ENABLE=1 KV_CONNECTOR=NixlConnector \
    EXTRA_ENV="VLLM_NIXL_SIDE_CHANNEL_HOST=10.0.0.1 VLLM_NIXL_SIDE_CHANNEL_PORT=5600" \
    hosts/deploy/pd-run.sh

注意：本脚本不会启动 PD 代理/路由，也不会自动验证 KV 传输是否真的成功；
      两个实例各自只提供半套能力，客户端必须指向你的代理（见 pd-README.md）。
EOF
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  "") ;;
  *) err "未知参数：$1（本脚本不接受位置参数；-h 查看用法）"; exit 2 ;;
esac

# --- 内部变量（刻意不放进 .env 模板）----------------------------------------
PD_ENABLE="${PD_ENABLE:-0}"
KV_CONNECTOR="${KV_CONNECTOR:-}"
PREFILL_PORT="${PREFILL_PORT:-8200}"
DECODE_PORT="${DECODE_PORT:-8201}"
PROXY_ADDR="${PROXY_ADDR:-}"
PD_RUNS_BASE="${PD_RUNS_BASE:-${RUNS_DIR:-$ROOT/runs}}"
PREFILL_RUN_DIR="${PD_PREFILL_RUN_DIR:-$PD_RUNS_BASE/pd-prefill}"
DECODE_RUN_DIR="${PD_DECODE_RUN_DIR:-$PD_RUNS_BASE/pd-decode}"

# 调用者显式给的 EXTRA_VLLM_ARGS 优先于模板里的值（与 hosts 层叠加规则一致）。
USER_EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

# 只读取模板的 EXTRA_VLLM_ARGS（在子 shell 里 source，避免污染本脚本环境）。
_template_extra_args() {
  local template="$1"
  (
    set +u
    set -a
    # shellcheck disable=SC1090
    . "$ROOT/hosts/$template.env" >/dev/null 2>&1 || true
    printf '%s' "${EXTRA_VLLM_ARGS:-}"
  )
}

_base_extra() {
  local template="$1"
  if [[ -n "$USER_EXTRA_VLLM_ARGS" ]]; then
    printf '%s' "$USER_EXTRA_VLLM_ARGS"
  else
    _template_extra_args "$template"
  fi
}

_extra_args_file_has_kv() {
  [[ -n "${EXTRA_ARGS_FILE:-}" && -r "${EXTRA_ARGS_FILE}" ]] \
    && grep -q -- '--kv-transfer-config' "$EXTRA_ARGS_FILE" 2>/dev/null
}

# 参数：<模板 base EXTRA_VLLM_ARGS> <kv_role> <connector>
_build_extra() {
  local base="$1" role="$2" connector="$3" json
  if [[ "$base" == *--kv-transfer-config* ]] || _extra_args_file_has_kv; then
    printf '%s' "$base"
    return 0
  fi
  json="$(printf '{"kv_connector":"%s","kv_role":"%s"}' "$connector" "$role")"
  if [[ -n "$base" ]]; then
    printf '%s --kv-transfer-config %s' "$base" "$json"
  else
    printf -- '--kv-transfer-config %s' "$json"
  fi
}

PREFILL_BASE_EXTRA="$(_base_extra pd-prefill)"
DECODE_BASE_EXTRA="$(_base_extra pd-decode)"
# 预览时 KV_CONNECTOR 可能为空，用占位符让命令仍然可读。
PREFILL_EXTRA="$(_build_extra "$PREFILL_BASE_EXTRA" kv_producer "${KV_CONNECTOR:-<KV_CONNECTOR>}")"
DECODE_EXTRA="$(_build_extra "$DECODE_BASE_EXTRA" kv_consumer "${KV_CONNECTOR:-<KV_CONNECTOR>}")"

if _extra_args_file_has_kv; then
  warn "EXTRA_ARGS_FILE 已含 --kv-transfer-config：跳过自动生成，请确认 prefill/decode 的 kv_role 分别正确"
fi

print_preview() {
  local reason="$1"
  cat <<EOF
================ PD 分离（实验性骨架）启动预览 ================
状态：$reason
  PD_ENABLE=${PD_ENABLE}
  KV_CONNECTOR=${KV_CONNECTOR:-<未设置>}
  PREFILL_PORT=$PREFILL_PORT
  DECODE_PORT=$DECODE_PORT
  PROXY_ADDR=${PROXY_ADDR:-<未设置>}
  prefill 运行目录=$PREFILL_RUN_DIR
  decode  运行目录=$DECODE_RUN_DIR

需要补齐/确认的变量：
  PD_ENABLE=1                 # 显式闸门（不设时只打印本预览并退出 2）
  KV_CONNECTOR=<connector>    # 例：NixlConnector / P2pConnector / MooncakeConnector
  PREFILL_PORT=8200           # 可选，prefill 端口
  DECODE_PORT=8201            # 可选，decode 端口
  PROXY_ADDR=<host:port>      # 可选，PD 前置代理/路由（本脚本不启动代理）

KV connector 要求：
  * vLLM 需支持 --kv-transfer-config；先自查：
      \${HOST_PYTHON:-python3} -c 'from vllm.entrypoints.cli.main import main; main()' serve --help | grep kv-transfer
  * connector 的额外字段/环境变量（NIXL side-channel、IB/RDMA 网卡等）本脚本不猜，
    请用 EXTRA_ENV=... 或 EXTRA_VLLM_ARGS=... 显式提供；
    若 EXTRA_VLLM_ARGS/EXTRA_ARGS_FILE 已含 --kv-transfer-config，本脚本不再自动追加。
  * prefill 与 decode 必须能互通（同机回环或 RDMA/IB），防火墙放行 connector 端口。
  * vLLM 版本、connector 名称、网卡/IP、kv_role 语义必须匹配；不确定就继续只读预览。

两个实例的命令预览（PD_ENABLE=1 且 KV_CONNECTOR 非空时执行）：
  PORT=$PREFILL_PORT EXTRA_VLLM_ARGS='$PREFILL_EXTRA' \\
    bash hosts/vllm-host.sh up deploy/pd-prefill -- --run-dir '$PREFILL_RUN_DIR'
  PORT=$DECODE_PORT EXTRA_VLLM_ARGS='$DECODE_EXTRA' \\
    bash hosts/vllm-host.sh up deploy/pd-decode -- --run-dir '$DECODE_RUN_DIR'

先看叠加结果（不会启动）：
  hosts/vllm-host.sh show deploy/pd-prefill
  hosts/vllm-host.sh show deploy/pd-decode
==============================================================
EOF
}

# --- 闸门：默认拒绝启动 -----------------------------------------------------
if [[ "$PD_ENABLE" != "1" ]]; then
  print_preview "PD_ENABLE=$PD_ENABLE（默认拒绝启动）"
  err "默认拒绝启动：确认预览后，显式设置 PD_ENABLE=1 且 KV_CONNECTOR=<connector> 再运行"
  exit 2
fi

if [[ -z "$KV_CONNECTOR" ]]; then
  print_preview "KV_CONNECTOR 为空（无法生成 --kv-transfer-config）"
  err "拒绝启动：KV_CONNECTOR 不能为空。请先确认 vLLM 支持的 KV connector 名称"
  exit 2
fi

# --- 参数校验 ---------------------------------------------------------------
if [[ ! "$KV_CONNECTOR" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  err "KV_CONNECTOR 含不安全字符：$KV_CONNECTOR（只允许字母、数字、_ . -）"
  exit 2
fi
for item in "PREFILL_PORT:$PREFILL_PORT" "DECODE_PORT:$DECODE_PORT"; do
  name="${item%%:*}"
  value="${item#*:}"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value < 1 || value > 65535 )); then
    err "$name 不是合法端口：$value（1-65535）"
    exit 2
  fi
done
if [[ "$PREFILL_PORT" == "$DECODE_PORT" ]]; then
  err "PREFILL_PORT 与 DECODE_PORT 不能相同：$PREFILL_PORT"
  exit 2
fi
if [[ -n "${EXTRA_ARGS_FILE:-}" && ! -r "${EXTRA_ARGS_FILE}" ]]; then
  err "EXTRA_ARGS_FILE 不存在或不可读：$EXTRA_ARGS_FILE"
  exit 2
fi
if [[ ! -f "$ROOT/hosts/vllm-host.sh" ]]; then
  err "找不到入口脚本：$ROOT/hosts/vllm-host.sh（仓库不完整？）"
  exit 2
fi

# --- 启动两个实例（仍然只通过 hosts/vllm-host.sh up）------------------------
start_instance() {
  local label="$1" template="$2" port="$3" extra="$4" run_dir="$5"
  printf '\n==> 启动 %s（模板 deploy/%s，PORT=%s）\n' "$label" "$template" "$port"
  printf '    运行目录：%s\n' "$run_dir"
  if ! PORT="$port" EXTRA_VLLM_ARGS="$extra" \
      bash "$ROOT/hosts/vllm-host.sh" up "deploy/$template" -- --run-dir "$run_dir"; then
    err "$label 启动失败（见上方 vllm-host.sh 输出与 $run_dir/serve.log）"
    return 1
  fi
}

say "PD 闸门已开：KV_CONNECTOR=$KV_CONNECTOR，prefill=:$PREFILL_PORT，decode=:$DECODE_PORT"
say "提示：本脚本不启动代理/路由，也不验证 KV 传输；两个实例就绪后请按 pd-README.md 复核。"

if ! start_instance "prefill" "pd-prefill" "$PREFILL_PORT" "$PREFILL_EXTRA" "$PREFILL_RUN_DIR"; then
  exit 1
fi

if ! start_instance "decode" "pd-decode" "$DECODE_PORT" "$DECODE_EXTRA" "$DECODE_RUN_DIR"; then
  warn "decode 启动失败；prefill 可能仍在运行。停止 prefill："
  say "  bash hosts/vllm-host.sh down deploy/pd-prefill -- --run-dir $PREFILL_RUN_DIR"
  exit 1
fi

cat <<EOF

✅ 两个实例已启动（实验性骨架；KV 传输尚未验证）
  prefill: http://127.0.0.1:$PREFILL_PORT  （运行目录 $PREFILL_RUN_DIR）
  decode : http://127.0.0.1:$DECODE_PORT   （运行目录 $DECODE_RUN_DIR）
  proxy  : ${PROXY_ADDR:-<未配置；客户端需要你自己的 PD 代理/路由>}

分别探测（端口必须显式给，模板默认端口不一定是运行端口）：
  PORT=$PREFILL_PORT bash hosts/vllm-host.sh probe deploy/pd-prefill
  PORT=$DECODE_PORT  bash hosts/vllm-host.sh probe deploy/pd-decode

停止（按运行目录，避免 down 只取最新目录）：
  bash hosts/vllm-host.sh down deploy/pd-prefill -- --run-dir $PREFILL_RUN_DIR
  bash hosts/vllm-host.sh down deploy/pd-decode  -- --run-dir $DECODE_RUN_DIR

排障：先看 $PREFILL_RUN_DIR/serve.log 与 $DECODE_RUN_DIR/serve.log，
      并确认 --kv-transfer-config 在两边分别出现且 kv_role 正确。
EOF
