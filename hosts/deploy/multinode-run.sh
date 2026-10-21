#!/usr/bin/env bash
# ============================================================================
# hosts/deploy/multinode-run.sh — 多机 TP（vLLM --nnodes）实验性启动骨架
#
# 设计原则：默认**拒绝启动**。不带闸门变量运行时只打印「需要补齐的变量 +
# 节点参数 + 命令预览」，然后退出 2；确认后再打开 MULTINODE_ENABLE=1。
#
# 用法：
#   hosts/deploy/multinode-run.sh --head     # head（NODE_RANK=0）
#   hosts/deploy/multinode-run.sh --worker   # worker（NODE_RANK>=1）
#   hosts/deploy/multinode-run.sh            # 按 NODE_RANK 自动判断（0=head）
#   hosts/deploy/multinode-run.sh -h         # 帮助，退出 0
#
# 内部变量（刻意放在本脚本，不写进 .env 模板）：
#   MULTINODE_ENABLE             默认 0；必须显式设 1
#   NODES                        总节点数，默认 2（必须 >=2）
#   NODE_RANK                    本节点 rank，默认 0（0=head，其余=worker）
#   HEAD_ADDR                    head 的可路由 IP/主机名（必需）
#   HEAD_PORT                    master 通信端口，默认 29500（各节点一致）
#   MULTINODE_SKIP_VERSION_CHECK 1=跳过 `vllm serve --help` 的 --nnodes 检查
#   MN_RUNS_BASE                运行目录根，默认 <仓库>/runs
#
# 本脚本把 NODES/NODE_RANK/HEAD_ADDR/HEAD_PORT 生成成
#   --nnodes N --node-rank R --master-addr A --master-port P
# 追加到模板的 EXTRA_VLLM_ARGS（模板已有的额外参数保留；显式 EXTRA_VLLM_ARGS 优先），
# 然后只调用 hosts/vllm-host.sh up deploy/multinode-head|worker。
# worker 用 --no-wait：headless 实例没有 /health，不能等就绪。
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

say()  { printf '%s\n' "$*"; }
warn() { printf '⚠️  %s\n' "$*" >&2; }
err()  { printf '❌ %s\n' "$*" >&2; }

usage() {
  cat <<'EOF'
用法：hosts/deploy/multinode-run.sh [--head|--worker] [-h]

多机 TP 实验性骨架：默认只打印预览并退出 2；只有设置
  MULTINODE_ENABLE=1
  HEAD_ADDR=<head 的可路由 IP/主机名>
才会执行：
  hosts/vllm-host.sh up deploy/multinode-head    # rank 0，等 /health
  hosts/vllm-host.sh up deploy/multinode-worker  # rank>=1，--no-wait + --headless

可用环境变量：
  MULTINODE_ENABLE=1          显式闸门（必需）
  NODES=2                     总节点数（必须 >=2）
  NODE_RANK=0                 本节点 rank（0=head，1..N-1=worker）
  HEAD_ADDR=<ip>              head 的可路由地址（必需；不要用 127.0.0.1 做真多机）
  HEAD_PORT=29500             master 通信端口（各节点一致）
  MULTINODE_SKIP_VERSION_CHECK=1
                              跳过 `vllm serve --help` 的 --nnodes 检查（已手工确认版本时）
  MN_RUNS_BASE=<仓库>/runs    运行目录根
  EXTRA_VLLM_ARGS / EXTRA_ARGS_FILE
                              额外 vLLM 参数；不要在里面重复 --nnodes/--node-rank 等

启动顺序：先在 head 机器执行 --head，再在每台 worker 上执行 --worker（rank 不同）。
每台机器都要各自配置 .env 的 HOST_PYTHON，并保证模型路径、TP、可见卡一致。
EOF
}

ROLE_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --head)   ROLE_OVERRIDE="head"; shift ;;
    --worker) ROLE_OVERRIDE="worker"; shift ;;
    *) err "未知参数：$1（可用：--head / --worker / -h）"; exit 2 ;;
  esac
done

# --- 内部变量（刻意不放进 .env 模板）----------------------------------------
MULTINODE_ENABLE="${MULTINODE_ENABLE:-0}"
NODES="${NODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
HEAD_ADDR="${HEAD_ADDR:-}"
HEAD_PORT="${HEAD_PORT:-29500}"
MULTINODE_SKIP_VERSION_CHECK="${MULTINODE_SKIP_VERSION_CHECK:-0}"
MN_RUNS_BASE="${MN_RUNS_BASE:-${RUNS_DIR:-$ROOT/runs}}"

if [[ -n "$ROLE_OVERRIDE" ]]; then
  ROLE="$ROLE_OVERRIDE"
elif [[ "$NODE_RANK" == "0" ]]; then
  ROLE="head"
else
  ROLE="worker"
fi

if [[ "$ROLE" == "head" && "$NODE_RANK" != "0" ]]; then
  err "角色冲突：--head 要求 NODE_RANK=0，当前 NODE_RANK=$NODE_RANK"
  exit 2
fi
if [[ "$ROLE" == "worker" && "$NODE_RANK" == "0" ]]; then
  err "角色冲突：--worker 要求 NODE_RANK>=1，当前 NODE_RANK=0"
  exit 2
fi

TEMPLATE="multinode-$ROLE"
RUN_DIR="$MN_RUNS_BASE/multinode-$ROLE"
USER_EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
DISPLAY_HEAD_ADDR="${HEAD_ADDR:-<HEAD_ADDR>}"
DISPLAY_NODE_ARGS="--nnodes $NODES --node-rank $NODE_RANK --master-addr $DISPLAY_HEAD_ADDR --master-port $HEAD_PORT"

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
  if [[ -n "$USER_EXTRA_VLLM_ARGS" ]]; then
    printf '%s' "$USER_EXTRA_VLLM_ARGS"
  else
    _template_extra_args "$TEMPLATE"
  fi
}

BASE_EXTRA="$(_base_extra)"
if [[ -n "$BASE_EXTRA" ]]; then
  DISPLAY_FINAL_EXTRA="$BASE_EXTRA $DISPLAY_NODE_ARGS"
else
  DISPLAY_FINAL_EXTRA="$DISPLAY_NODE_ARGS"
fi

print_preview() {
  local reason="$1" wait_flag=""
  [[ "$ROLE" == "worker" ]] && wait_flag=" --no-wait"
  cat <<EOF
================ 多机 TP（实验性骨架）命令预览 ================
状态：$reason
  MULTINODE_ENABLE=${MULTINODE_ENABLE}
  NODES=$NODES
  NODE_RANK=$NODE_RANK
  HEAD_ADDR=${HEAD_ADDR:-<未设置>}
  HEAD_PORT=$HEAD_PORT
  role=$ROLE（模板 deploy/$TEMPLATE）
  运行目录=$RUN_DIR
  MULTINODE_SKIP_VERSION_CHECK=$MULTINODE_SKIP_VERSION_CHECK
  模板 base EXTRA_VLLM_ARGS：${BASE_EXTRA:-<空>}

需要补齐/确认的变量：
  MULTINODE_ENABLE=1                 # 显式闸门（不设时只打印本预览并退出 2）
  HEAD_ADDR=<head 可路由 IP/主机名>  # 必需；真多机不要用 127.0.0.1
  NODES=2                            # 总节点数（>=2）
  NODE_RANK=0                        # head=0；worker=1..N-1
  HEAD_PORT=29500                    # 各节点一致；防火墙放行
  MULTINODE_SKIP_VERSION_CHECK=1     # 可选：已手工确认 vLLM 支持 --nnodes 时跳过检查

本脚本将追加到 EXTRA_VLLM_ARGS 的节点参数：
  --nnodes $NODES --node-rank $NODE_RANK --master-addr $DISPLAY_HEAD_ADDR --master-port $HEAD_PORT

在对应机器上执行的入口命令（head 先启动，worker 随后）：
  # head 机（rank 0；等 /health）
  MULTINODE_ENABLE=1 NODES=$NODES NODE_RANK=0 HEAD_ADDR=$DISPLAY_HEAD_ADDR HEAD_PORT=$HEAD_PORT \\
    hosts/deploy/multinode-run.sh --head
  # 每台 worker 机（rank=1..N-1；下面以 1 为例），每台 NODE_RANK 不同
  MULTINODE_ENABLE=1 NODES=$NODES NODE_RANK=1 HEAD_ADDR=$DISPLAY_HEAD_ADDR HEAD_PORT=$HEAD_PORT \\
    hosts/deploy/multinode-run.sh --worker

启用后脚本内部最终执行的命令（按当前 role=$ROLE、NODE_RANK=$NODE_RANK 生成）：
  EXTRA_VLLM_ARGS='$DISPLAY_FINAL_EXTRA' \\
    bash hosts/vllm-host.sh up deploy/$TEMPLATE -- --run-dir '$RUN_DIR'$wait_flag

版本检查（只有 MULTINODE_ENABLE=1 时才做）：
  \${HOST_PYTHON:-python3} -c 'from vllm.entrypoints.cli.main import main; main()' serve --help | grep nnodes
  查不到时脚本会拒绝启动；已手工确认版本可设 MULTINODE_SKIP_VERSION_CHECK=1。
==============================================================
EOF
}

# --- 闸门与必需变量：默认拒绝启动 -------------------------------------------
if [[ "$MULTINODE_ENABLE" != "1" ]]; then
  print_preview "MULTINODE_ENABLE=$MULTINODE_ENABLE（默认拒绝启动）"
  err "默认拒绝启动：确认预览后，显式设置 MULTINODE_ENABLE=1 与 HEAD_ADDR=<head 地址> 再运行"
  exit 2
fi

if [[ -z "$HEAD_ADDR" ]]; then
  print_preview "HEAD_ADDR 为空（多机无法生成 --master-addr）"
  err "拒绝启动：HEAD_ADDR 不能为空（head 的可路由 IP/主机名）"
  exit 2
fi

if [[ ! "$NODES" =~ ^[1-9][0-9]*$ ]] || (( NODES < 2 )); then
  err "NODES 必须是 >=2 的整数，当前：$NODES"
  exit 2
fi
if [[ ! "$NODE_RANK" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NODES )); then
  err "NODE_RANK 必须是 0..$((NODES - 1)) 的整数，当前：$NODE_RANK"
  exit 2
fi
if [[ ! "$HEAD_PORT" =~ ^[0-9]+$ ]] || (( HEAD_PORT < 1 || HEAD_PORT > 65535 )); then
  err "HEAD_PORT 不是合法端口：$HEAD_PORT（1-65535）"
  exit 2
fi
if [[ "$HEAD_ADDR" == "127.0.0.1" || "$HEAD_ADDR" == "localhost" ]]; then
  warn "HEAD_ADDR=$HEAD_ADDR 是回环地址：真多机不可用，仅适合单机冒烟/演示"
fi
if [[ -n "${EXTRA_ARGS_FILE:-}" && ! -r "${EXTRA_ARGS_FILE}" ]]; then
  err "EXTRA_ARGS_FILE 不存在或不可读：$EXTRA_ARGS_FILE"
  exit 2
fi
if [[ ! -f "$ROOT/hosts/vllm-host.sh" ]]; then
  err "找不到入口脚本：$ROOT/hosts/vllm-host.sh（仓库不完整？）"
  exit 2
fi
if [[ "$BASE_EXTRA" == *--nnodes* || "$BASE_EXTRA" == *--node-rank* \
   || "$BASE_EXTRA" == *--master-addr* || "$BASE_EXTRA" == *--master-port* ]]; then
  err "EXTRA_VLLM_ARGS/模板里已经有 --nnodes/--node-rank/--master-addr/--master-port："
  err "请删掉这些重复参数，由本脚本根据 NODES/NODE_RANK/HEAD_ADDR/HEAD_PORT 统一生成"
  exit 2
fi

# --- 版本检查：vllm serve --help 是否含 --nnodes ----------------------------
_resolve_host_python() {
  if [[ -n "${HOST_PYTHON:-}" ]]; then
    printf '%s' "$HOST_PYTHON"
    return 0
  fi
  if [[ -f "$ROOT/.env" ]]; then
    (
      set +u
      set -a
      # shellcheck disable=SC1090
      . "$ROOT/.env" >/dev/null 2>&1 || true
      printf '%s' "${HOST_PYTHON:-python3}"
    )
    return 0
  fi
  printf 'python3'
}

version_check() {
  local py help
  py="$(_resolve_host_python)"
  say "版本检查：$py -c '...main()' serve --help（查找 --nnodes）"
  if ! command -v "$py" >/dev/null 2>&1 && [[ ! -x "$py" ]]; then
    warn "找不到 HOST_PYTHON：$py（每台机器都要各自配置 .env 的 HOST_PYTHON）"
    return 1
  fi
  if ! help="$("$py" -c 'from vllm.entrypoints.cli.main import main; main()' serve --help 2>&1)"; then
    warn "无法执行 vllm serve --help（vLLM 未安装或导入失败）：$py"
    printf '%s\n' "$help" | tail -n 5 >&2 || true
    return 1
  fi
  if [[ "$help" != *--nnodes* ]]; then
    warn "vllm serve --help 未出现 --nnodes：该 vLLM 版本不支持多机 TP"
    return 1
  fi
  say "  ✅ vllm serve --help 含 --nnodes"
  if [[ "$ROLE" == "worker" && "$help" != *--headless* ]]; then
    warn "vllm serve --help 未出现 --headless：worker 模板依赖它，请按版本调整 multinode-worker.env 的 EXTRA_VLLM_ARGS"
  fi
  return 0
}

if [[ "$MULTINODE_SKIP_VERSION_CHECK" == "1" ]]; then
  warn "已跳过 vLLM 版本检查（MULTINODE_SKIP_VERSION_CHECK=1）；请自行确认支持 --nnodes/--headless"
elif ! version_check; then
  err "拒绝启动：版本检查未通过。确认 vLLM 版本支持 --nnodes 后重试；"
  err "若已手工确认（例如自定义 HOST_SERVE_ENTRYPOINT），可设 MULTINODE_SKIP_VERSION_CHECK=1 跳过"
  exit 2
fi

# --- 生成节点参数并启动（仍然只通过 hosts/vllm-host.sh up）------------------
NODE_ARGS="--nnodes $NODES --node-rank $NODE_RANK --master-addr $HEAD_ADDR --master-port $HEAD_PORT"
if [[ -n "$BASE_EXTRA" ]]; then
  FINAL_EXTRA="$BASE_EXTRA $NODE_ARGS"
else
  FINAL_EXTRA="$NODE_ARGS"
fi

printf '\n==> 启动多机 %s（NODE_RANK=%s/%s，HEAD=%s:%s）\n' \
  "$ROLE" "$NODE_RANK" "$NODES" "$HEAD_ADDR" "$HEAD_PORT"
printf '    模板：deploy/%s   运行目录：%s\n' "$TEMPLATE" "$RUN_DIR"
printf '    EXTRA_VLLM_ARGS：%s\n' "$FINAL_EXTRA"

if [[ "$ROLE" == "head" ]]; then
  if ! EXTRA_VLLM_ARGS="$FINAL_EXTRA" \
      bash "$ROOT/hosts/vllm-host.sh" up deploy/multinode-head -- --run-dir "$RUN_DIR"; then
    err "head 启动失败（见上方 vllm-host.sh 输出与 $RUN_DIR/serve.log）"
    exit 1
  fi
else
  if ! EXTRA_VLLM_ARGS="$FINAL_EXTRA" \
      bash "$ROOT/hosts/vllm-host.sh" up deploy/multinode-worker -- --run-dir "$RUN_DIR" --no-wait; then
    err "worker 启动失败（见上方 vllm-host.sh 输出与 $RUN_DIR/serve.log）"
    exit 1
  fi
fi

if [[ "$ROLE" == "head" ]]; then
  cat <<EOF

✅ head 已就绪（等所有 worker 加入后才会完全可用）
  API：http://$HEAD_ADDR:8300（模板 PORT=8300）
  状态：bash hosts/vllm-host.sh status deploy/multinode-head
  探测：PORT=8300 bash hosts/vllm-host.sh probe deploy/multinode-head
  日志：tail -f $RUN_DIR/serve.log
  停止：bash hosts/vllm-host.sh down deploy/multinode-head -- --run-dir $RUN_DIR
EOF
else
  cat <<EOF

✅ worker 已拉起（headless；不提供 HTTP /health，--no-wait 不会等就绪）
  日志：tail -f $RUN_DIR/serve.log
  确认：在 head 上看日志/指标是否出现 rank=$NODE_RANK 的 worker 加入
  停止：bash hosts/vllm-host.sh down deploy/multinode-worker -- --run-dir $RUN_DIR
EOF
fi
