#!/usr/bin/env bash
# 物理机（host）启动模板入口：把 hosts/<类别>/<名字>.env 叠到 .env 之上，
# 再调用仓库里现有的 scripts/*.sh（不修改 start_vllm.sh / common.sh）。
#
# 用法：
#   hosts/vllm-host.sh list                        # 列出全部模板
#   hosts/vllm-host.sh show  <模板>...             # 只打印生效配置，不启动
#   hosts/vllm-host.sh check <模板>... [选项]      # 物理机预检（透传 host_check.sh 选项）
#   hosts/vllm-host.sh dry-run <模板>...           # 打印 serve_launch.sh，不启动
#   hosts/vllm-host.sh up    <模板>... [-- 参数]   # 启动（等 /health 就绪）
#   hosts/vllm-host.sh up-kernel <模板>...         # 以 kernel 档启动（profiler + NVTX）
#   hosts/vllm-host.sh down  <模板>...             # 停止本工具启动的实例
#   hosts/vllm-host.sh status <模板>...            # 状态/端点/指标
#   hosts/vllm-host.sh probe <模板>...             # 全链路探测 + REPORT.md
#   hosts/vllm-host.sh suite <模板>...             # 标准化 8 场景套件
#   hosts/vllm-host.sh trace <模板>...             # 抓 torch trace（需 kernel 档启动）
#   hosts/vllm-host.sh pyspy <模板>...             # 本机 PID 栈采样
#   hosts/vllm-host.sh logs  <模板>...             # tail -f 最新运行目录的 serve.log
#
# 模板可写短名（hygon）或全名（vendor/hygon）。多个模板按书写顺序叠加，
# 后写的覆盖先写的；命令行显式导出的变量（如 TP=2）优先级最高。
#   hosts/vllm-host.sh up vendor/hygon scenario/tp4-prod spec/dflash
#
# 也可以用 HOST_TEMPLATES 环境变量给默认模板（空格或逗号分隔）：
#   HOST_TEMPLATES="vendor/hygon scenario/tp4-prod spec/dflash" hosts/vllm-host.sh up
#
# 额外参数放在 `--` 之后，透传给底层脚本：
#   hosts/vllm-host.sh up vendor/hygon -- --skip-check
#   hosts/vllm-host.sh check vendor/hygon -- --json --skip-gpu --skip-vllm-import

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/env_overlay.sh
source "$SCRIPT_DIR/lib/env_overlay.sh"

usage() {
  # 打印文件顶部的连续注释块（跳过 shebang 与空行），改代码行数不会截断帮助。
  awk 'NR == 1 { next }
       /^#/ { sub(/^# ?/, ""); print; next }
       /^[[:space:]]*$/ { next }
       { exit }' "$0"
}

die() {
  printf '❌ %s\n' "$*" >&2
  exit 2
}

# 必须在任何配置加载之前快照显式环境。
host_template_snapshot_explicit

action="${1:-}"
if [[ -z "$action" || "$action" == "-h" || "$action" == "--help" ]]; then
  usage
  exit 0
fi
shift

case "$action" in
  list|show|check|dry-run|up|up-kernel|down|status|probe|suite|trace|pyspy|logs) ;;
  *)
    usage >&2
    die "未知动作：$action"
    ;;
esac

templates=()
passthrough=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --)
      shift
      passthrough=("$@")
      break
      ;;
    -*)
      die "未知选项：$1（模板写成名字；底层脚本参数放到 -- 之后）"
      ;;
    *)
      templates+=("$1")
      shift
      ;;
  esac
done

if ((${#templates[@]} == 0)) && [[ -n "${HOST_TEMPLATES:-}" ]]; then
  # 支持空格或逗号分隔；shellcheck 对刻意分词给出提示，这里就是所需行为。
  read -r -a templates <<< "${HOST_TEMPLATES//,/ }"
fi

if [[ "$action" == "list" ]]; then
  host_template_list
  exit 0
fi

host_template_apply "${templates[@]}"

# 与 common.sh 对齐的最小默认值，保证 show/status 在缺少 .env 时也能读。
: "${HOST_PYTHON:=python3}"
: "${PORT:=8100}"
: "${TP:=1}"
: "${PROBE_LEVEL:=steps}"

case "$action" in
  show)
    host_template_show
    printf '\n校验：\n'
    if host_template_validate; then
      printf '  ✅ 必填项与类型检查通过（模型/驱动是否真的可用请再跑 check）\n'
    else
      printf '  ⚠️  有配置问题（见上）；show 不阻断，up/dry-run 会拒绝启动\n' >&2
    fi
    exit 0
    ;;
  check)
    exec bash "$HOST_PROJECT_ROOT/scripts/host_check.sh" "${passthrough[@]}"
    ;;
  dry-run)
    host_template_validate || die "配置校验未通过，拒绝 dry-run（修好后重试，或显式覆盖对应变量）"
    exec bash "$HOST_PROJECT_ROOT/scripts/start_vllm.sh" --host --dry-run "${passthrough[@]}"
    ;;
  up)
    host_template_validate || die "配置校验未通过，拒绝启动（修好后重试，或显式覆盖对应变量）"
    exec bash "$HOST_PROJECT_ROOT/scripts/start_vllm.sh" --host "${passthrough[@]}"
    ;;
  up-kernel)
    host_template_validate || die "配置校验未通过，拒绝启动（修好后重试，或显式覆盖对应变量）"
    exec bash "$HOST_PROJECT_ROOT/scripts/start_vllm.sh" --host --level kernel "${passthrough[@]}"
    ;;
  down)
    exec bash "$HOST_PROJECT_ROOT/scripts/stop_vllm.sh" "${passthrough[@]}"
    ;;
  status)
    exec bash "$HOST_PROJECT_ROOT/scripts/status.sh" --port "$PORT" "${passthrough[@]}"
    ;;
  probe)
    probe_args=(--runtime host --port "$PORT")
    # probe_all.sh 不读已导出的 RUN_DIR（只会取最新目录或新建），
    # 所以显式 RUN_DIR（命令行/.env/external-probe.sh）必须转成 --run-dir。
    [[ -n "${RUN_DIR:-}" ]] && probe_args+=(--run-dir "$RUN_DIR")
    exec bash "$HOST_PROJECT_ROOT/scripts/probe_all.sh" "${probe_args[@]}" "${passthrough[@]}"
    ;;
  suite)
    exec bash "$HOST_PROJECT_ROOT/scripts/run_suite.sh" --port "$PORT" "${passthrough[@]}"
    ;;
  trace)
    exec bash "$HOST_PROJECT_ROOT/scripts/trace_capture.sh" --port "$PORT" "${passthrough[@]}"
    ;;
  pyspy)
    exec bash "$HOST_PROJECT_ROOT/scripts/pyspy.sh" --mode dump "${passthrough[@]}"
    ;;
  logs)
    runs_dir="${RUNS_DIR:-$HOST_PROJECT_ROOT/runs}"
    latest="$(find "$runs_dir" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
    [[ -n "$latest" && -f "$latest/serve.log" ]] || die "没找到 serve.log（$runs_dir 下还没有运行目录）"
    printf '# tail -f %s\n' "$latest/serve.log" >&2
    exec tail -n 100 -f "$latest/serve.log"
    ;;
esac
