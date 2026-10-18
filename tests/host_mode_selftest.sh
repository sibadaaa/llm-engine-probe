#!/usr/bin/env bash
# 物理机（裸机）模式自测：RUNTIME=host 的启动/停止/状态/预检/日志分支。
#
# 特点：不需要 GPU、不需要安装 vLLM、不需要 Docker（有/无 Docker 的机器都能跑）。
#   * 用临时 MODEL 目录 + 假入口（python3 -c 'import time;time.sleep(120)'）代替真实推理；
#   * --dry-run 校验 host 分支生成的是本机命令（不是 docker run）；
#   * 真实起停一个假进程，校验 serve.pid / serve_launch.sh / cmd.txt 与 stop 的清理；
#   * host_check.sh 只在存在时校验（允许当前机器有 fail 项，但必须产出 JSON）。
#
# 用法：bash tests/host_mode_selftest.sh
# 退出码：0 = 全部通过（跳过项不算通过，但也不判失败）；非 0 = 有失败。

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PASS=0
FAIL=0
SKIP=0
TMP="$(mktemp -d /tmp/vllm-host-selftest.XXXXXX)"

chk_pass() { printf '  \033[32m✅ %s\033[0m\n' "$*"; PASS=$((PASS + 1)); }
chk_fail() { printf '  \033[31m❌ %s\033[0m\n' "$*"; FAIL=$((FAIL + 1)); }
chk_skip() { printf '  \033[33m⏭️  %s（跳过）\033[0m\n' "$*"; SKIP=$((SKIP + 1)); }
dump_log() { [[ -f "$1" ]] && sed 's/^/      /' "$1" | tail -n 25 || true; }

# 清理：先按 PID 文件温和终止假入口进程（只杀命令行里带本脚本特征串的），
# 兜底按 serve.pgid 杀整个进程组（避开自测自身所在的进程组），再删临时目录。
cleanup() {
  set +e
  local pidfile pid pgid cmdline my_pgid
  my_pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"
  for pidfile in "$TMP/run/serve.pid"; do
    [[ -f "$pidfile" ]] || continue
    pid="$(tr -dc '0-9' < "$pidfile" 2>/dev/null || true)"
    [[ -n "$pid" && "$pid" -gt 1 ]] || continue
    cmdline=""
    if [[ -r "/proc/$pid/cmdline" ]]; then
      cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    fi
    if [[ "$cmdline" == *"time.sleep(120)"* ]]; then
      pgid=""
      if [[ -f "$TMP/run/serve.pgid" ]]; then
        pgid="$(tr -dc '0-9' < "$TMP/run/serve.pgid" 2>/dev/null || true)"
      fi
      if [[ -n "$pgid" && "$pgid" -gt 1 && "$pgid" != "$$" && "$pgid" != "$my_pgid" ]]; then
        kill -TERM -- "-$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      else
        kill -TERM "$pid" 2>/dev/null || true
      fi
      for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
      if [[ -n "$pgid" && "$pgid" -gt 1 && "$pgid" != "$$" && "$pgid" != "$my_pgid" ]]; then
        kill -KILL -- "-$pgid" 2>/dev/null || true
      fi
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
  rm -rf "$TMP"
}
trap cleanup EXIT INT TERM

pid_alive() { [[ -n "${1:-}" && "$1" -gt 1 ]] && kill -0 "$1" 2>/dev/null; }

echo "==================== 物理机（host）模式自测 ===================="
echo "临时目录：$TMP"

# ---------------------------------------------------------------------------
# 0) 前置：核心脚本必须已支持 RUNTIME=host
#    未就绪时明确“跳过”，但按失败退出 —— host 自测只有在 core 合入后真实通过才算数。
# ---------------------------------------------------------------------------
if ! grep -q 'runtime_is_host' scripts/lib/common.sh 2>/dev/null; then
  echo
  echo "  ⏭️  SKIP：scripts/lib/common.sh 还没有 runtime_is_host（RUNTIME=host 尚未合入）。"
  echo "      这是开发中的临时状态：host 自测需要 core-runtime 就绪后才能真实通过，"
  echo "      因此这里按失败返回，避免“跳过 = 通过”的假绿。"
  exit 1
fi

# ---------------------------------------------------------------------------
# 准备：临时模型目录（config.json 空对象）+ 空闲端口
# ---------------------------------------------------------------------------
MODEL_DIR="$TMP/model"
mkdir -p "$MODEL_DIR"
printf '{}\n' > "$MODEL_DIR/config.json"
FREE_PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
echo "假模型：$MODEL_DIR  空闲端口：$FREE_PORT"

# ---------------------------------------------------------------------------
# 1) dry-run：host 分支输出的必须是本机命令，不是 docker run
# ---------------------------------------------------------------------------
echo
echo "---- 1) --dry-run --host：命令形态 ----"
DRY_DIR="$TMP/run-dry"
DRY_OUT="$TMP/dry-run.out"
set +e
MODEL="$MODEL_DIR" PORT="$FREE_PORT" RUNTIME=host HOST_PYTHON=python3 \
  bash scripts/start_vllm.sh --dry-run --host --skip-check --level kernel \
  --run-dir "$DRY_DIR" > "$DRY_OUT" 2>&1
dry_rc=$?
set -e
if [[ $dry_rc -eq 0 ]]; then chk_pass "start_vllm.sh --dry-run --host 退出码 0"; else
  chk_fail "start_vllm.sh --dry-run --host 退出码 $dry_rc"; dump_log "$DRY_OUT"
fi

check_dry() { # $1=描述 $2=grep 模式 $3=invert(0/1)
  local desc="$1" pattern="$2" invert="${3:-0}"
  if [[ "$invert" == "1" ]]; then
    if grep -qE "$pattern" "$DRY_OUT"; then chk_fail "$desc（不应出现：$pattern）"; dump_log "$DRY_OUT";
    else chk_pass "$desc"; fi
  else
    if grep -qE "$pattern" "$DRY_OUT"; then chk_pass "$desc"; else chk_fail "$desc（未找到：$pattern）"; dump_log "$DRY_OUT"; fi
  fi
}
check_dry "输出标明 host 运行方式" 'host'
check_dry "输出不含 docker run" 'docker[[:space:]]+run' 1
if grep -qF "$DRY_DIR/traces" "$DRY_OUT"; then
  chk_pass "输出包含 $DRY_DIR/traces 路径"
else
  chk_fail "输出缺少 $DRY_DIR/traces 路径"; dump_log "$DRY_OUT"
fi
check_dry "输出包含 exec（本机前台执行）" '(^|[^[:alnum:]_])exec([^[:alnum:]_]|$)'

# ---------------------------------------------------------------------------
# 2) 假入口真实启动 + 停止（不等待就绪、跳过预检）
# ---------------------------------------------------------------------------
echo
echo "---- 2) 假入口启动 / 停止 ----"
RUN_DIR="$TMP/run"
START_OUT="$TMP/start.out"
set +e
MODEL="$MODEL_DIR" PORT="$FREE_PORT" RUNTIME=host HOST_PYTHON=python3 \
  HOST_SERVE_ENTRYPOINT="python3 -c 'import time;time.sleep(120)'" \
  bash scripts/start_vllm.sh --host --no-wait --skip-check --run-dir "$RUN_DIR" \
  --level steps > "$START_OUT" 2>&1
start_rc=$?
set -e
if [[ $start_rc -eq 0 ]]; then chk_pass "假入口启动退出码 0"; else
  chk_fail "假入口启动退出码 $start_rc"; dump_log "$START_OUT"
fi

SERVE_PID=""
for _ in $(seq 1 50); do
  if [[ -s "$RUN_DIR/serve.pid" ]]; then SERVE_PID="$(tr -dc '0-9' < "$RUN_DIR/serve.pid" | head -c 20 || true)"; fi
  [[ -n "$SERVE_PID" ]] && break
  sleep 0.1
done
if [[ -n "$SERVE_PID" ]] && pid_alive "$SERVE_PID"; then
  chk_pass "serve.pid 已写入且进程存活（PID $SERVE_PID）"
else
  chk_fail "serve.pid 缺失或进程未存活"; dump_log "$START_OUT"
fi
[[ -f "$RUN_DIR/serve_launch.sh" ]] && chk_pass "serve_launch.sh 已生成" || { chk_fail "serve_launch.sh 未生成"; dump_log "$START_OUT"; }
[[ -s "$RUN_DIR/cmd.txt" ]] && chk_pass "cmd.txt 非空（可复现命令）" || { chk_fail "cmd.txt 缺失或为空"; dump_log "$START_OUT"; }
[[ -d "$RUN_DIR/traces" ]] && chk_pass "traces 目录已创建（$RUN_DIR/traces）" || chk_fail "traces 目录未创建"
[[ -f "$RUN_DIR/serve.log" ]] && chk_pass "serve.log 已生成" || chk_fail "serve.log 未生成"

STOP_OUT="$TMP/stop.out"
set +e
# PROBE_SKIP_VRAM_WAIT=1：假入口没占显存，跳过 stop 的显存回落等待（本机有 SMI 时能省最多 60s）
PROBE_SKIP_VRAM_WAIT=1 RUNTIME=host bash scripts/stop_vllm.sh --run-dir "$RUN_DIR" > "$STOP_OUT" 2>&1
stop_rc=$?
set -e
[[ $stop_rc -eq 0 ]] && chk_pass "stop_vllm.sh 退出码 0" || { chk_fail "stop_vllm.sh 退出码 $stop_rc"; dump_log "$STOP_OUT"; }

stopped=1
if [[ -n "$SERVE_PID" ]]; then
  for _ in $(seq 1 100); do
    if ! pid_alive "$SERVE_PID"; then stopped=0; break; fi
    sleep 0.1
  done
fi
[[ $stopped -eq 0 ]] && chk_pass "stop 后假入口进程已结束（PID $SERVE_PID）" || {
  chk_fail "stop 后进程仍存活（PID ${SERVE_PID:-未知}）"; dump_log "$STOP_OUT"; }

# ---------------------------------------------------------------------------
# 3) status：host 模式应显示“运行方式”，且无实例时也返回 0
# ---------------------------------------------------------------------------
echo
echo "---- 3) status.sh（host）----"
STATUS_OUT="$TMP/status.out"
set +e
RUNTIME=host bash scripts/status.sh --port "$FREE_PORT" > "$STATUS_OUT" 2>&1
status_rc=$?
set -e
[[ $status_rc -eq 0 ]] && chk_pass "status.sh 退出码 0（端口 $FREE_PORT 无实例）" || {
  chk_fail "status.sh 退出码 $status_rc"; dump_log "$STATUS_OUT"; }
if grep -q '运行方式' "$STATUS_OUT"; then chk_pass "status 输出包含「运行方式」"; else
  chk_fail "status 输出缺少「运行方式」"; dump_log "$STATUS_OUT"; fi

# ---------------------------------------------------------------------------
# 4) host_check.sh：存在则必须产出 host_check.json（允许有 fail 项）
# ---------------------------------------------------------------------------
echo
echo "---- 4) host_check.sh（可选）----"
HC_DIR="$TMP/hc"
if [[ -f scripts/host_check.sh ]]; then
  HC_OUT="$TMP/host-check.out"
  set +e
  RUNTIME=host bash scripts/host_check.sh --json --skip-vllm-import --skip-gpu \
    --outdir "$HC_DIR" > "$HC_OUT" 2>&1
  hc_rc=$?
  set -e
  # 允许预检项有 fail（例如本机没有 GPU），但 --json 必须落盘 host_check.json
  if [[ -f "$HC_DIR/host_check.json" ]]; then
    chk_pass "host_check.json 已生成（预检退出码 $hc_rc，允许有 fail 项）"
    if python3 -c 'import json,sys; json.load(open(sys.argv[1])); print("  JSON 可解析")' \
        "$HC_DIR/host_check.json" > "$TMP/hc_parse.out" 2>&1; then
      chk_pass "host_check.json 是合法 JSON"
    else
      chk_fail "host_check.json 不是合法 JSON"; dump_log "$TMP/hc_parse.out"
    fi
  else
    chk_fail "host_check.sh 存在但没有生成 host_check.json（退出码 $hc_rc）"; dump_log "$HC_OUT"
  fi
else
  chk_skip "scripts/host_check.sh 尚未提供"
fi

# ---------------------------------------------------------------------------
# 5) watch_steps.sh --docker：host 模式必须明确拒绝（非零退出）
# ---------------------------------------------------------------------------
echo
echo "---- 5) watch_steps.sh --docker 在 host 模式应拒绝 ----"
WATCH_OUT="$TMP/watch.out"
set +e
if command -v timeout >/dev/null 2>&1; then
  RUNTIME=host timeout 30 bash scripts/watch_steps.sh --docker --run-dir "$RUN_DIR" > "$WATCH_OUT" 2>&1
  watch_rc=$?
else
  RUNTIME=host bash scripts/watch_steps.sh --docker --run-dir "$RUN_DIR" > "$WATCH_OUT" 2>&1
  watch_rc=$?
fi
set -e
if [[ $watch_rc -eq 124 ]]; then
  chk_fail "watch_steps.sh --docker 超时（host 模式没有明确拒绝 --docker）"; dump_log "$WATCH_OUT"
elif [[ $watch_rc -ne 0 ]]; then
  chk_pass "watch_steps.sh --docker 在 host 模式非零退出（$watch_rc）"
else
  chk_fail "watch_steps.sh --docker 在 host 模式意外返回 0"; dump_log "$WATCH_OUT"
fi

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
echo
printf '\033[36m==== host 自测结果：%d 项通过 / %d 项失败 / %d 项跳过 ====\033[0m\n' "$PASS" "$FAIL" "$SKIP"
[[ "$FAIL" -eq 0 ]]
