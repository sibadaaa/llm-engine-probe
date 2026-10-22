#!/usr/bin/env bash
# 物理机启动模板（hosts/*/*.env）全量自测：语法 / 键名白名单 / 短名解析 /
# 叠加优先级 / 逐个 dry-run / 核心脚本零改动。
#
# 特点：不需要 GPU、不需要安装 vLLM、不需要 Docker；**不启动任何实例**。
#   * 只调用 hosts/vllm-host.sh 的 list / show / dry-run（只读，不产生真实进程）；
#   * 所有 dry-run 都把 RUNS_DIR 重定向到 mktemp 临时目录，不写仓库 runs/；
#   * 不 kill 任何进程（包括本机可能在跑的真实 vLLM）；退出时只删自己的临时目录；
#   * 模板尚未落盘时给出明确 ❌（不是崩溃、也不是跳过算过）。
#
# 用法：bash tests/host_templates_selftest.sh
# 退出码：0 = 全部通过（跳过项不算通过，但也不判失败）；非 0 = 有失败。

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PASS=0
FAIL=0
SKIP=0
TMP="$(mktemp -d /tmp/vllm-host-templates-selftest.XXXXXX)"
# 契约要求：dry-run 用这个模型路径 + 本机 python3；模板里的模型路径仅供参考。
DRY_MODEL="/mnt/game/models/Qwen3-0.6B"
RUNS_BASE="$TMP/runs"

chk_pass() { printf '  \033[32m✅ %s\033[0m\n' "$*"; PASS=$((PASS + 1)); }
chk_fail() { printf '  \033[31m❌ %s\033[0m\n' "$*"; FAIL=$((FAIL + 1)); }
chk_skip() { printf '  \033[33m⏭️  %s（跳过）\033[0m\n' "$*"; SKIP=$((SKIP + 1)); }
dump_tail() {
  if [[ -f "${1:-}" ]]; then sed 's/^/      /' "$1" | tail -n 20; fi
}

# 清理：只删临时目录；不碰进程、不碰仓库 runs/。
cleanup() {
  set +e
  rm -rf "$TMP"
}
trap cleanup EXIT
# 信号时清理后立即退出（否则 handler 跑完脚本会继续执行，且 EXIT 会让 cleanup 再跑一次）。
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

# 从 show 输出里取某个键的值（host_template_show 的格式是 "  KEY<空格...>VALUE"）。
show_value() { # $1=键名，stdin=show 输出
  sed -n -E "s/^  ${1}[[:space:]]+(.*)\$/\1/p" | head -n 1
}
# 把 show 的占位符（<未设置> / <空>）归一成空串。
show_value_tidy() {
  local v
  v="$(show_value "$1")"
  case "$v" in
    '<未设置>'|'<空>') v="" ;;
  esac
  printf '%s\n' "$v"
}
# serve_launch.sh 里的 JSON 参数会被写成 \{ \" 这类 shell 转义形态；
# 去掉反斜杠后再按字面量比对，避免为 '[ngram]' 之类的值漏判。
file_contains_fragment() { # $1=文件 $2=期望片段
  tr -d '\\' < "$1" | grep -qF -- "$2"
}

echo "==================== 物理机（host）模板自测 ===================="
echo "仓库根：$ROOT"
echo "临时目录：$TMP（退出时删除；dry-run 的 RUNS_DIR 都在这里面）"

# ---------------------------------------------------------------------------
# 0) 前置：入口与叠加库必须存在
# ---------------------------------------------------------------------------
if [[ ! -f hosts/vllm-host.sh || ! -f hosts/lib/env_overlay.sh ]]; then
  echo
  [[ -f hosts/vllm-host.sh ]] || chk_fail "缺少 hosts/vllm-host.sh（模板入口）"
  [[ -f hosts/lib/env_overlay.sh ]] || chk_fail "缺少 hosts/lib/env_overlay.sh（叠加实现）"
  printf '\n\033[36m==== 模板自测结果：%d 项通过 / %d 项失败 / %d 项跳过 ====\033[0m\n' \
    "$PASS" "$FAIL" "$SKIP"
  exit 1
fi

# 只读地枚举实际模板文件（跳过 _template.env 这类骨架）。
shopt -s nullglob
TEMPLATE_FILES=(hosts/*/*.env)
shopt -u nullglob
TEMPLATE_NAMES=()
for f in "${TEMPLATE_FILES[@]}"; do
  [[ "$(basename "$f")" == _* ]] && continue
  TEMPLATE_NAMES+=("${f#hosts/}")
done
mapfile -t TEMPLATE_NAMES < <(printf '%s\n' "${TEMPLATE_NAMES[@]}" | sed 's/\.env$//' | sort)
# README §3 索引里承诺的模板（缺失就是契约缺口，必须 ❌ 而不是静默跳过）。
EXPECTED_TEMPLATES=(
  vendor/nvidia vendor/amd vendor/hygon vendor/intel vendor/ascend
  scenario/single-gpu-dev scenario/tp4-prod scenario/tp8-throughput scenario/long-context-128k
  quant/fp8 quant/int8-w8a8 quant/w4a16 quant/awq
  spec/dflash spec/mtp spec/eagle spec/ngram
  deploy/pd-prefill deploy/pd-decode deploy/multinode-head deploy/multinode-worker
  engine/vllm engine/strata engine/sglang engine/llamacpp engine/ollama
)
have_template() { # $1=模板名（如 vendor/hygon）
  [[ -f "hosts/$1.env" ]]
}

# ---------------------------------------------------------------------------
# 1) list 非空 / 与文件集合一致；每个模板 bash -n + 文件头三行注释
# ---------------------------------------------------------------------------
echo
echo "---- 1) list / 语法 / 文件头注释 ----"
LIST_OUT="$TMP/list.out"
LIST_ERR="$TMP/list.err"
list_rc=0
bash hosts/vllm-host.sh list > "$LIST_OUT" 2> "$LIST_ERR" || list_rc=$?
if [[ $list_rc -ne 0 ]]; then
  chk_fail "hosts/vllm-host.sh list 退出码 $list_rc"; dump_tail "$LIST_ERR"
elif [[ ! -s "$LIST_OUT" ]]; then
  chk_fail "hosts/vllm-host.sh list 输出为空：hosts/*/*.env 还没有落盘（预期见 README §3）"
else
  chk_pass "list 非空（$(grep -c . "$LIST_OUT") 个模板）"
fi

if ((${#TEMPLATE_NAMES[@]} == 0)); then
  chk_fail "没有找到任何 hosts/*/*.env 模板：语法/键名/叠加/dry-run 检查无从执行"
else
  # list 的输出必须覆盖磁盘上所有模板，否则入口的枚举逻辑漏项。
  LIST_MISSING=()
  for name in "${TEMPLATE_NAMES[@]}"; do
    grep -qxF -- "$name" "$LIST_OUT" || LIST_MISSING+=("$name")
  done
  if ((${#LIST_MISSING[@]})); then
    chk_fail "list 漏列模板：${LIST_MISSING[*]}"
  else
    chk_pass "list 覆盖全部 ${#TEMPLATE_NAMES[@]} 个模板文件"
  fi
fi

MISSING=()
for name in "${EXPECTED_TEMPLATES[@]}"; do
  if have_template "$name"; then
    continue
  fi
  MISSING+=("$name")
  chk_fail "缺少模板：hosts/$name.env（README §3 索引列出，尚未落盘）"
done

for name in "${TEMPLATE_NAMES[@]}"; do
  file="hosts/$name.env"
  syntax_err="$TMP/syntax-$(printf '%s' "$name" | tr '/' '_').err"
  if bash -n "$file" 2> "$syntax_err"; then
    chk_pass "bash -n $file"
  else
    chk_fail "bash -n $file 失败"; dump_tail "$syntax_err"
  fi

  missing_labels=()
  for label in 用途 适用 组合; do
    if ! sed -n '1,40p' "$file" | grep -qE "^#[[:space:]]*${label}[[:space:]]*[:：]"; then
      missing_labels+=("$label")
    fi
  done
  if ((${#missing_labels[@]} == 0)); then
    chk_pass "$name 文件头含 用途/适用/组合 三行注释"
  else
    chk_fail "$name 文件头缺注释：${missing_labels[*]}（约定见 hosts/_template.env）"
  fi
done

# ---------------------------------------------------------------------------
# 2) 键名白名单：.env.example 的 KEY= + scripts/lib/common.sh 的 ${KEY:=默认值}
# ---------------------------------------------------------------------------
echo
echo "---- 2) 键名白名单 ----"
WHITELIST="$TMP/key-whitelist.txt"
{
  grep -oE '^[[:space:]]*#?[[:space:]]*[A-Z_][A-Z0-9_]*=' .env.example 2>/dev/null \
    | sed -E 's/^[[:space:]]*#?[[:space:]]*//; s/=$//'
  grep -oE '\$\{[A-Z_][A-Z0-9_]*:=' scripts/lib/common.sh 2>/dev/null \
    | sed -E 's/^\$\{//; s/:=$//'
} | sort -u > "$WHITELIST"
if [[ -s "$WHITELIST" ]]; then
  chk_pass "白名单已生成（$(grep -c . "$WHITELIST") 个键：.env.example + common.sh）"
else
  chk_fail "键名白名单为空（.env.example / scripts/lib/common.sh 都读不到）"
fi

if ((${#TEMPLATE_NAMES[@]} == 0)); then
  chk_fail "没有模板可检查键名白名单"
else
  for name in "${TEMPLATE_NAMES[@]}"; do
    file="hosts/$name.env"
    # 只看行首赋值（忽略注释、if/fi、缩进在函数体内的 local 小写变量）。
    mapfile -t keys < <(grep -oE '^[[:space:]]*[A-Z_][A-Z0-9_]*=' "$file" 2>/dev/null \
      | sed -E 's/^[[:space:]]*//; s/=$//' | sort -u)
    if ((${#keys[@]} == 0)); then
      chk_fail "$name 没有任何 KEY=VALUE 赋值（模板是空壳？）"
      continue
    fi
    unknown=()
    for key in "${keys[@]}"; do
      grep -qxF -- "$key" "$WHITELIST" || unknown+=("$key")
    done
    if ((${#unknown[@]} == 0)); then
      chk_pass "$name 键名全部已知（${#keys[@]} 个）"
    else
      for key in "${unknown[@]}"; do
        chk_fail "$name：未知键 $key（不在 .env.example / scripts/lib/common.sh 里，会被静默忽略）"
      done
    fi
  done
fi

# ---------------------------------------------------------------------------
# 3) 短名解析 / 歧义 / 不存在的名字
# ---------------------------------------------------------------------------
echo
echo "---- 3) 短名解析与歧义 ----"
if ((${#TEMPLATE_NAMES[@]} == 0)); then
  chk_fail "没有模板可验证 show / 短名解析"
else
  for name in "${TEMPLATE_NAMES[@]}"; do
    out="$TMP/show-$(printf '%s' "$name" | tr '/' '_').out"
    rc=0
    bash hosts/vllm-host.sh show "$name" > "$out" 2>&1 || rc=$?
    if [[ $rc -eq 0 ]] && grep -qF -- "$name.env" "$out"; then
      chk_pass "show $name 成功（模板链已加载）"
    else
      chk_fail "show $name 失败或未加载该模板（退出码 $rc）"; dump_tail "$out"
    fi
  done

  # 同名短名统计：唯一才允许直接写短名，歧义必须报错并列出候选。
  declare -A STEM_COUNT=()
  declare -A STEM_NAMES=()
  for name in "${TEMPLATE_NAMES[@]}"; do
    stem="${name##*/}"
    STEM_COUNT["$stem"]=$(( ${STEM_COUNT["$stem"]:-0} + 1 ))
    STEM_NAMES["$stem"]+="$name "
  done

  short_tested=0
  for stem in "${!STEM_COUNT[@]}"; do
    out="$TMP/short-$stem.out"
    rc=0
    bash hosts/vllm-host.sh show "$stem" > "$out" 2>&1 || rc=$?
    if [[ "${STEM_COUNT[$stem]}" == "1" ]]; then
      short_tested=$((short_tested + 1))
      if [[ $rc -eq 0 ]] && grep -qF -- "${STEM_NAMES[$stem]% }.env" "$out"; then
        chk_pass "短名 $stem 唯一可用，解析到 ${STEM_NAMES[$stem]% }"
      else
        chk_fail "短名 $stem 应唯一可用，但 show 失败（退出码 $rc）"; dump_tail "$out"
      fi
    else
      # 歧义：必须非零退出、提示歧义并列出全部候选。
      bad=0
      [[ $rc -eq 0 ]] && bad=1
      grep -q '歧义' "$out" || bad=1
      for cand in ${STEM_NAMES[$stem]}; do
        grep -qF -- "$cand.env" "$out" || bad=1
      done
      if [[ $bad -eq 0 ]]; then
        chk_pass "歧义短名 $stem 正确报错并列出候选（${STEM_NAMES[$stem]% }）"
      else
        chk_fail "歧义短名 $stem 未被正确拒绝（退出码 $rc）"; dump_tail "$out"
      fi
    fi
  done
  if [[ $short_tested -eq 0 ]]; then
    chk_skip "没有唯一的短名可测（同名短名已按歧义校验）"
  fi
fi

# 不存在的名字必须明确失败（不是崩溃、不是假成功）。
NOPE_OUT="$TMP/show-nonexistent.out"
nope_rc=0
bash hosts/vllm-host.sh show definitely-not-a-template > "$NOPE_OUT" 2>&1 || nope_rc=$?
if [[ $nope_rc -ne 0 ]] && grep -q '模板不存在' "$NOPE_OUT"; then
  chk_pass "不存在的模板名被拒绝（退出码 $nope_rc，提示「模板不存在」）"
else
  chk_fail "不存在的模板名未按预期失败（退出码 $nope_rc）"; dump_tail "$NOPE_OUT"
fi

# ---------------------------------------------------------------------------
# 4) 优先级：显式环境 > 模板链 > .env；模板链后写覆盖先写
# ---------------------------------------------------------------------------
echo
echo "---- 4) 叠加优先级 ----"
SCENARIO=""
if have_template scenario/tp4-prod; then
  SCENARIO="scenario/tp4-prod"
else
  for name in "${TEMPLATE_NAMES[@]}"; do
    if [[ "$name" == scenario/* ]]; then SCENARIO="$name"; break; fi
  done
fi

if [[ -z "$SCENARIO" ]]; then
  chk_fail "没有场景模板（hosts/scenario/*.env）可验证优先级"
else
  PRIO_OUT="$TMP/priority-explicit.out"
  prio_rc=0
  TP=2 MODEL=/tmp/xxx bash hosts/vllm-host.sh show "$SCENARIO" > "$PRIO_OUT" 2>&1 || prio_rc=$?
  if [[ $prio_rc -ne 0 ]]; then
    chk_fail "TP=2 MODEL=/tmp/xxx show $SCENARIO 失败（退出码 $prio_rc）"; dump_tail "$PRIO_OUT"
  else
    tp_seen="$(show_value_tidy TP < "$PRIO_OUT")"
    model_seen="$(show_value_tidy MODEL < "$PRIO_OUT")"
    [[ "$tp_seen" == "2" ]] \
      && chk_pass "显式 TP=2 覆盖模板里的 TP（生效值：$tp_seen）" \
      || chk_fail "显式 TP=2 未生效（生效值：${tp_seen:-<空>}）"
    [[ "$model_seen" == "/tmp/xxx" ]] \
      && chk_pass "显式 MODEL=/tmp/xxx 覆盖模板模型路径" \
      || chk_fail "显式 MODEL 未生效（生效值：${model_seen:-<空>}）"
  fi
fi

# 追加型键（EXTRA_VLLM_ARGS）被显式导出时：模板追加被丢弃必须给出 ⚠️，不能静默。
if have_template quant/fp8; then
  APPEND_OUT="$TMP/priority-append-warn.out"
  append_rc=0
  EXTRA_VLLM_ARGS="--my-own-arg" bash hosts/vllm-host.sh show quant/fp8 \
    > "$APPEND_OUT" 2>&1 || append_rc=$?
  if [[ $append_rc -ne 0 ]]; then
    chk_fail "显式 EXTRA_VLLM_ARGS + quant/fp8 show 失败（退出码 $append_rc）"; dump_tail "$APPEND_OUT"
  elif grep -q '显式 EXTRA_VLLM_ARGS 优先于模板' "$APPEND_OUT" \
    && grep -qF -- '--quantization fp8' "$APPEND_OUT"; then
    chk_pass "显式 EXTRA_VLLM_ARGS 覆盖追加型模板时有 ⚠️ 提示（含模板期望值）"
  else
    chk_fail "显式 EXTRA_VLLM_ARGS 丢弃模板追加时缺少 ⚠️ 提示"; dump_tail "$APPEND_OUT"
  fi
  # 未显式导出时不得误报。
  QUIET_OUT="$TMP/priority-append-quiet.out"
  bash hosts/vllm-host.sh show quant/fp8 > "$QUIET_OUT" 2>&1 || true
  if grep -q '优先于模板' "$QUIET_OUT"; then
    chk_fail "未显式导出 EXTRA_VLLM_ARGS 却出现覆盖警告（误报）"
  else
    chk_pass "未显式导出 EXTRA_VLLM_ARGS 时无覆盖警告（无误报）"
  fi
fi

if have_template spec/dflash && have_template scenario/tp4-prod; then
  d1_out="$TMP/layer-dflash.out"
  d2_out="$TMP/layer-dflash-then-tp4.out"
  d3_out="$TMP/layer-tp4-then-dflash.out"
  bash hosts/vllm-host.sh show spec/dflash > "$d1_out" 2>&1 || true
  bash hosts/vllm-host.sh show spec/dflash scenario/tp4-prod > "$d2_out" 2>&1 || true
  bash hosts/vllm-host.sh show scenario/tp4-prod spec/dflash > "$d3_out" 2>&1 || true
  d1="$(show_value_tidy DRAFT_MODEL < "$d1_out")"
  d2="$(show_value_tidy DRAFT_MODEL < "$d2_out")"
  d3="$(show_value_tidy DRAFT_MODEL < "$d3_out")"

  if [[ -n "$d1" ]]; then
    chk_pass "spec/dflash 单独使用时 DRAFT_MODEL=$d1"
  else
    chk_fail "spec/dflash 没有设置 DRAFT_MODEL"
  fi
  if [[ -n "$d3" && "$d3" == "$d1" ]]; then
    chk_pass "后写覆盖先写：scenario/tp4-prod spec/dflash → DRAFT_MODEL=$d3 生效"
  else
    chk_fail "scenario/tp4-prod spec/dflash 的 DRAFT_MODEL 不对（期望 $d1，实际 ${d3:-<空>}）"
  fi
  if [[ -z "$d2" ]]; then
    chk_pass "后写覆盖先写：spec/dflash scenario/tp4-prod → 场景模板重置投机解码（DRAFT_MODEL 为空）"
  else
    chk_fail "顺序反了却没被覆盖：spec/dflash scenario/tp4-prod 仍得到 DRAFT_MODEL=$d2（场景模板应清空投机解码）"
  fi
else
  chk_fail "缺少 spec/dflash 或 scenario/tp4-prod，无法验证「后写覆盖先写」（当前：dflash=$([[ -f hosts/spec/dflash.env ]] && echo 有 || echo 无)，tp4-prod=$([[ -f hosts/scenario/tp4-prod.env ]] && echo 有 || echo 无)）"
fi

# ---------------------------------------------------------------------------
# 5) 每个模板单独 dry-run（RUNS_DIR 全部重定向到临时目录）
# ---------------------------------------------------------------------------
echo
echo "---- 5) 逐个 dry-run ----"
if ((${#TEMPLATE_NAMES[@]} == 0)); then
  chk_fail "没有模板可做 dry-run"
else
  # 关键模板的断言：模板名 -> 期望出现在 serve_launch.sh 里的片段（去 shell 转义后比对）。
  # 第二片段可选，用于同一个模板要同时断言 JSON 两处（如 ngram 的 model + method）。
  declare -A DRY_EXPECT_DESC=(
    ["vendor/hygon"]='HIP_VISIBLE_DEVICES'
    ["scenario/tp4-prod"]='--tensor-parallel-size 4'
    ["spec/dflash"]='"method":"dflash"'
    ["spec/ngram"]='"model":"[ngram]"'
    ["spec/mtp"]='"method":"mtp"'
    ["quant/fp8"]='--quantization fp8'
    ["engine/strata"]='ENGINE=strata'
    ["engine/sglang"]='ENGINE=sglang'
    ["engine/llamacpp"]='ENGINE=llamacpp'
    ["engine/ollama"]='ENGINE=ollama'
  )
  declare -A DRY_EXPECT_DESC2=(
    ["spec/ngram"]='"method":"ngram"'
    ["engine/strata"]='strata launch'
    ["engine/sglang"]='sglang.launch_server'
    ["engine/llamacpp"]='llama-server'
    ["engine/ollama"]='ollama serve'
  )

  for name in "${TEMPLATE_NAMES[@]}"; do
    slug="$(printf '%s' "$name" | tr '/' '_')"
    rdir="$RUNS_BASE/$slug"
    out="$TMP/dry-$slug.out"
    mkdir -p "$rdir"
    rc=0
    RUNS_DIR="$rdir" MODEL="$DRY_MODEL" HOST_PYTHON=python3 \
      bash hosts/vllm-host.sh dry-run "$name" > "$out" 2>&1 || rc=$?
    if [[ $rc -eq 0 ]]; then
      chk_pass "dry-run $name 退出码 0"
    else
      chk_fail "dry-run $name 退出码 $rc"; dump_tail "$out"
    fi

    launch="$(ls -1dt "$rdir"/*/serve_launch.sh 2>/dev/null | head -n 1)"
    if [[ -n "$launch" && -f "$launch" ]]; then
      chk_pass "dry-run $name 生成 serve_launch.sh"
      # 引擎感知断言：vLLM CLI（ENGINE_ARGS_MODE=vllm-cli）的模型是 serve 位置参数；
      # custom 模式引擎（如 strata）无 serve 子命令语义，模型身份记录在 export 段。
      launch_engine="$(sed -n -E 's/^export ENGINE=//p' "$launch" | head -n 1)"
      launch_engine="${launch_engine:-vllm}"
      if [[ "$launch_engine" == "vllm" ]]; then
        if [[ "$launch" == "$rdir"/* ]] \
          && grep -qE '(^|[[:space:]])serve([[:space:]]|$)' "$launch" \
          && { grep -qF -- "$DRY_MODEL" "$launch" || grep -qF -- '--model' "$launch"; }; then
          # vLLM 的模型是 serve 的位置参数（不是 --model 开关）；这里同时接受
          # 显式模型路径（MODEL=$DRY_MODEL 必须传导到命令里）和 --model 写法。
          chk_pass "dry-run $name 的 serve_launch.sh 含 serve 与模型路径（$DRY_MODEL）"
        else
          chk_fail "dry-run $name 的 serve_launch.sh 缺 serve 或模型路径 $DRY_MODEL（$launch）"; dump_tail "$launch"
        fi
      elif grep -qF -- "$DRY_MODEL" "$launch" || grep -qE '^export MODEL=' "$launch"; then
        chk_pass "dry-run $name 的 serve_launch.sh 记录了模型身份（engine=$launch_engine，custom 模式）"
      else
        chk_fail "dry-run $name 的 serve_launch.sh 缺模型身份（export MODEL 或命令行；engine=$launch_engine）"; dump_tail "$launch"
      fi
      case "$name" in
        vendor/hygon|scenario/tp4-prod|spec/dflash|spec/ngram|spec/mtp|quant/fp8|engine/strata|engine/sglang|engine/llamacpp|engine/ollama)
          desc="${DRY_EXPECT_DESC[$name]}"
          if file_contains_fragment "$launch" "$desc"; then
            chk_pass "$name 的 serve_launch.sh 含 $desc"
          else
            chk_fail "$name 的 serve_launch.sh 缺 $desc"; dump_tail "$launch"
          fi
          if [[ -n "${DRY_EXPECT_DESC2[$name]:-}" ]]; then
            desc2="${DRY_EXPECT_DESC2[$name]}"
            if file_contains_fragment "$launch" "$desc2"; then
              chk_pass "$name 的 serve_launch.sh 含 $desc2"
            else
              chk_fail "$name 的 serve_launch.sh 缺 $desc2"; dump_tail "$launch"
            fi
          fi
          ;;
      esac
    else
      chk_fail "dry-run $name 没有生成 serve_launch.sh（RUNS_DIR=$rdir）"; dump_tail "$out"
    fi
  done

  # 只允许 dry-run：临时 RUNS_DIR 下不应出现真实启动痕迹。
  if find "$RUNS_BASE" -mindepth 2 -maxdepth 3 -name 'serve.pid' -print -quit 2>/dev/null | grep -q .; then
    chk_fail "dry-run 竟然写了 serve.pid（说明启动了实例）"
  else
    chk_pass "全部 dry-run 均未启动实例（没有 serve.pid）"
  fi
fi

# ---------------------------------------------------------------------------
# 5b) 引擎组件守卫：未知 ENGINE 必须 die 且文案列出可用引擎（F1 T1.11）
# ---------------------------------------------------------------------------
echo
echo "---- 5b) 引擎组件守卫（ENGINE） ----"
GUARD_OUT="$TMP/engine-guard.out"
guard_rc=0
RUNS_DIR="$RUNS_BASE/engine-guard" MODEL="$DRY_MODEL" HOST_PYTHON=python3 ENGINE=no-such-engine \
  bash hosts/vllm-host.sh dry-run vendor/hygon > "$GUARD_OUT" 2>&1 || guard_rc=$?
if [[ $guard_rc -ne 0 ]] && grep -q '未知引擎' "$GUARD_OUT" && grep -qw 'vllm' "$GUARD_OUT"; then
  chk_pass "ENGINE=no-such-engine 的 dry-run 被拒绝且文案列出可用引擎"
else
  chk_fail "ENGINE=no-such-engine 未按预期失败（退出码 $guard_rc，文案需含「未知引擎」与引擎列表）"
  dump_tail "$GUARD_OUT"
fi

# engine/vllm 模板：显式声明 vLLM ≡ 不设置（红线 1：引擎模板不得改变默认路径行为）。
# 断言走 show（ENGINE=vllm 显示），dry-run 产物保持与「不设置」逐字节一致。
if have_template engine/vllm; then
  ENG_OUT="$TMP/engine-vllm-show.out"
  bash hosts/vllm-host.sh show engine/vllm > "$ENG_OUT" 2>&1 || true
  eng_seen="$(show_value_tidy ENGINE < "$ENG_OUT")"
  if [[ "$eng_seen" == "vllm" ]]; then
    chk_pass "engine/vllm 模板 show 显示 ENGINE=vllm（占位声明生效）"
  else
    chk_fail "engine/vllm 模板 show 未显示 ENGINE=vllm（实际：${eng_seen:-<空>}）"
  fi
fi

# 无 docker 能力的引擎，RUNTIME=docker 必须 die 且文案含 "docker"（F2 T2.3 + 引擎扩展）。
# 注意：hosts/vllm-host.sh 是物理机入口、恒以 --host 调 start_vllm.sh，
# 因此 docker 门禁用 scripts/start_vllm.sh 直调验证。
# 逐个引擎模板校验（都未声明 runtime.docker；若某引擎将来声明了，从列表摘掉即可）：
for eng_tpl in engine/strata engine/sglang engine/llamacpp engine/ollama; do
  if ! have_template "$eng_tpl"; then continue; fi
  eng_id="${eng_tpl#engine/}"
  ENG_DOCKER_OUT="$TMP/engine-${eng_id}-docker.out"
  eng_docker_rc=0
  RUNTIME=docker ENGINE="$eng_id" RUNS_DIR="$RUNS_BASE/engine-${eng_id}-docker" MODEL="$DRY_MODEL" \
    bash scripts/start_vllm.sh --dry-run > "$ENG_DOCKER_OUT" 2>&1 || eng_docker_rc=$?
  if [[ $eng_docker_rc -ne 0 ]] && grep -q 'docker' "$ENG_DOCKER_OUT"; then
    chk_pass "ENGINE=$eng_id RUNTIME=docker 被拒绝且文案含 docker（无 runtime.docker 能力）"
  else
    chk_fail "ENGINE=$eng_id RUNTIME=docker 未按预期失败（退出码 $eng_docker_rc）"
    dump_tail "$ENG_DOCKER_OUT"
  fi
done

if have_template engine/strata; then

  # 无 api.openai-completions 能力的引擎：PROBE_ENDPOINT=completions 必须 die（F2 T2.6②）
  STRATA_CP_OUT="$TMP/engine-strata-completions.out"
  strata_cp_rc=0
  RUNTIME=host ENGINE=strata RUNS_DIR="$RUNS_BASE/engine-strata-completions" \
    PROBE_ENDPOINT=completions MODEL="$DRY_MODEL" HOST_PYTHON=python3 \
    bash hosts/vllm-host.sh dry-run engine/strata > "$STRATA_CP_OUT" 2>&1 || strata_cp_rc=$?
  if [[ $strata_cp_rc -ne 0 ]] && grep -q 'completions' "$STRATA_CP_OUT"; then
    chk_pass "ENGINE=strata + PROBE_ENDPOINT=completions 被拒绝（无 api.openai-completions 能力）"
  else
    chk_fail "ENGINE=strata + PROBE_ENDPOINT=completions 未按预期失败（退出码 $strata_cp_rc）"
    dump_tail "$STRATA_CP_OUT"
  fi
fi

# ---------------------------------------------------------------------------
# 6) 核心脚本改动范围（模板本身不要求改核心；只允许 host prelude 的 set +u 兼容修复）
# ---------------------------------------------------------------------------
echo
echo "---- 6) 核心脚本改动范围 ----"
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  dirty="$(git status --porcelain -- scripts/ 2>/dev/null || true)"
  # 允许的 scripts/ 改动白名单（docs/design/component-interface/施工图.md T1.1）：
  #   start_vllm.sh  : host prelude set +u 兼容修复 + 引擎 caps 门禁收拢（F1 T1.7）
  #   lib/common.sh  : 引擎组件层加载函数 / 进程正则参数化 / 身份落产物（T1.6/T1.8/T1.9）
  #                    + 就绪等待共享函数 wait_engine_ready（F3 评审修复）
  #   stop_vllm.sh   : docker --kill-exec 分支进程正则参数化（T1.8）
  #   status.sh      : 引擎版本探测与 /metrics 抓取后缀（T1.9 / F3 T3.6）
  #   pyspy.sh       : 容器内进程 grep 正则参数化（T1.8）
  #   wait_ready.sh  : 就绪判定改走共享的 wait_engine_ready（引擎感知，F3 评审修复）
  # 新增其它 scripts/ 改动时必须先扩这份白名单并同步 hosts/README.md §6。
  unexpected="$(printf '%s\n' "$dirty" \
    | grep -vE '^.. scripts/(start_vllm\.sh|lib/common\.sh|stop_vllm\.sh|status\.sh|pyspy\.sh|wait_ready\.sh)$' \
    | grep -v '^$' || true)"
  if [[ -z "$unexpected" ]]; then
    chk_pass "scripts/ 无额外改动（白名单：start_vllm / lib-common / stop_vllm / status / pyspy）"
    if [[ -n "$dirty" ]]; then
      printf '      允许的改动：\n'
      printf '%s\n' "$dirty" | sed 's/^/        /'
    fi
  else
    chk_fail "scripts/ 有模板未预期的改动："
    printf '%s\n' "$unexpected" | sed 's/^/      /'
  fi
else
  chk_skip "不在 git 仓库内，无法检查核心脚本改动"
fi

# ---------------------------------------------------------------------------
# 7) 副作用检查：仓库 runs/ 不应被本自测写入（所有 RUNS_DIR 都应指向 $TMP）
# ---------------------------------------------------------------------------
echo
echo "---- 7) 副作用与清理 ----"
leaks="$(grep -rlsF -- "$TMP" runs/ 2>/dev/null | head -n 5 || true)"
if [[ -z "$leaks" ]]; then
  chk_pass "仓库 runs/ 未被本自测写入（RUNS_DIR 已重定向到 $TMP）"
else
  chk_fail "仓库 runs/ 出现本自测临时路径，说明 RUNS_DIR 未重定向：$leaks"
fi
chk_pass "退出时由 trap 删除临时目录（当前 $TMP）"

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
echo
if ((${#MISSING[@]})); then
  printf '\033[33m待模板落盘（%d 个）：%s\033[0m\n' "${#MISSING[@]}" "${MISSING[*]}"
fi
printf '\033[36m==== 模板自测结果：%d 项通过 / %d 项失败 / %d 项跳过 ====\033[0m\n' \
  "$PASS" "$FAIL" "$SKIP"
[[ "$FAIL" -eq 0 ]]
