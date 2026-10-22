#!/usr/bin/env bash
# 物理机启动模板的环境叠加库（被 hosts/vllm-host.sh source）。
#
# 设计目标：不改 scripts/start_vllm.sh、不改 scripts/lib/common.sh，
# 只用「显式环境变量优先」这一既有约定把模板叠上去。
#
# 叠加顺序（后者覆盖前者，但显式导出的环境变量始终最高）：
#   1) 调用者显式导出的环境变量（如 `TP=2 hosts/vllm-host.sh up ...`）
#   2) 模板链：按命令行顺序依次 source（后加载的覆盖先加载的）
#   3) 仓库根 .env（本机基线）
#
# 模板文件是普通 shell 片段，在 `set -a` 下 source：
#   * 只写 `KEY=VALUE`，可用 `${OTHER:-default}`；
#   * 可以写 `if [[ -f ... ]]; then ... fi` 这类无副作用的判断；
#   * 不要执行命令、不要 cd、不要写文件（模板要能被 `show`/`dry-run` 安全加载）。
#
# 约定：本文件只定义函数与少量只读变量，不主动加载任何配置。

HOST_TEMPLATES_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_PROJECT_ROOT="$(cd "$HOST_TEMPLATES_ROOT/.." && pwd)"
export HOST_TEMPLATES_ROOT HOST_PROJECT_ROOT

# 显式环境快照：在加载 .env / 模板之前调用一次。
declare -gA HOST_TEMPLATE_EXPLICIT_ENV=()
HOST_TEMPLATE_SNAPSHOT_TAKEN=0
HOST_TEMPLATE_LOADED=()

host_template_die() {
  printf '❌ %s\n' "$*" >&2
  return 1
}

# 记录「调用者本来就在环境里的变量」。之后的 .env / 模板都不能覆盖它们。
host_template_snapshot_explicit() {
  local name
  HOST_TEMPLATE_EXPLICIT_ENV=()
  while IFS= read -r name; do
    [[ -n "$name" ]] || continue
    HOST_TEMPLATE_EXPLICIT_ENV["$name"]="${!name}"
  done < <(compgen -e)
  HOST_TEMPLATE_SNAPSHOT_TAKEN=1
}

# 把显式环境恢复回去（模板可能覆盖了同名变量）。
host_template_restore_explicit() {
  local name
  for name in "${!HOST_TEMPLATE_EXPLICIT_ENV[@]}"; do
    export "$name=${HOST_TEMPLATE_EXPLICIT_ENV[$name]}"
  done
}

# 在 set -a 下 source 一个 env 文件；保留调用者原有的 allexport 状态。
host_template_source_file() {
  local file="$1" had_allexport=0 rc=0
  [[ $- == *a* ]] && had_allexport=1
  set -a
  # shellcheck disable=SC1090
  . "$file" || rc=$?
  [[ $had_allexport == 1 ]] || set +a
  return $rc
}

# 把模板名解析成绝对路径。
#   vendor/hygon  -> hosts/vendor/hygon.env
#   hygon         -> 在 hosts/*/hygon.env 里唯一匹配
#   xxx.env       -> 去掉后缀后同上
host_template_resolve() {
  local name="$1" candidate f
  local -a matches=()
  name="${name%.env}"
  name="${name#./}"

  if [[ "$name" == */* ]]; then
    candidate="$HOST_TEMPLATES_ROOT/$name.env"
    [[ -f "$candidate" ]] || {
      host_template_die "模板不存在：$name（找不到 $candidate）"
      return 1
    }
    printf '%s\n' "$candidate"
    return 0
  fi

  candidate="$HOST_TEMPLATES_ROOT/$name.env"
  if [[ -f "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  shopt -s nullglob
  for f in "$HOST_TEMPLATES_ROOT"/*/"$name.env"; do
    matches+=("$f")
  done
  shopt -u nullglob

  case "${#matches[@]}" in
    0)
      host_template_die "模板不存在：$name（可用 hosts/vllm-host.sh list 查看全部模板）"
      return 1
      ;;
    1)
      printf '%s\n' "${matches[0]}"
      ;;
    *)
      {
        printf '❌ 模板名有歧义：%s（请写成 目录/名字）\n' "$name"
        printf '   候选：\n'
        printf '     %s\n' "${matches[@]#"$HOST_TEMPLATES_ROOT"/}"
      } >&2
      return 1
      ;;
  esac
}

# 列出全部模板（相对路径，不带 .env），按字母序。
host_template_list() {
  local f
  shopt -s nullglob
  for f in "$HOST_TEMPLATES_ROOT"/*/*.env; do
    [[ "$(basename "$f")" == _* ]] && continue
    printf '%s\n' "${f#"$HOST_TEMPLATES_ROOT"/}" | sed 's/\.env$//'
  done | sort
  shopt -u nullglob
}

# 依次加载 .env 与模板链，最后恢复显式环境。
# 用法：host_template_apply [模板名...]
host_template_apply() {
  local name file
  [[ "${HOST_TEMPLATE_SNAPSHOT_TAKEN:-0}" == "1" ]] || host_template_snapshot_explicit

  if [[ -f "$HOST_PROJECT_ROOT/.env" ]]; then
    host_template_source_file "$HOST_PROJECT_ROOT/.env" || {
      host_template_die "加载 .env 失败：$HOST_PROJECT_ROOT/.env"
      return 1
    }
  fi

  HOST_TEMPLATE_LOADED=()
  for name in "$@"; do
    [[ -n "$name" ]] || continue
    file="$(host_template_resolve "$name")" || return 1
    host_template_source_file "$file" || {
      host_template_die "加载模板失败：$name（$file）"
      return 1
    }
    HOST_TEMPLATE_LOADED+=("${file#"$HOST_TEMPLATES_ROOT"/}")
  done

  # 追加型键（EXTRA_VLLM_ARGS / EXTRA_ENV / EXTRA_ENGINE_ARGS）：模板是「在已有值末尾
  # 追加」，而显式环境优先级最高、马上会被恢复——如果调用者显式导出了这些键，模板的
  # 追加会被整体丢弃。这是优先级约定的副作用，必须提示而不是静默（对 spec/mtp 意味着
  # 投机解码悄悄不生效）。
  local key
  for key in EXTRA_VLLM_ARGS EXTRA_ENV EXTRA_ENGINE_ARGS; do
    if [[ -v "HOST_TEMPLATE_EXPLICIT_ENV[$key]" && "${!key:-}" != "${HOST_TEMPLATE_EXPLICIT_ENV[$key]}" ]]; then
      {
        printf '⚠️  显式 %s 优先于模板：模板的追加将被丢弃（模板期望值：%s）；' \
          "$key" "${!key:-<空>}"
        printf '如需追加，请把对应参数合并进显式值\n'
      } >&2
    fi
  done

  host_template_restore_explicit

  # host 模板永远按物理机模式运行；显式给了其它值只提示，不覆盖模板语义。
  if [[ "${RUNTIME:-}" != "host" && -n "${RUNTIME:-}" ]]; then
    printf '⚠️  模板用于物理机模式，已把 RUNTIME 从 %s 覆盖为 host\n' "$RUNTIME" >&2
  fi
  export RUNTIME=host
}

# 轻量校验（启动前用；check/status 等只告警不阻断，由调用者决定）。
# 只检查「明显填错」的项，模型/驱动是否存在交给 scripts/host_check.sh。
host_template_validate() {
  local rc=0
  if [[ -z "${MODEL:-}" ]]; then
    printf '❌ MODEL 未设置：模板应给出模型目录，或用 MODEL=... 显式覆盖\n' >&2
    rc=1
  fi
  if [[ ! "${TP:-}" =~ ^[1-9][0-9]*$ ]]; then
    printf '❌ TP 必须是正整数，当前：%s\n' "${TP:-<未设置>}" >&2
    rc=1
  fi
  if [[ ! "${PORT:-}" =~ ^[0-9]+$ ]]; then
    printf '❌ PORT 必须是整数，当前：%s\n' "${PORT:-<未设置>}" >&2
    rc=1
  fi
  if [[ -n "${DRAFT_MODEL:-}" && -z "${SPEC_METHOD:-}" ]]; then
    printf '❌ 设置了 DRAFT_MODEL 但没有 SPEC_METHOD（如 dflash / mtp / eagle / ngram）\n' >&2
    rc=1
  fi
  if [[ -n "${SPEC_METHOD:-}" && -z "${DRAFT_MODEL:-}" && "${SPEC_METHOD}" != "mtp" ]]; then
    printf '⚠️  设置了 SPEC_METHOD=%s 但没有 DRAFT_MODEL；只有 mtp 允许不带草稿模型\n' "$SPEC_METHOD" >&2
  fi
  if [[ -n "${EXTRA_ARGS_FILE:-}" && ! -f "${EXTRA_ARGS_FILE}" ]]; then
    printf '❌ EXTRA_ARGS_FILE 不存在：%s\n' "$EXTRA_ARGS_FILE" >&2
    rc=1
  fi
  if [[ -n "${MODEL:-}" && ! -e "${MODEL}" ]]; then
    printf '⚠️  MODEL 路径当前不存在：%s（换机器请改模板或 MODEL=...）\n' "$MODEL" >&2
  fi
  return $rc
}

# 打印生效配置（只列排障时最关心的键）。
host_template_show() {
  local key value
  local -a keys=(
    RUNTIME ENGINE VENDOR MODEL SERVED_NAME PORT TP DTYPE GPU_MEM_UTIL
    MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS EAGER VISIBLE_DEVICES
    DRAFT_MODEL SPEC_METHOD SPEC_TOKENS SPEC_DTYPE SPEC_KV_DTYPE SPEC_EXTRA_JSON
    PROBE_LEVEL PROBE_PLUGIN PROBE_ENGINE_EXTRAS HOST_PYTHON HOST_PRELUDE_FILES HOST_PRECHECK
    SRC_DIR EXTRA_VLLM_ARGS EXTRA_ENGINE_ARGS EXTRA_ARGS_FILE EXTRA_ENV
    HOST_ENGINE_PATTERN
  )

  printf '模板链：'
  if ((${#HOST_TEMPLATE_LOADED[@]} == 0)); then
    printf '（无，只用 .env）\n'
  else
    printf '%s\n' "${HOST_TEMPLATE_LOADED[*]}"
  fi
  printf '\n'
  for key in "${keys[@]}"; do
    if [[ -v "$key" ]]; then
      value="${!key}"
    else
      value='<未设置>'
    fi
    [[ -n "$value" ]] || value='<空>'
    printf '  %-26s %s\n' "$key" "$value"
  done
}
