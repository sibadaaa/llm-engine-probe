"""引擎组件注册表（Python 端）：探针层的引擎身份与能力位（caps）一致来源。

与 Shell 端的 ``engines/<id>.sh`` 一一对应（键集合由 tests/test_engines.py 校验，
模式复刻 ``vendors/*.sh`` ↔ 本仓库 ``probe/vendors.py`` 的双端守卫）：

* Shell 端（``engines/<id>.sh``）负责**启动适配**：serve 入口、进程清理正则、
  就绪指标、能力位声明（``ENGINE_CAPS``）——被 ``scripts/start_vllm.sh`` 消费。
* Python 端（本模块）负责**探测适配**：探针按能力位（caps）协商行为——
  metrics 抓取、http 请求体、报告降级注记、hostcheck 引擎预检。

能力键是**封闭词表**（``CAP_VOCAB``，语义见
``docs/design/component-interface/设计.md`` §3）：两端注册表只能声明词表内的键，
``tests/test_engines.py`` 会拦下拼写错误导致的静默失效。

分层约定（F3 起）：``EngineProfile`` 里还有一批 **Python 侧专有** 的探测字段
（``metric_prefix`` / ``chat_body_extras`` / ``delta_content_keys`` /
``env_check_metrics`` / ``extra_metric_families`` / ``diagnose_overrides``），
它们描述"探针如何打这个引擎"，Shell 端（启动/清理）用不到，因此**不进**
``engines/<id>.sh``，也不参与 tests/test_engines.py 的双端一致性守卫。

新增引擎的完整流程见 ``docs/design/README.md`` §4（三份新文件 + 两行索引）。
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# 能力键词表（封闭集合；新增键必须同步更新 设计.md §3 表格与 tests/test_engines.py）
# ---------------------------------------------------------------------------
# api.*    : HTTP API 面（探针按此决定哪些端点可用）
# metrics.*: /metrics 的输出形态
# obs.*    : 观测能力（逐 step 日志 / 逐请求指标 / profiler / 图模式）
# spec.decode: 引擎自带投机解码配置面
# runtime.*: 运行方式支持（docker 镜像 / 物理机拉起）
CAP_VOCAB: frozenset[str] = frozenset({
    "api.openai-chat",
    "api.openai-completions",
    "api.models-list",
    "api.health",
    "api.props",
    "metrics.vllm-named",
    "metrics.prometheus",
    "obs.step-log",
    "obs.per-request-metrics",
    "obs.spec-metrics",
    "obs.trace-pytorch",
    "obs.graph-mode",
    "spec.decode",
    "runtime.docker",
    "runtime.host",
})

METRICS_FLAVORS = ("vllm-named", "prometheus", "none")
ARGS_MODES = ("vllm-cli", "custom")
CHECK_SEVERITIES = ("warn", "fail")


@dataclass(frozen=True)
class EngineProfile:
    """一个引擎组件在探测侧的全部静态契约。

    字段语义见 ``docs/design/component-interface/设计.md`` §4.1/§5；
    与 Shell 端 ``engines/<id>.sh`` 的 ``ENGINE_*`` 变量一一对应
    （双端一致由 tests/test_engines.py 强制）。
    """

    id: str
    label: str
    # 能力位（封闭词表 CAP_VOCAB 的子集）；缺失能力时探针按 设计.md §3 的
    # 「缺失时的探针行为」列显式降级，不做静默。
    caps: frozenset[str]
    metrics_flavor: str = "vllm-named"     # vllm-named | prometheus | none
    args_mode: str = "vllm-cli"            # vllm-cli | custom
    entrypoint: str = ""                   # host 模式默认 serve 入口（HOST_SERVE_ENTRYPOINT 可覆盖）
    process_pattern: str = ""              # 进程清理正则（空 → 回落 HOST_VLLM_PATTERN）
    health_path: str = "/health"
    models_path: str = "/v1/models"
    ready_metric: str = ""                 # 就绪轮询 /metrics 的正则；空 → 回落厂商默认
    metrics_fetch_suffix: str = ""         # 抓 /metrics 追加的查询串（如 "?format=prometheus"）
    version_cmd: str = ""                  # hostcheck 版本探测命令（空 → 走 HTTP 元数据/import 探测）
    check_severity: str = "fail"           # 引擎预检缺失时的处置：warn | fail
    docker_image: str = ""                 # 空 = 无 docker 支持（RUNTIME=docker 下显式拒绝）
    # 报告里按引擎固定输出的一句说明（能力降级注记的补充；vllm 为空 = 基线不变）
    report_notes: tuple[str, ...] = ()
    notes: str = ""

    # ------------------------------------------------------------------
    # F3 · Python 侧专有探测字段（无 shell 对应物，不参与 tests/test_engines.py
    # 的双端一致性；语义见 docs/design/observability-compat/设计.md §2）
    # ------------------------------------------------------------------
    # 指标名前缀过滤（save_tsv/diff_snapshots/--print 用）；空 → 按 metrics_flavor 派生：
    # vllm-named → "vllm:"，prometheus → ""（不过滤），none → 不使用。
    metric_prefix: str = ""
    # chat 请求体的引擎专属注入（引擎无关的 model/max_tokens/... 之外）；vllm 取值
    # = 历史硬编码行为（引擎请求体逐字节一致），strata 用顶层 reasoning_effort。
    chat_body_extras: dict[str, Any] = field(default_factory=dict)
    # 流式 delta 的内容键序列（TTFT/ITL 计数依据；fallback 到 choice.text 的逻辑不变）
    delta_content_keys: tuple[str, ...] = ("content", "reasoning_content", "reasoning")
    # `probe env` 可选指标存在性表的清单（vllm = 历史硬编码 7 项；
    # vllm:cache_config_info 走 env_check 的独立展示块，不在此列）
    env_check_metrics: tuple[str, ...] = ()
    # 快照 JSON engine_extra 段的采集前缀（--engine-extras 开启时生效；vllm 为空=无）
    extra_metric_families: tuple[str, ...] = ()
    # 报告 diagnose 阈值覆盖（在 vendor 覆盖之后合并；模式同 probe/vendors.py）
    diagnose_overrides: dict[str, float] = field(default_factory=dict)


ENGINE_REGISTRY: dict[str, EngineProfile] = {
    "vllm": EngineProfile(
        id="vllm",
        label="vLLM（默认引擎）",
        caps=frozenset({
            "api.openai-chat", "api.openai-completions", "api.models-list", "api.health",
            "api.props", "metrics.vllm-named", "metrics.prometheus",
            "obs.step-log", "obs.per-request-metrics", "obs.spec-metrics",
            "obs.trace-pytorch", "obs.graph-mode", "spec.decode",
            "runtime.docker", "runtime.host",
        }),
        process_pattern="",           # 空 → 回落 HOST_VLLM_PATTERN（common.sh 默认值）
        version_cmd="",               # 空 → 走 import vllm 探测（hostcheck 现行为）
        check_severity="fail",
        report_notes=(),
        # ---- F3 · Python 侧专有（= 历史硬编码行为，保证 vLLM 链路逐字节一致）----
        metric_prefix="",             # 空 → 按 flavor 派生 "vllm:"
        chat_body_extras={"chat_template_kwargs": {"reasoning_effort": "low"}},
        delta_content_keys=("content", "reasoning_content", "reasoning"),
        env_check_metrics=(
            "vllm:kv_block_lifetime_seconds",
            "vllm:estimated_flops_per_gpu_total",
            "vllm:spec_decode_num_drafts_total",
            "vllm:num_requests_waiting_by_reason",
            "vllm:inter_token_latency_seconds",
            "vllm:prefix_cache_hits_total",
            "vllm:prompt_tokens_cached_total",
        ),
        extra_metric_families=(),
        diagnose_overrides={},
    ),
    # Strata（F2 落地）：字段与 engines/strata.sh 逐字段一致（tests/test_engines.py 强制）。
    # 上游事实见 docs/design/strata-engine/事实核对.md；⭐ 真机核对项见其施工图 §4。
    "strata": EngineProfile(
        id="strata",
        label="Strata（MoE 专家分层推理引擎）",
        caps=frozenset({
            "api.openai-chat", "api.models-list", "api.health", "api.props",
            "spec.decode", "metrics.vllm-named", "metrics.prometheus", "runtime.host",
        }),
        entrypoint="strata launch --yes --no-tui",
        process_pattern=r"strata|server\.py",
        metrics_flavor="vllm-named",
        args_mode="custom",
        ready_metric=r"^vllm:num_requests_(running|waiting)",
        metrics_fetch_suffix="?format=prometheus",
        version_cmd="strata --version",
        check_severity="warn",
        report_notes=(
            "Strata 为消费级 MoE 专家分层引擎：逐 step / stat-logger / torch profiler / "
            "图模式观测不适用；思考计入 max_tokens（建议 PROBE_MAX_TOKENS≥256）；"
            "默认单请求串行，并发吞吐解释需谨慎。",
        ),
        notes="详见 docs/design/strata-engine/事实核对.md",
        # ---- F3 · Python 侧专有（上游语义，见 事实核对.md §5/§6/§7）----
        metric_prefix="",             # flavor=vllm-named → 派生 "vllm:"（Strata 恰好说 vLLM 名）
        chat_body_extras={"reasoning_effort": "low"},  # 顶层键（上游走 OpenAI reasoning_effort）
        delta_content_keys=("content", "reasoning_content"),
        env_check_metrics=(
            "vllm:time_to_first_token_seconds",
            "vllm:inter_token_latency_seconds",
            "vllm:e2e_request_latency_seconds",
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
            "vllm:prefix_cache_hits_total",
        ),
        extra_metric_families=(
            "strata:live_", "strata:last_", "strata:totals_",
            "strata:gpu_", "strata:ram_", "strata:cpu", "strata:engine_",
        ),
        diagnose_overrides={},
    ),
    # SGLang（2026-10 接入）：字段与 engines/sglang.sh 逐字段一致（tests/test_engines.py 强制）。
    # 上游事实：docs.sglang.io / github.com/sgl-project/sglang；⭐ = 真机核对项
    # （指标名以真机 GET /metrics 实测为准，缺失项探针按缺项容错）。
    "sglang": EngineProfile(
        id="sglang",
        label="SGLang（高性能 OpenAI 兼容推理引擎）",
        caps=frozenset({
            "api.openai-chat", "api.openai-completions", "api.models-list", "api.health",
            "spec.decode", "metrics.prometheus", "runtime.host",
        }),
        entrypoint="python3 -m sglang.launch_server",
        process_pattern=r"sglang",
        metrics_flavor="prometheus",
        args_mode="custom",
        ready_metric=r"^sglang:num_running_reqs",
        metrics_fetch_suffix="",
        version_cmd="sglang version",  # 新版 CLI 子命令（无 sglang --version）；旧版无 CLI 按 warn
        check_severity="warn",
        report_notes=(
            "SGLang 为 OpenAI 兼容推理引擎：逐 step / stat-logger / torch profiler / "
            "图模式观测未接入本工具；/metrics 需启动 flag --enable-metrics（默认关，"
            "缺失时指标小节为空属预期）；指标为 sglang: 前缀 Prometheus 文本；"
            "思考内容需 --reasoning-parser 才拆到 reasoning_content（否则混在 content）；"
            "温度 0 下动态批+radix cache 非确定，严格 A/B 需 "
            "--enable-deterministic-inference。",
        ),
        notes="RadixAttention 前缀缓存 + PD 分离/EP；/metrics 需 --enable-metrics 启动（默认关）；无逐 step 日志/torch profiler 接入面；投机解码（EAGLE/NEXTN/NGRAM）flag 面可用；指标为 sglang: 前缀 Prometheus 文本；显存旋钮是 --mem-fraction-static。",
        # ---- Python 侧专有（sglang: 前缀过滤；请求体零注入=标准 OpenAI 形态）----
        metric_prefix="sglang:",      # flavor=prometheus 的派生值是不过滤，这里显式收紧
        chat_body_extras={},          # reasoning_effort 别同时设服务端默认与请求级，零注入最稳
        delta_content_keys=("content", "reasoning_content"),
        env_check_metrics=(
            "sglang:num_running_reqs",
            "sglang:num_queue_reqs",
            "sglang:time_to_first_token_seconds",
            "sglang:gen_throughput",
            "sglang:cache_hit_rate",
        ),
        extra_metric_families=(),
        diagnose_overrides={},
    ),
    # llama.cpp（2026-10 接入）：字段与 engines/llamacpp.sh 逐字段一致（tests/test_engines.py 强制）。
    # 上游事实：github.com/ggml-org/llama.cpp tools/server/README.md；⭐ = 真机核对项。
    "llamacpp": EngineProfile(
        id="llamacpp",
        label="llama.cpp（llama-server GGUF 推理）",
        caps=frozenset({
            "api.openai-chat", "api.openai-completions", "api.models-list", "api.health",
            "api.props", "spec.decode", "metrics.prometheus", "runtime.host",
        }),
        entrypoint="llama-server",
        process_pattern=r"llama-server",
        metrics_flavor="prometheus",
        args_mode="custom",
        ready_metric=r"^llamacpp:(n_decode_total|requests_processing|prompt_tokens_total)",
        metrics_fetch_suffix="",
        version_cmd="llama-server --version",
        check_severity="warn",
        report_notes=(
            "llama.cpp 为单节点 GGUF 推理：逐 step / stat-logger / torch profiler / "
            "图模式观测不适用；/metrics 需启动 flag --metrics（默认关，缺失时指标小节为空"
            "属预期）；指标为 llamacpp: 前缀 Counter/Gauge（无 TTFT/延迟直方图）；"
            "无跨卡张量并行（--tensor-split 仅显存切分），-c 为总上下文、按 --parallel "
            "槽位平分；思考模型经 --reasoning-format（默认 auto）输出 reasoning_content"
            "（TTFT 含思考首 token）。",
        ),
        notes="单节点 GGUF 推理：/metrics 需 --metrics 启动（默认关）；-ngl GPU 层数、--parallel 槽位、-c 总上下文按 slots 平分；无跨卡张量并行（--tensor-split 仅显存切分）；--jinja 新版默认开（旧版要显式）；思考模型经 --reasoning-format（默认 auto）输出 reasoning_content。",
        # ---- Python 侧专有（llamacpp: 前缀过滤；请求体零注入=标准 OpenAI 形态）----
        metric_prefix="llamacpp:",
        chat_body_extras={},
        delta_content_keys=("content", "reasoning_content"),
        env_check_metrics=(
            "llamacpp:prompt_tokens_total",
            "llamacpp:tokens_predicted_total",
            "llamacpp:n_decode_total",
        ),
        extra_metric_families=(),
        diagnose_overrides={},
    ),
    # Ollama（2026-10 接入）：字段与 engines/ollama.sh 逐字段一致（tests/test_engines.py 强制）。
    # 上游事实：github.com/ollama/ollama / docs.ollama.com；主线无 /metrics
    # （指标 PR #16998/#18508 尚未合并，合并后按 prometheus flavor 回填）。
    "ollama": EngineProfile(
        id="ollama",
        label="Ollama（本地模型管理/推理服务）",
        caps=frozenset({
            "api.openai-chat", "api.openai-completions", "api.models-list", "api.health",
            "runtime.host",
        }),
        entrypoint="ollama serve",
        process_pattern=r"ollama",
        metrics_flavor="none",        # 主线无 /metrics：探针跳过指标面，就绪只看 GET /
        args_mode="custom",
        health_path="/",              # 无 /health；根路径 200 "Ollama is running" 承担就绪判定
        ready_metric="",              # 无指标面：就绪判定只走 health_path 轮询
        version_cmd="ollama --version",
        check_severity="warn",
        report_notes=(
            "Ollama 为本地模型管理/推理服务：无 /metrics 指标面，本报告无服务端指标小节"
            "（时延/吞吐均为客户端口径）；逐 step / 投机解码 / 图模式观测不适用；"
            "并发默认 1（OLLAMA_NUM_PARALLEL，压测前显式设置并注意 KV 显存随 "
            "num_ctx×parallel 膨胀）；keep_alive 默认 5m 到期卸载模型（长压测 "
            "OLLAMA_KEEP_ALIVE=-1）；思考内容在 delta.reasoning 字段（探针已计入），"
            "/v1 层用 reasoning_effort:\"none\" 关闭思考。",
        ),
        notes="模型以 tag 名寻址（需先 ollama pull）；无 /metrics 指标面（探测只走 HTTP 时延/吞吐）；并发默认 1（OLLAMA_NUM_PARALLEL）；keep_alive 默认 5m 到期卸载；思考内容在 delta.reasoning 字段。",
        # ---- Python 侧专有（无指标面：过滤与 env 指标表均为空）----
        metric_prefix="",
        chat_body_extras={},          # /v1 层不接受 think 字段（400 风险），零注入
        delta_content_keys=("content", "reasoning"),  # 上游思考流在 delta.reasoning（非 reasoning_content）
        env_check_metrics=(),         # flavor=none 无指标存在性表可查（测试按 flavor 放行）
        extra_metric_families=(),
        diagnose_overrides={},
    ),
}


def list_engines() -> list[str]:
    """返回注册表里的全部引擎 id（排序）。"""
    return sorted(ENGINE_REGISTRY)


def get_engine(engine_id: str) -> EngineProfile:
    try:
        return ENGINE_REGISTRY[engine_id]
    except KeyError:
        raise KeyError(
            f"未知引擎 {engine_id!r}（可用：{', '.join(list_engines())}）"
        ) from None


def resolve_engine(requested: str | None = None) -> EngineProfile:
    """解析引擎：显式参数 > 环境变量 ENGINE（空/auto 忽略）> vllm。

    ``auto`` 别名语义同 ``VENDOR=auto``：忽略显式空值，直接用环境变量 / 默认值。
    """
    if requested and requested != "auto":
        return get_engine(requested)
    env_value = (os.environ.get("ENGINE") or "").strip()
    if env_value and env_value != "auto":
        return get_engine(env_value)
    return get_engine("vllm")


def has_cap(profile: EngineProfile, cap: str) -> bool:
    """引擎是否声明了某能力键（词表校验交给 tests/test_engines.py）。"""
    return cap in profile.caps


def engine_info(profile: EngineProfile | None = None, requested: str | None = None) -> dict[str, Any]:
    """汇总引擎的静态契约 + 本机运行时事实（入口命令是否存在等）。"""
    profile = profile or resolve_engine(requested)
    caps = sorted(profile.caps)
    unknown_caps = sorted(set(caps) - set(CAP_VOCAB))
    return {
        "id": profile.id,
        "label": profile.label,
        "requested": requested if requested is not None else os.environ.get("ENGINE", "vllm"),
        "available": list_engines(),
        "caps": caps,
        "unknown_caps": unknown_caps,  # 非空 = 词表外键（应被测试拦截，运行时兜底展示）
        "metrics_flavor": profile.metrics_flavor,
        "args_mode": profile.args_mode,
        "entrypoint": profile.entrypoint,
        "entrypoint_available": bool(profile.entrypoint) and shutil.which(
            profile.entrypoint.split()[0]
        ) is not None,
        "process_pattern": profile.process_pattern,
        "health_path": profile.health_path,
        "models_path": profile.models_path,
        "ready_metric": profile.ready_metric,
        "metrics_fetch_suffix": profile.metrics_fetch_suffix,
        "version_cmd": profile.version_cmd,
        "check_severity": profile.check_severity,
        "docker_image": profile.docker_image,
        "report_notes": list(profile.report_notes),
        "notes": profile.notes,
    }


def format_engine_info(info: dict[str, Any]) -> str:
    """人类可读的 ``probe engine`` 输出（风格对齐 ``format_vendor_info``）。"""
    lines = [
        f"引擎: {info['id']} — {info['label']}",
        f"请求值 ENGINE={info['requested']}  可用引擎: {', '.join(info['available'])}",
        f"metrics 形态: {info['metrics_flavor']}   参数拼装: {info['args_mode']}",
    ]
    if info["entrypoint"]:
        marker = "" if info["entrypoint_available"] else "  ⚠️ 当前 PATH 中未找到"
        lines.append(f"默认入口: {info['entrypoint']}{marker}")
    else:
        lines.append("默认入口: （无；vLLM 走 host_serve_entrypoint 的内置默认）")
    if info["process_pattern"]:
        lines.append(f"进程正则: {info['process_pattern']}")
    else:
        lines.append("进程正则: （未声明；回落 HOST_VLLM_PATTERN）")
    if info["ready_metric"]:
        lines.append(f"就绪指标正则: {info['ready_metric']}")
    if info["metrics_fetch_suffix"]:
        lines.append(f"metrics 抓取后缀: {info['metrics_fetch_suffix']}")
    if info["version_cmd"]:
        lines.append(f"版本探测: {info['version_cmd']}（缺失按 {info['check_severity']} 处置）")
    if info["docker_image"]:
        lines.append(f"docker 镜像: {info['docker_image']}")
    lines.append(f"能力位（{len(info['caps'])}）:")
    for cap in info["caps"]:
        flag = "" if cap in CAP_VOCAB else "  ⚠️ 词表外"
        lines.append(f"  - {cap}{flag}")
    for note in info["report_notes"]:
        lines.append(f"注记: {note}")
    if info["notes"]:
        lines.append(f"备注: {info['notes']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# F3 · 探针层的派生 helper（profile 字段的统一读取口径）
# ---------------------------------------------------------------------------


def effective_metric_prefix(profile: EngineProfile) -> str:
    """指标名前缀过滤值：显式 ``metric_prefix`` 优先，否则按 metrics_flavor 派生。

    * ``vllm-named`` → ``"vllm:"``（Strata 恰好以 vLLM 指标名输出，结果不变）；
    * ``prometheus`` → ``""``（不过滤，保留全部名字）；
    * ``none``       → ``""``（引擎无 /metrics 面，调用方不应走到过滤）。
    """
    if profile.metric_prefix:
        return profile.metric_prefix
    if profile.metrics_flavor == "vllm-named":
        return "vllm:"
    return ""


def profile_for_config(cfg: Any) -> EngineProfile:
    """从 ProbeConfig（``.engine`` 字段）解析 profile；解析失败回落 vllm（容错不崩）。"""
    engine_id = str(getattr(cfg, "engine", "") or "vllm")
    try:
        return get_engine(engine_id)
    except KeyError:
        return get_engine("vllm")
