"""统一命令行入口：``python3 -m probe <子命令>``。

子命令一览::

    env      环境自检：服务是否可达、哪些观测开关已经生效
    metrics  /metrics 快照、结构化解析、两次快照做差
    http     请求级压测与逐请求生命周期打点
    logs     引擎日志 → 逐 step 结构化事件（支持 --live / --follow / stdin）
    trace    torch profiler trace 分析（算子/kernel 级）
    plugin   汇总 stat logger 插件的 probe_steps.jsonl
    vendor   厂商适配信息：检测结果、SMI 可用性、trace 规则包
    telemetry GPU 遥测采样（各厂商 SMI → 统一 JSONL）
    suite    标准化对比测试套件（固定场景矩阵 → 可跨厂商对比的 JSON）
    compare  并排对比两份 suite 结果
    report   把一次运行目录里的所有产物合成中文 Markdown 报告
    hostcheck 物理机（host）模式环境预检：起 vLLM 前先看缺什么

所有子命令都可用 ``--outdir`` 指定产物目录（默认取环境变量 ``PROBE_OUTDIR``，
再默认 ``runs/latest``）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .config import ProbeConfig


# --------------------------------------------------------------------------
# 公共参数
# --------------------------------------------------------------------------


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", dest="base_url", default=None,
                        help="服务地址，默认 http://127.0.0.1:8100")
    parser.add_argument("--port", dest="port", type=int, default=None,
                        help="服务端口（等价于 --base-url http://127.0.0.1:<port>）")
    parser.add_argument("--model", dest="model", default=None, help="served-model-name")
    parser.add_argument("--api-key", dest="api_key", default=None, help="API key（占位即可）")
    parser.add_argument("--outdir", dest="outdir", default=None, help="产物目录")
    parser.add_argument("--timeout", dest="timeout", type=float, default=None, help="超时秒数")


def cfg_from(args: argparse.Namespace) -> ProbeConfig:
    cfg = ProbeConfig.from_env()
    cfg.apply_namespace(args)
    # 地址优先级：--base-url > --port > 环境变量/默认值
    if getattr(args, "base_url", None):
        cfg.base_url = args.base_url
    elif getattr(args, "port", None):
        cfg.base_url = f"http://127.0.0.1:{cfg.port}"
    return cfg


# --------------------------------------------------------------------------
# 子命令：env
# --------------------------------------------------------------------------


def cmd_env(args: argparse.Namespace) -> int:
    import urllib.error
    import urllib.request

    from .http_probe import request_once
    from .metrics_probe import fetch_metrics

    cfg = cfg_from(args)
    cfg.ensure_outdir()
    checks: list[dict[str, object]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})
        print(f"[{'✅' if ok else '❌'}] {name}: {detail}")

    # 1) 基本端点
    for path, label in (("/health", "健康检查"), ("/v1/models", "模型列表")):
        url = cfg.base_url.rstrip("/") + path
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                body = resp.read(2000).decode("utf-8", errors="replace")
                check(label, resp.status == 200, f"HTTP {resp.status} {body[:120].replace(chr(10), ' ')}")
        except urllib.error.HTTPError as exc:
            check(label, False, f"HTTP {exc.code}")
        except Exception as exc:  # noqa: BLE001
            check(label, False, f"{type(exc).__name__}: {exc}")

    # 2) /metrics
    snap = None
    try:
        snap = fetch_metrics(cfg)
        check("指标端点", True, f"{len(snap.families)} 个指标族")
    except Exception as exc:  # noqa: BLE001
        check("指标端点", False, f"{type(exc).__name__}: {exc}")

    if snap is not None:
        optional = [
            ("vllm:kv_block_lifetime_seconds", "--kv-cache-metrics"),
            ("vllm:estimated_flops_per_gpu_total", "--enable-mfu-metrics"),
            ("vllm:spec_decode_num_drafts_total", "投机解码已配置"),
            ("vllm:num_requests_waiting_by_reason", "调度等待原因分解"),
            ("vllm:inter_token_latency_seconds", "ITL 直方图"),
            ("vllm:prefix_cache_hits_total", "前缀缓存指标"),
            ("vllm:prompt_tokens_cached_total", "缓存 prompt token 计数"),
        ]
        for name, hint in optional:
            present = snap.family(name) is not None
            check(f"指标 {name}", present, hint if present else f"缺失（需要 {hint}）")
        info = snap.family("vllm:cache_config_info")
        if info and info.series:
            labels = info.series[0].labels
            check(
                "缓存配置",
                True,
                "prefix_caching=" + str(labels.get("enable_prefix_caching"))
                + " kv_cache_size_tokens=" + str(labels.get("kv_cache_size_tokens"))
                + " block_size=" + str(labels.get("block_size")),
            )

    # 3) 逐请求指标（发一个极短请求）
    if not args.no_request:
        try:
            result = request_once(cfg, index=0, prompt="ping", max_tokens=args.probe_tokens)
            ok = result.ok
            check("推理请求", ok, f"ttft={result.ttft_ms} e2e={result.e2e_ms} err={result.error}")
            check(
                "逐请求指标 metrics 字段",
                bool(result.server_metrics),
                "已启用 --enable-per-request-metrics" if result.server_metrics
                else "响应里没有 metrics（需要 --enable-per-request-metrics）",
            )
            if result.server_metrics:
                check("服务端计时字段", True, json.dumps(result.server_metrics, ensure_ascii=False)[:200])
            check(
                "prompt_tokens_details（缓存命中）",
                result.cached_tokens is not None,
                f"cached_tokens={result.cached_tokens}"
                if result.cached_tokens is not None
                else "需要 --enable-prompt-tokens-details",
            )
            check(
                "逐请求投机解码指标",
                bool(result.spec_decode),
                "已启用 --per-request-spec-decode-metrics"
                if result.spec_decode else "响应里没有 speculative_decoding",
            )
        except Exception as exc:  # noqa: BLE001
            check("推理请求", False, f"{type(exc).__name__}: {exc}")

    # 4) profiler 端点（可选，会真的开关一次 profiler）
    if args.check_profile:
        for endpoint, label in (("/start_profile", "profiler 启动端点"), ("/stop_profile", "profiler 停止端点")):
            url = cfg.base_url.rstrip("/") + endpoint
            try:
                request = urllib.request.Request(url, data=b"", method="POST")
                with urllib.request.urlopen(request, timeout=15) as resp:
                    check(label, resp.status == 200, f"HTTP {resp.status}")
            except urllib.error.HTTPError as exc:
                check(label, False, f"HTTP {exc.code}（未配置 --profiler-config 时不存在）")
            except Exception as exc:  # noqa: BLE001
                check(label, False, f"{type(exc).__name__}: {exc}")

    payload = {"checks": checks, "base_url": cfg.base_url, "version": __version__}
    try:
        from .vendors import vendor_info

        payload["vendor"] = vendor_info()
    except KeyError:
        payload["vendor"] = {"id": "unknown"}  # VENDOR 环境变量非法时不阻塞自检
    out = cfg.out_path("env_check.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")
    failed = [c for c in checks if not c["ok"]]
    print(f"通过 {len(checks) - len(failed)}/{len(checks)} 项")
    return 0 if not failed else 1


# --------------------------------------------------------------------------
# 子命令：metrics
# --------------------------------------------------------------------------


def cmd_metrics(args: argparse.Namespace) -> int:
    from .metrics_probe import (
        MetricsSnapshot,
        diff_snapshots,
        fetch_metrics,
        format_stage_table,
        save_snapshot,
        stage_table,
    )

    cfg = cfg_from(args)
    cfg.ensure_outdir()

    if args.diff:
        before = MetricsSnapshot.load(args.diff[0])
        after = MetricsSnapshot.load(args.diff[1])
        diff = diff_snapshots(before, after, args.elapsed or 0.0, model=cfg.model)
        out = cfg.out_path("metrics_diff.json")
        out.write_text(json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {out}")
        if args.print:
            for name, rate in sorted((diff.get("rates") or {}).items()):
                if rate:
                    print(f"  {name}: {rate:.3f}/s  (Δ={diff['counters'].get(name):.0f})")
            for name, row in (diff.get("histograms") or {}).items():
                print(f"  {name}: n={row['count']:.0f} mean={row['mean_ms']:.1f}ms")
        return 0

    path, snap = save_snapshot(cfg, tag=args.tag)
    print(f"已写入 {path}")
    print(format_stage_table(stage_table(snap)))
    if args.print:
        for fam in snap.families.values():
            if fam.name.startswith("vllm:"):
                for series in fam.series:
                    labels = ",".join(f"{k}={v}" for k, v in sorted(series.labels.items()))
                    print(f"  {series.name}{{{labels}}} = {series.value}")
    return 0


# --------------------------------------------------------------------------
# 子命令：http
# --------------------------------------------------------------------------


def cmd_http(args: argparse.Namespace) -> int:
    from .http_probe import format_summary, results_to_jsonl, run_matrix

    cfg = cfg_from(args)
    cfg.apply_namespace(args)
    cfg.ensure_outdir()
    print(f"探测目标: {cfg.request_url()}\n{cfg.describe()}\n")
    results, summary = run_matrix(cfg, warmup=args.warmup)
    results_to_jsonl(results, cfg.out_path("http_requests.jsonl"))
    out = cfg.out_path("http_summary.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + format_summary(summary))
    print(f"\n已写入 {out}")
    return 0 if summary.get("ok") else 1


# --------------------------------------------------------------------------
# 子命令：logs
# --------------------------------------------------------------------------


def cmd_logs(args: argparse.Namespace) -> int:
    from .log_probe import (
        follow_file,
        format_live,
        iter_events,
        read_events,
        summarize,
        summarize_text,
    )

    cfg = cfg_from(args)

    if args.stdin:
        source = sys.stdin
        live = True
    elif args.file:
        source = None
        live = args.live
    else:
        print("需要 --file 或 --stdin", file=sys.stderr)
        return 2

    events = []
    if args.stdin or args.follow:
        stream = sys.stdin if args.stdin else follow_file(args.file, poll_s=args.poll)
        for line in stream:
            for event in iter_events([line]):
                events.append(event)
                if live:
                    text = format_live(event)
                    if text:
                        print(text, flush=True)
            if len(events) >= args.max_events:
                break
    else:
        events = read_events(args.file)
        if args.tail and len(events) > args.tail:
            events = events[-args.tail :]
        if live:
            for event in events[-args.live_tail :]:
                text = format_live(event)
                if text:
                    print(text)

    if args.jsonl:
        out = Path(args.jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for event in events:
                fh.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        print(f"已写入 {out}（{len(events)} 条事件）", file=sys.stderr)

    if args.summary:
        summary = summarize(events)
        out = Path(args.summary)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {out}", file=sys.stderr)
        print(summarize_text(summary))
    elif not live and not args.jsonl:
        print(summarize_text(summarize(events)))
    return 0


# --------------------------------------------------------------------------
# 子命令：trace
# --------------------------------------------------------------------------


def cmd_trace(args: argparse.Namespace) -> int:
    from .trace_probe import analyze_trace, format_summary, parse_key_averages, summarize_key_averages, to_dict

    cfg = cfg_from(args)
    cfg.ensure_outdir()

    paths: list[Path] = []
    if args.path:
        paths.append(Path(args.path))
    if args.glob:
        paths.extend(sorted(Path.cwd().glob(args.glob)))
        paths.extend(sorted(Path(cfg.outdir).glob(args.glob)))
    if args.dir:
        paths.extend(sorted(Path(args.dir).glob("*.pt.trace.json*")))
    # 去重且保持顺序
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen and path.exists():
            seen.add(key)
            unique.append(path)
    if not unique:
        print("未找到 trace 文件（--path/--glob/--dir）", file=sys.stderr)
        return 2

    primary = unique[0]
    print(f"分析 {primary}（共发现 {len(unique)} 个 trace 文件）")
    summary = analyze_trace(primary, limit=args.limit, vendor=getattr(args, "vendor", None))
    if summary.vendor:
        print(f"厂商规则包: {summary.vendor}")
    payload = to_dict(summary, top=args.top)
    payload["files"] = [str(p) for p in unique]
    if args.key_averages:
        rows = parse_key_averages(args.key_averages)
        ka = summarize_key_averages(rows, top=args.top)
        (cfg.out_path("key_averages.json")).write_text(
            json.dumps(ka, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        payload["key_averages_file"] = args.key_averages
        print(f"已写入 {cfg.out_path('key_averages.json')}")
    out = cfg.out_path("trace_summary.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(format_summary(summary, top=args.top))
    print(f"\n已写入 {out}")
    return 0


# --------------------------------------------------------------------------
# 子命令：plugin
# --------------------------------------------------------------------------


def cmd_plugin(args: argparse.Namespace) -> int:
    """汇总 stat logger 插件产出的 probe_steps.jsonl。"""
    from .plugin_records import format_summary, load_records, summarize

    cfg = cfg_from(args)
    cfg.ensure_outdir()
    path = Path(args.file) if args.file else Path(cfg.outdir) / "probe_steps.jsonl"
    records, header = load_records(path)
    if not records:
        print(
            f"没有读到插件记录：{path}\n"
            "  提示：需要 (1) 用 PROBE_PLUGIN=1 启动（插件会把 JSONL 写到 /logs/probe_steps.jsonl），\n"
            "        且 (2) --enable-logging-iteration-details 才会带上逐 step 明细。",
            file=sys.stderr,
        )
        return 2
    summary = summarize(records, header)
    out = Path(args.summary) if args.summary else Path(cfg.outdir) / "probe_steps_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(format_summary(summary))
    print(f"\n已写入 {out}")
    return 0


# --------------------------------------------------------------------------
# 子命令：vendor
# --------------------------------------------------------------------------


def cmd_vendor(args: argparse.Namespace) -> int:
    """展示厂商适配信息（检测结果、SMI 可用性、trace 规则包、镜像提示）。"""
    from .vendors import format_vendor_info, vendor_info

    if getattr(args, "vendor", None):
        info = vendor_info(requested=args.vendor)
    else:
        info = vendor_info()
    print(format_vendor_info(info))
    if getattr(args, "json", False):
        print(json.dumps(info, ensure_ascii=False, indent=2))
    if getattr(args, "write", False):
        cfg = cfg_from(args)
        cfg.ensure_outdir()
        out = cfg.out_path("vendor_info.json")
        out.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {out}")
    return 0


# --------------------------------------------------------------------------
# 子命令：telemetry（GPU 遥测采样）
# --------------------------------------------------------------------------


def cmd_telemetry(args: argparse.Namespace) -> int:
    from .telemetry import (
        backend_by_name,
        format_record,
        load_records,
        sample_loop,
        sample_once,
        summarize_records,
        vram_perc_sum,
    )
    from .vendors import resolve_vendor

    cfg = cfg_from(args)
    profile = resolve_vendor(args.vendor)

    if args.parse:
        backend = backend_by_name(args.backend or profile.smi_cmd or "")
        text = Path(args.parse).read_text(encoding="utf-8", errors="replace")
        cards = backend.parse(text)
        print(json.dumps(cards, ensure_ascii=False, indent=2))
        return 0 if cards else 1

    if args.vram_sum:
        value = vram_perc_sum(profile)
        print("none" if value is None else value)
        return 0

    if args.once:
        record = sample_once(profile)
        if args.file:
            out = Path(args.file)
        else:
            out = cfg.out_path("gpu_telem.jsonl") if args.write else None
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"已追加 {out}", file=sys.stderr)
        print(json.dumps(record, ensure_ascii=False, indent=2) if args.json else format_record(record))
        return 0 if record.get("ok") else 1

    # --duration 模式（gpu_sample.sh 走这里）
    interval = args.interval if args.interval is not None else float(os.environ.get("GPU_SAMPLE_INTERVAL", "1") or 1)
    out = Path(args.file) if args.file else cfg.out_path("gpu_telem.jsonl")
    raw = None
    if not args.no_raw:
        raw = Path(args.raw_file) if args.raw_file else out.with_name("gpu_telem_raw.txt")
    cfg.ensure_outdir()
    count = sample_loop(
        duration=args.duration, interval=interval, out_jsonl=out, raw_file=raw,
        profile=profile, print_each=args.print,
    )
    summary = summarize_records(load_records(out))
    print(f"采样 {count} 次 → {out}")
    if summary.get("ok_samples"):
        vram = summary.get("vram_perc_sum")
        power = summary.get("power_w_sum")
        temp = summary.get("temp_c_max")
        print(f"厂商={summary['vendor']} SMI={summary['smi']} 卡={','.join(summary.get('cards') or [])}")
        if vram:
            print(f"显存占用合计 %: 均值={vram['mean']:.1f} max={vram['max']:.1f}")
        if power:
            print(f"功耗合计 W:    均值={power['mean']:.1f} max={power['max']:.1f}")
        if temp:
            print(f"最高温度 C:    均值={temp['mean']:.1f} max={temp['max']:.1f}")
    else:
        print(f"（无有效遥测数据：{summary.get('samples', 0)} 次采样都没有解析出卡片信息；")
        print("  确认厂商 SMI 命令在本机可用，或设置 VENDOR=... 显式指定）")
    return 0


# --------------------------------------------------------------------------
# 子命令：suite / compare（标准化对比测试）
# --------------------------------------------------------------------------


def cmd_suite(args: argparse.Namespace) -> int:
    from .suite import (
        default_scenarios,
        run_suite,
        select_scenarios,
    )

    cfg = cfg_from(args)
    cfg.ensure_outdir()
    matrix = default_scenarios(getattr(args, "long_context_tokens", None))

    if args.list:
        print(f"共 {len(matrix)} 个场景（--scenarios 可用子串过滤，如 s02,s05）：\n")
        for scenario in matrix:
            tags = []
            if scenario.warmup:
                tags.append("热身")
            if scenario.optional:
                tags.append("可选")
            tag = f"  [{'+'.join(tags)}]" if tags else ""
            print(
                f"  {scenario.id:<20} {scenario.label:<28} "
                f"p={scenario.prompt_tokens} g={scenario.max_tokens} "
                f"c={scenario.concurrency} r={scenario.repeat}"
                f"{' prefix=' + str(scenario.shared_prefix_tokens) if scenario.shared_prefix_tokens else ''}{tag}"
            )
        return 0

    try:
        selected = select_scenarios(args.scenarios, matrix)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"标准化套件：{len(selected)} 个场景 → {cfg.outdir}")
    payload = run_suite(
        cfg, selected, verbose=args.verbose,
        progress=lambda line: print(line, flush=True),
    )

    print("\n场景汇总：")
    print(f"  {'场景':<22}{'OK':<6}{'TTFT p50':>10}{'TPOT p50':>10}{'输出 tok/s':>12}{'引擎 gen tok/s':>14}")
    for scenario in selected:
        result = payload["scenarios"].get(scenario.id, {})
        def g(*path):
            node = result
            for key in path:
                if not isinstance(node, dict):
                    return None
                node = node.get(key)
            return node
        ttft = g("ttft_ms", "p50")
        tpot = g("tpot_ms", "p50")
        out_tps = g("throughput", "output_tokens_per_s")
        eng_tps = g("engine", "generation_tokens_per_s")
        flag = "ok" if result.get("ok") else ("skip" if result.get("optional") else "ERR")
        fmt = lambda v, d=0: "—" if v is None else f"{v:.{d}f}"  # noqa: E731
        print(
            f"  {scenario.id:<22}{flag:<6}{fmt(ttft):>10}{fmt(tpot, 1):>10}"
            f"{fmt(out_tps, 1):>12}{fmt(eng_tps, 1):>14}"
        )
    print(f"\n结果：{Path(cfg.outdir) / 'suite_results.json'}")
    if not payload["ok"]:
        print("⚠️ 存在未成功场景（详见结果 JSON 的 errors 字段）")
    return 0 if payload["ok"] else 1


def cmd_compare(args: argparse.Namespace) -> int:
    from .suite import compare_payloads, load_suite

    try:
        a = load_suite(args.a)
        b = load_suite(args.b)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"读取失败：{exc}", file=sys.stderr)
        return 2
    text = compare_payloads(a, b)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"已写入 {out}")
    else:
        print(text)
    return 0


# --------------------------------------------------------------------------
# 子命令：report
# --------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    from .report import build_report

    cfg = cfg_from(args)
    outdir = Path(cfg.outdir)
    if not outdir.exists():
        print(f"运行目录不存在: {outdir}", file=sys.stderr)
        return 2
    text = build_report(outdir, title=args.title)
    out = Path(args.out) if args.out else outdir / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"已生成报告 {out}（{len(text)} 字符）")
    return 0


# --------------------------------------------------------------------------
# 子命令：hostcheck
# --------------------------------------------------------------------------


def cmd_hostcheck(args: argparse.Namespace) -> int:
    from .hostcheck import run_hostcheck

    return run_hostcheck(
        host_python=args.host_python,
        model=args.model,
        port=args.port,
        outdir=args.outdir,
        prelude=args.prelude,
        quick=args.quick,
        start=args.start,
        skip_vllm_import=args.skip_vllm_import,
        skip_gpu=args.skip_gpu,
        write_json_file=args.json,
    )


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="probe",
        description="vLLM 推理全流程探测工具（多厂商：NVIDIA/AMD/Hygon/Intel/Ascend，vLLM 0.18.x）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"probe {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_env = sub.add_parser("env", help="环境自检：可达性与观测开关")
    add_common(p_env)
    p_env.add_argument("--no-request", action="store_true", help="不发推理请求")
    p_env.add_argument("--probe-tokens", type=int, default=4, help="自检请求的输出 token 数")
    p_env.add_argument("--check-profile", action="store_true",
                       help="真的调用 /start_profile 和 /stop_profile 来验证 profiler 端点")
    p_env.set_defaults(func=cmd_env)

    p_metrics = sub.add_parser("metrics", help="指标快照/差分")
    add_common(p_metrics)
    p_metrics.add_argument("--tag", default="metrics", help="快照文件名前缀（默认 metrics）")
    p_metrics.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"), help="对两个快照 JSON 做差")
    p_metrics.add_argument("--elapsed", type=float, default=None, help="两次快照之间的秒数")
    p_metrics.add_argument("--print", action="store_true", help="打印全部 vllm:* 明细")
    p_metrics.set_defaults(func=cmd_metrics)

    p_http = sub.add_parser("http", help="请求级探测")
    add_common(p_http)
    p_http.add_argument("--prompt-tokens", type=int, default=None, help="输入长度（约）")
    p_http.add_argument("--max-tokens", type=int, default=None, help="输出长度")
    p_http.add_argument("--concurrency", type=int, default=None, help="并发数")
    p_http.add_argument("--repeat", type=int, default=None, help="每并发重复轮数")
    p_http.add_argument("--endpoint", choices=["chat", "completions"], default=None)
    p_http.add_argument("--no-stream", dest="stream", action="store_false", default=None, help="非流式")
    p_http.add_argument("--warmup", type=int, default=1, help="预热请求数（不计入统计）")
    p_http.add_argument("--shared-prefix-tokens", type=int, default=None, help="共享前缀长度，用于探测前缀缓存")
    p_http.add_argument("--temperature", type=float, default=None)
    p_http.set_defaults(func=cmd_http)

    p_logs = sub.add_parser("logs", help="引擎日志 → 逐 step 事件")
    add_common(p_logs)
    p_logs.add_argument("--file", default=None, help="日志文件（如 runs/latest/serve.log）")
    p_logs.add_argument("--stdin", action="store_true", help="从标准输入读（配合 docker logs -f）")
    p_logs.add_argument("--follow", action="store_true", help="跟随文件尾部（tail -f）")
    p_logs.add_argument("--live", action="store_true", help="实时打印每个 step")
    p_logs.add_argument("--live-tail", type=int, default=40, help="--live 且非 follow 时回放的行数")
    p_logs.add_argument("--jsonl", default=None, help="事件写入 JSONL")
    p_logs.add_argument("--summary", default=None, help="汇总写入 JSON")
    p_logs.add_argument("--tail", type=int, default=None, help="只解析文件末尾 N 条事件")
    p_logs.add_argument("--poll", type=float, default=0.5, help="--follow 轮询间隔")
    p_logs.add_argument("--max-events", type=int, default=10**9)
    p_logs.set_defaults(func=cmd_logs)

    p_trace = sub.add_parser("trace", help="torch profiler trace 分析")
    add_common(p_trace)
    p_trace.add_argument("--path", default=None, help="trace 文件（.json 或 .json.gz）")
    p_trace.add_argument("--glob", default=None, help="相对 run 目录的 glob，如 'traces/rank0*'")
    p_trace.add_argument("--dir", default=None, help="trace 目录")
    p_trace.add_argument("--key-averages", default=None, help="profiler_out_*.txt 路径")
    p_trace.add_argument("--vendor", default=None,
                         help="厂商规则包（默认：trace 元数据 → VENDOR 环境变量 → 跨厂商超集）")
    p_trace.add_argument("--top", type=int, default=25)
    p_trace.add_argument("--limit", type=int, default=None, help="只解析前 N 条事件（调试）")
    p_trace.set_defaults(func=cmd_trace)

    p_report = sub.add_parser("report", help="生成中文 Markdown 报告")
    add_common(p_report)
    p_report.add_argument("--out", default=None, help="输出文件（默认 <outdir>/REPORT.md）")
    p_report.add_argument("--title", default="vLLM 推理全流程探测报告")
    p_report.set_defaults(func=cmd_report)

    p_plugin = sub.add_parser("plugin", help="汇总 stat logger 插件的 probe_steps.jsonl")
    add_common(p_plugin)
    p_plugin.add_argument("--file", default=None, help="probe_steps.jsonl 路径")
    p_plugin.add_argument("--summary", default=None, help="汇总输出路径")
    p_plugin.set_defaults(func=cmd_plugin)

    p_vendor = sub.add_parser("vendor", help="厂商适配信息（检测/SMI/trace 规则包）")
    add_common(p_vendor)
    p_vendor.add_argument("--vendor", default=None,
                          help="显式指定厂商（默认按 VENDOR 环境变量/自动探测）")
    p_vendor.add_argument("--json", action="store_true", help="额外打印 JSON")
    p_vendor.add_argument("--write", action="store_true", help="写入 <outdir>/vendor_info.json")
    p_vendor.set_defaults(func=cmd_vendor)

    p_telem = sub.add_parser("telemetry", help="GPU 遥测采样（hy-smi/nvidia-smi/rocm-smi/npu-smi/xpu-smi）")
    add_common(p_telem)
    p_telem.add_argument("--vendor", default=None, help="显式指定厂商（默认自动探测）")
    p_telem.add_argument("--once", action="store_true", help="只采一次")
    p_telem.add_argument("--duration", type=float, default=60.0, help="采样总时长（秒），默认 60")
    p_telem.add_argument("--interval", type=float, default=None,
                         help="采样间隔（秒），默认取 GPU_SAMPLE_INTERVAL 或 1")
    p_telem.add_argument("--file", default=None, help="JSONL 输出路径（默认 <outdir>/gpu_telem.jsonl）")
    p_telem.add_argument("--raw-file", default=None, help="原始文本快照路径（默认 gpu_telem_raw.txt）")
    p_telem.add_argument("--no-raw", action="store_true", help="不落原始文本快照")
    p_telem.add_argument("--json", action="store_true", help="--once 时打印完整 JSON")
    p_telem.add_argument("--print", action="store_true", help="循环模式中实时打印")
    p_telem.add_argument("--write", action="store_true", help="--once 时也写入 outdir")
    p_telem.add_argument("--vram-sum", action="store_true",
                         help="只打印各卡 VRAM 占用百分比之和（无数据打印 none）")
    p_telem.add_argument("--parse", default=None,
                         help="离线解析一份原始 SMI 文本（配合 --backend，测试/调试用）")
    p_telem.add_argument("--backend", default=None, help="--parse 时指定 backend 名")
    p_telem.set_defaults(func=cmd_telemetry)

    p_suite = sub.add_parser("suite", help="标准化对比测试套件（固定 8 场景矩阵 → suite_results.json）")
    add_common(p_suite)
    p_suite.add_argument("--scenarios", default=None,
                         help="场景子串过滤（逗号分隔，如 s02,s05；默认全部）")
    p_suite.add_argument("--list", action="store_true", help="只列出场景矩阵")
    p_suite.add_argument("--verbose", action="store_true", help="打印逐请求进度")
    p_suite.add_argument("--long-context-tokens", type=int, default=None,
                         help="长上下文场景的 prompt token 数（默认 24576 或 SUITE_LONG_CONTEXT_TOKENS）")
    p_suite.set_defaults(func=cmd_suite)

    p_compare = sub.add_parser("compare", help="对比两份 suite_results.json（跨厂商/跨配置）")
    add_common(p_compare)
    p_compare.add_argument("--a", required=True, help="A 的 suite_results.json")
    p_compare.add_argument("--b", required=True, help="B 的 suite_results.json")
    p_compare.add_argument("--out", default=None, help="输出 Markdown 文件（默认打印）")
    p_compare.set_defaults(func=cmd_compare)

    p_hostcheck = sub.add_parser("hostcheck", help="物理机（host）模式环境预检：起 vLLM 前体检缺什么")
    p_hostcheck.add_argument("--json", action="store_true",
                             help="额外写 <outdir>/host_check.json")
    p_hostcheck.add_argument("--outdir", dest="outdir", default=None,
                             help="产物目录（默认 $PROBE_OUTDIR 或 runs/latest）")
    p_hostcheck.add_argument("--host-python", dest="host_python", default=None,
                             help="宿主 Python 解释器（默认 $HOST_PYTHON 或 python3）")
    p_hostcheck.add_argument("--model", default=None,
                             help="模型目录（默认 $MODEL）")
    p_hostcheck.add_argument("--port", type=int, default=None,
                             help="服务端口（默认 $PORT 或 8100）")
    p_hostcheck.add_argument("--quick", action="store_true",
                             help="跳过耗时项（import vllm/torch、GPU SMI 采样）")
    p_hostcheck.add_argument("--start", action="store_true",
                             help="启动前快速预检：只跑 fail 级核心项（隐含 --quick）")
    p_hostcheck.add_argument("--skip-vllm-import", dest="skip_vllm_import", action="store_true",
                             help="跳过 import vllm 检查")
    p_hostcheck.add_argument("--skip-gpu", dest="skip_gpu", action="store_true",
                             help="跳过 GPU SMI 采样")
    p_hostcheck.add_argument("--prelude", default=None,
                             help="厂商 prelude 脚本列表（空格分隔，默认 $HOST_PRELUDE_FILES）")
    p_hostcheck.set_defaults(func=cmd_hostcheck)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "outdir", None) is None:
        args.outdir = ProbeConfig.from_env().outdir
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
