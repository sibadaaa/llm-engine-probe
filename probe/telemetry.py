"""GPU 遥测抽象层：多厂商 SMI 工具 → 统一的 JSONL 记录。

各家 SMI（hy-smi / nvidia-smi / rocm-smi / npu-smi / xpu-smi）输出的表格格式
各不相同。本模块把它们归一化成同一种记录结构（``gpu_telem.jsonl`` 每行一条）::

    {
      "ts": 1739500000.0, "time": "2026-...", "vendor": "hygon",
      "smi": "hy-smi", "ok": true,
      "cards": [{"index": 0, "name": "Z100L", "temp_c": 45.0, "power_w": 120.0,
                 "power_cap_w": 300.0, "vram_perc": 55.0,
                 "vram_used_mb": 12345, "vram_total_mb": 16384,
                 "util_perc": null, "perf": null, "mode": null}],
      "vram_perc_sum": 55.0, "vram_used_mb_sum": 12345,
      "power_w_sum": 120.0, "temp_c_max": 45.0, "util_perc_mean": null
    }

新增一家厂商的 SMI 格式：实现一个 ``SmiBackend`` 子类并在 ``BACKEND_BY_SMI``
注册（同时参考 vendors/README.md 的适配指南）。
"""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .vendors import VendorProfile, resolve_vendor

# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

_NA_VALUES = {"n/a", "na", "none", "", "[n/a]", "-", "unknown"}


def _num(value: Any) -> float | None:
    """尽力把 SMI 输出的字段转成 float；N/A 等一律返回 None。"""
    if value is None:
        return None
    text = str(value).strip().strip("%").strip()
    if text.lower() in _NA_VALUES:
        return None
    try:
        return float(text)
    except ValueError:
        m = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(m.group(0)) if m else None


def _norm_card(**kwargs: Any) -> dict[str, Any]:
    card = {
        "index": None, "name": None, "temp_c": None, "power_w": None,
        "power_cap_w": None, "vram_perc": None, "vram_used_mb": None,
        "vram_total_mb": None, "util_perc": None, "perf": None, "mode": None,
    }
    card.update(kwargs)
    # 派生 vram_perc
    if card["vram_perc"] is None and card["vram_used_mb"] and card["vram_total_mb"]:
        card["vram_perc"] = round(100.0 * card["vram_used_mb"] / card["vram_total_mb"], 1)
    return card


def _b_to_mb(value: float | None) -> float | None:
    return None if value is None else round(value / (1024 * 1024), 1)


# ---------------------------------------------------------------------------
# Backend 基类
# ---------------------------------------------------------------------------


@dataclass
class SmiBackend:
    """一种 SMI 输出格式的解析器 + 命令行。"""

    name: str = "base"

    def commands(self) -> list[list[str]]:
        """按优先级返回候选命令（第一个输出能被 parse 出数据即采用）。"""
        return []

    def parse(self, text: str) -> list[dict[str, Any]]:
        return []

    # -- 便捷：执行命令并解析 --
    def run(self, timeout: float = 15.0) -> tuple[str, list[dict[str, Any]]]:
        last_cmd = ""
        for cmd in self.commands():
            if not shutil.which(cmd[0]):
                continue
            last_cmd = " ".join(cmd)
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if proc.returncode != 0:
                continue
            cards = self.parse(proc.stdout)
            if cards:
                return last_cmd, cards
        return last_cmd, []


# ---------------------------------------------------------------------------
# hy-smi（Hygon DCU）—— 真正的现网格式（正则来自本项目最早的 hy-smi 采样脚本，已迁移到本模块）
# ---------------------------------------------------------------------------


@dataclass
class HyBackend(SmiBackend):
    name: str = "hy-smi"

    _ROW_RE = re.compile(
        r"^\s*(\d+)\s+([\d.]+)C\s+([\d.]+)W\s+(\S+)\s+([\d.]+)W\s+(\d+)%\s+(\S+)\s+(\S+)\s*$"
    )

    def commands(self) -> list[list[str]]:
        return [["hy-smi"]]

    def parse(self, text: str) -> list[dict[str, Any]]:
        cards = []
        for line in text.splitlines():
            m = self._ROW_RE.match(line)
            if not m:
                continue
            cards.append(_norm_card(
                index=int(m.group(1)),
                temp_c=_num(m.group(2)),
                power_w=_num(m.group(3)),
                perf=m.group(4),
                power_cap_w=_num(m.group(5)),
                vram_perc=_num(m.group(6)),
                util_perc=None if m.group(7).upper() in ("N/A",) else _num(m.group(7)),
                mode=m.group(8),
            ))
        return cards


# ---------------------------------------------------------------------------
# nvidia-smi
# ---------------------------------------------------------------------------

_NVIDIA_FIELDS = [
    "index", "name", "temperature.gpu", "power.draw", "power.limit",
    "memory.used", "memory.total", "utilization.gpu",
]


@dataclass
class NvidiaBackend(SmiBackend):
    name: str = "nvidia-smi"

    def commands(self) -> list[list[str]]:
        query = ",".join(_NVIDIA_FIELDS)
        return [
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        ]

    def parse(self, text: str) -> list[dict[str, Any]]:
        cards = []
        for row in csv.reader(io.StringIO(text)):
            row = [cell.strip() for cell in row if cell.strip() != ""]
            if len(row) < len(_NVIDIA_FIELDS):
                continue
            try:
                index = int(row[0])
            except ValueError:
                continue
            cards.append(_norm_card(
                index=index,
                name=row[1],
                temp_c=_num(row[2]),
                power_w=_num(row[3]),
                power_cap_w=_num(row[4]),
                vram_used_mb=_num(row[5]),
                vram_total_mb=_num(row[6]),
                util_perc=_num(row[7]),
            ))
        return cards


# ---------------------------------------------------------------------------
# rocm-smi（AMD）
# ---------------------------------------------------------------------------


@dataclass
class RocmBackend(SmiBackend):
    name: str = "rocm-smi"

    def commands(self) -> list[list[str]]:
        return [
            # 新版本支持 --json（字段名用模糊匹配，跨版本稳）
            ["rocm-smi", "--showmeminfo", "vram", "--showtemp", "--showpower",
             "--showuse", "--json"],
            # 旧版本回退到人类可读输出
            ["rocm-smi", "--showmeminfo", "vram", "--showtemp", "--showpower", "--showuse"],
        ]

    # -- JSON 模式 --
    def _parse_json(self, text: str) -> list[dict[str, Any]]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, dict):
            return []
        cards = []
        for key in sorted(payload):
            if not re.match(r"^card\d+$", str(key)):
                continue
            info = payload[key]
            if not isinstance(info, dict):
                continue
            index = int(re.sub(r"\D", "", key) or len(cards))
            card: dict[str, Any] = {"index": index}
            for raw_key, raw_val in info.items():
                k = str(raw_key).lower()
                v = _num(raw_val)
                if "series" in k or "card series" in k:
                    card["name"] = str(raw_val)
                elif "used memory" in k and "(b)" in k:
                    card["vram_used_mb"] = _b_to_mb(v)
                elif "total memory" in k and "(b)" in k and v:
                    card["vram_total_mb"] = _b_to_mb(v)
                elif "temperature" in k and v is not None and "temp_c" not in card:
                    card["temp_c"] = v  # 取第一个温度传感器（通常是 edge）
                elif "power" in k and "(w)" in k and v is not None and "power_w" not in card:
                    card["power_w"] = v
                elif ("gpu use" in k or "utilization" in k) and v is not None:
                    card["util_perc"] = v
            cards.append(_norm_card(**card))
        return cards

    # -- 人类可读模式（GPU[N] : 字段: 值）--
    _ROW_RE = re.compile(r"^GPU\[(\d+)\]\s*:\s*([^:]+):\s*(.+)$")

    def _parse_human(self, text: str) -> list[dict[str, Any]]:
        by_index: dict[int, dict[str, Any]] = {}
        for line in text.splitlines():
            m = self._ROW_RE.match(line.strip())
            if not m:
                continue
            index = int(m.group(1))
            key = m.group(2).strip().lower()
            value = m.group(3).strip()
            card = by_index.setdefault(index, {"index": index})
            num = _num(value)
            if "used memory" in key and "(b)" in key:
                card["vram_used_mb"] = _b_to_mb(num)
            elif "total memory" in key and "(b)" in key:
                card["vram_total_mb"] = _b_to_mb(num)
            elif "temperature" in key and num is not None and "temp_c" not in card:
                card["temp_c"] = num
            elif "power" in key and "(w)" in key and num is not None and "power_w" not in card:
                card["power_w"] = num
            elif "use (%)" in key or "utilization" in key:
                card["util_perc"] = num
            elif "series" in key:
                card["name"] = value
        return [_norm_card(**card) for _, card in sorted(by_index.items())]

    def parse(self, text: str) -> list[dict[str, Any]]:
        return self._parse_json(text) or self._parse_human(text)


# ---------------------------------------------------------------------------
# npu-smi（Huawei Ascend）
# ---------------------------------------------------------------------------

# 第一行（含名称与健康状态，名称必含字母）： | 0       910B2              | OK     | 92.3  42  0 / 0 |
# 第二行（设备号与 Bus-Id）：               | 0       0                  | 0000:81:00.0 | 0  0 / 65536 |
_NPU_HEAD_RE = re.compile(
    r"^\|\s*(\d+)\s+(\w*[A-Za-z]\w*)\s*\|\s*([A-Za-z][A-Za-z_ ]*?)\s*\|\s*"
    r"([\d.]+|N/A)\s+([\d.]+|N/A)"
)
_NPU_SUB_RE = re.compile(
    r"^\|\s*(\d+)\s+(\d+)\s*\|\s*([0-9a-fA-F:.]+)\s*\|\s*([\d.NA]+)\s+(\d+)\s*/\s*(\d+)"
)


@dataclass
class NpuBackend(SmiBackend):
    name: str = "npu-smi"

    def commands(self) -> list[list[str]]:
        return [["npu-smi", "info"]]

    def parse(self, text: str) -> list[dict[str, Any]]:
        cards: dict[int, dict[str, Any]] = {}
        pending: int | None = None  # 上一行是表头行、等待子行的 npu id
        for line in text.splitlines():
            m = _NPU_HEAD_RE.match(line)
            if m:
                npu_id = int(m.group(1))
                cards[npu_id] = _norm_card(
                    index=npu_id,
                    name=m.group(2),
                    power_w=_num(m.group(4)),
                    temp_c=_num(m.group(5)),
                )
                pending = npu_id
                continue
            if pending is not None:
                m2 = _NPU_SUB_RE.match(line)
                if m2 and int(m2.group(1)) == pending:
                    card = cards[pending]
                    card["util_perc"] = _num(m2.group(4))
                    card["vram_used_mb"] = _num(m2.group(5))
                    card["vram_total_mb"] = _num(m2.group(6))
                    if card["vram_total_mb"]:
                        card["vram_perc"] = round(
                            100.0 * card["vram_used_mb"] / card["vram_total_mb"], 1
                        )
                pending = None
        return [cards[k] for k in sorted(cards)]


# ---------------------------------------------------------------------------
# xpu-smi（Intel）
# ---------------------------------------------------------------------------


@dataclass
class XpuBackend(SmiBackend):
    name: str = "xpu-smi"

    _KV_RE = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*$")

    def commands(self) -> list[list[str]]:
        return [["xpu-smi", "stats"], ["xpu-smi", "stats", "-j"]]

    def _parse_table(self, text: str) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        cur: dict[str, Any] | None = None
        for line in text.splitlines():
            m = self._KV_RE.match(line.strip())
            if not m:
                continue
            key = m.group(1).strip().lower()
            value = m.group(2).strip()
            if key.startswith("device id"):
                if cur:
                    cards.append(cur)
                cur = {"index": int(value) if value.isdigit() else len(cards)}
                continue
            if cur is None:
                continue
            num = _num(value)
            if "gpu utilization" in key or "gpu use" in key:
                cur["util_perc"] = num
            elif "power" in key and num is not None:
                cur["power_w"] = num
            elif "temperature" in key and num is not None:
                cur["temp_c"] = num
            elif "memory used" in key:
                cur["vram_used_mb"] = num
            elif "memory size" in key or "memory total" in key:
                cur["vram_total_mb"] = num
        if cur:
            cards.append(cur)
        return [_norm_card(**c) for c in cards]

    def parse(self, text: str) -> list[dict[str, Any]]:
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return self._parse_json(text)
        return self._parse_table(text)

    def _parse_json(self, text: str) -> list[dict[str, Any]]:
        """xpu-smi -j 的 JSON 形态因版本而异，做宽松字段匹配。"""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
        items = payload if isinstance(payload, list) else [payload]
        cards = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            flat: dict[str, Any] = {}

            def walk(node: Any) -> None:
                if isinstance(node, dict):
                    for k, v in node.items():
                        if isinstance(v, (dict, list)):
                            walk(v)
                        else:
                            flat[str(k).lower()] = v
                elif isinstance(node, list):
                    for v in node:
                        walk(v)

            walk(item)
            card: dict[str, Any] = {"index": i}
            for key, value in flat.items():
                num = _num(value)
                if "device" in key and "id" in key and num is not None:
                    card["index"] = int(num)
                elif "utilization" in key or "gpu use" in key:
                    card["util_perc"] = num
                elif "power" in key:
                    card["power_w"] = num
                elif "temperature" in key or "temp" in key:
                    card["temp_c"] = num
                elif "used" in key and "memory" in key:
                    card["vram_used_mb"] = num
                elif ("total" in key or "size" in key) and "memory" in key:
                    card["vram_total_mb"] = num
            cards.append(_norm_card(**card))
        return cards


# ---------------------------------------------------------------------------
# 注册表与选择
# ---------------------------------------------------------------------------

BACKEND_BY_SMI: dict[str, type[SmiBackend]] = {
    "hy-smi": HyBackend,
    "nvidia-smi": NvidiaBackend,
    "rocm-smi": RocmBackend,
    "amd-smi": RocmBackend,
    "npu-smi": NpuBackend,
    "xpu-smi": XpuBackend,
}

# 厂商未知时的嗅探顺序（按 SMI 命令是否在 PATH）
_SNIFF_ORDER: list[tuple[str, type[SmiBackend]]] = [
    ("npu-smi", NpuBackend),
    ("nvidia-smi", NvidiaBackend),
    ("hy-smi", HyBackend),
    ("rocm-smi", RocmBackend),
    ("amd-smi", RocmBackend),
    ("xpu-smi", XpuBackend),
]


def candidate_backends(profile: VendorProfile | None = None) -> list[SmiBackend]:
    """按厂商 profile 给出候选 backend（未知厂商时按 SMI 命令可用性嗅探）。"""
    profile = profile or resolve_vendor()
    instances: list[SmiBackend] = []
    seen: set[str] = set()
    smi_names = [profile.smi_cmd, *profile.smi_fallbacks]
    for smi in smi_names:
        if not smi or smi in seen:
            continue
        seen.add(smi)
        cls = BACKEND_BY_SMI.get(smi)
        if cls is not None:
            instances.append(cls())
    if instances:
        return instances
    for smi, cls in _SNIFF_ORDER:
        if smi in seen or not shutil.which(smi):
            continue
        instances.append(cls())
    return instances


def backend_by_name(name: str) -> SmiBackend:
    """按 backend 名（hy-smi/nvidia-smi/... 或类名）取实例（供 --parse 使用）。"""
    cls = BACKEND_BY_SMI.get(name)
    if cls is None:
        reverse = {
            "hy": HyBackend, "hy-smi": HyBackend, "hygon": HyBackend,
            "nvidia": NvidiaBackend,
            "rocm": RocmBackend, "amd": RocmBackend,
            "ascend": NpuBackend, "npu": NpuBackend,
            "intel": XpuBackend,
        }
        cls = reverse.get(name)
    if cls is None:
        raise KeyError(f"未知 SMI backend：{name}（可用：{sorted(BACKEND_BY_SMI)}）")
    return cls()


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------


def summarize_cards(cards: list[dict[str, Any]]) -> dict[str, Any]:
    vram = [c["vram_perc"] for c in cards if c.get("vram_perc") is not None]
    used = [c["vram_used_mb"] for c in cards if c.get("vram_used_mb") is not None]
    power = [c["power_w"] for c in cards if c.get("power_w") is not None]
    temp = [c["temp_c"] for c in cards if c.get("temp_c") is not None]
    util = [c["util_perc"] for c in cards if c.get("util_perc") is not None]
    return {
        "vram_perc_sum": round(sum(vram), 1) if vram else None,
        "vram_used_mb_sum": round(sum(used), 1) if used else None,
        "power_w_sum": round(sum(power), 1) if power else None,
        "temp_c_max": round(max(temp), 1) if temp else None,
        "util_perc_mean": round(sum(util) / len(util), 1) if util else None,
    }


def sample_once(profile: VendorProfile | None = None) -> dict[str, Any]:
    """采一次样：依次尝试该厂商的 backend，返回归一化记录。"""
    profile = profile or resolve_vendor()
    record: dict[str, Any] = {
        "ts": time.time(),
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "vendor": profile.id,
        "smi": None,
        "ok": False,
        "cards": [],
        "error": None,
    }
    errors = []
    for backend in candidate_backends(profile):
        cmd, cards = backend.run()
        if cards:
            record["smi"] = cmd or backend.name
            record["cards"] = cards
            record["ok"] = True
            record.update(summarize_cards(cards))
            return record
        errors.append(backend.name)
    record["error"] = "无可用 SMI 或解析为空（尝试过：" + ", ".join(errors) + "）"
    return record


def format_record(record: dict[str, Any]) -> str:
    if not record.get("ok"):
        return f"[遥测] 无数据：{record.get('error')}"
    parts = [f"显存%合计={record.get('vram_perc_sum')}"]
    if record.get("power_w_sum") is not None:
        parts.append(f"功耗W合计={record['power_w_sum']}")
    if record.get("temp_c_max") is not None:
        parts.append(f"温度max={record['temp_c_max']}C")
    if record.get("util_perc_mean") is not None:
        parts.append(f"利用率均值={record['util_perc_mean']}%")
    head = f"[遥测 {record['smi']}] " + "  ".join(parts)
    lines = [head]
    for card in record.get("cards", []):
        detail = []
        if card.get("vram_perc") is not None:
            detail.append(f"VRAM={card['vram_perc']}%")
        if card.get("vram_used_mb") is not None:
            detail.append(f"used={card['vram_used_mb']}MB")
        if card.get("power_w") is not None:
            detail.append(f"pwr={card['power_w']}W")
        if card.get("temp_c") is not None:
            detail.append(f"temp={card['temp_c']}C")
        name = card.get("name") or ""
        lines.append(f"  card{card.get('index')} {name}: " + "  ".join(detail))
    return "\n".join(lines)


def sample_loop(
    duration: float,
    interval: float,
    out_jsonl: Path,
    raw_file: Path | None = None,
    profile: VendorProfile | None = None,
    print_each: bool = False,
) -> int:
    """循环采样：追加写 JSONL（+ 可选原始文本），返回采样次数。"""
    profile = profile or resolve_vendor()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    raw_handle = None
    if raw_file is not None:
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        raw_handle = raw_file.open("a", encoding="utf-8")
    count = 0
    end = time.time() + max(0.0, duration)
    try:
        while True:
            record = sample_once(profile)
            with out_jsonl.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if print_each:
                print(format_record(record), flush=True)
            if raw_handle is not None and record.get("ok"):
                raw_handle.write(
                    f"# ts={record['ts']:.3f} smi={record['smi']} cards={len(record['cards'])}\n"
                )
            count += 1
            if time.time() >= end:
                break
            time.sleep(max(0.05, interval))
    finally:
        if raw_handle is not None:
            raw_handle.close()
    return count


def vram_perc_sum(profile: VendorProfile | None = None) -> float | None:
    """各卡 VRAM 占用百分比之和（无数据返回 None）——stop_vllm.sh 等待显存释放用。"""
    record = sample_once(profile)
    return record.get("vram_perc_sum")


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """对一批 JSONL 记录做汇总（报告用）。"""
    import statistics

    ok = [r for r in records if r.get("ok")]
    if not ok:
        return {"samples": len(records), "ok_samples": 0}

    def series(key: str) -> list[float]:
        return [r[key] for r in ok if r.get(key) is not None]

    vram = series("vram_perc_sum")
    power = series("power_w_sum")
    temp = series("temp_c_max")
    util = series("util_perc_mean")
    card_names = sorted({c.get("name") for r in ok for c in r.get("cards", []) if c.get("name")})
    return {
        "samples": len(records),
        "ok_samples": len(ok),
        "smi": ok[0].get("smi") if ok else None,
        "vendor": ok[0].get("vendor") if ok else None,
        "cards": card_names,
        "vram_perc_sum": {
            "mean": statistics.fmean(vram), "max": max(vram)} if vram else None,
        "power_w_sum": {
            "mean": statistics.fmean(power), "max": max(power)} if power else None,
        "temp_c_max": {
            "mean": statistics.fmean(temp), "max": max(temp)} if temp else None,
        "util_perc_mean": {
            "mean": statistics.fmean(util), "max": max(util)} if util else None,
    }


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """读取 gpu_telem.jsonl（容忍坏行）。"""
    records: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records
