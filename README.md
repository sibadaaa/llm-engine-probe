# vllm-docker-probe · vLLM 推理全流程探测工具箱（多厂商）

在 Docker 中启动一个**可观测的 vLLM 实例**，并把一次推理从「HTTP 请求 → 调度排队 → 预填充 → 解码 → 投机解码验证 → GPU kernel」的**每一步**都采出来，最后汇总成一份中文 Markdown 报告，直接告诉你下一步该优化什么。

**多厂商支持**：容器启动、设备直通、GPU 遥测（SMI）、算子归因规则、报告文案全部走厂商抽象层（`vendors/` ↔ `probe/vendors.py`）。内置 **NVIDIA / AMD / Hygon / Intel / Ascend** 五家适配，其他厂商按 `vendors/README.md` 的模板自行扩展；另有 `generic` 兜底用于 CI / 无 GPU 环境。

**标准化对比**：`make suite` 跑固定 8 场景矩阵，产出稳定 schema 的 `suite_results.json`；`make compare A=… B=…` 跨厂商/跨机器并排对比。

本工具箱最初针对以下**已用真实部署验证过**的现网环境开发（Hygon DCU 适配即由此而来）：

| 项目 | 值 |
|---|---|
| 加速卡 | Hygon DCU Z100 / Z100L（DTK 26.04，ROCm 系）；其余四家为结构适配，见 §4b |
| vLLM | `0.18.1+das.3266200.dtk2604`（源码 checkout：`https://github.com/sibadaaa/z100-vllm`，容器内挂到 `/src`） |
| 现网模型 | `qwen3.8-27b`（FP8/INT8 + DFlash2 投机解码 k=7，TP=4，`--enforce-eager`） |
| 容器配方 | 由厂商 profile 生成（hygon：`/dev/kfd /dev/mkfd /dev/dri` + video 组 + `/opt/hyhal`），镜像 `z100-s9-snap-s14`(本地镜像) |

---

## 1. 快速开始（三步）

```bash
cd /vllm-docker-probe

# ① 复制配置模板，按需修改（重点是 VENDOR / PORT / MODEL / PROBE_LEVEL）
cp .env.example .env

# ②（可选）看看厂商检测结果与能力矩阵
python3 -m probe vendor        # 或 make vendor

# ③ 起一个"可观测"的实例（默认端口 8100，刻意避开现网的 8000）
make up                      # 等价于 scripts/start_vllm.sh --level steps
#   想连 profiler 一起开：make up-kernel

# ④ 一键全链路探测 + 出报告
make probe                   # 等价于 scripts/probe_all.sh
cat runs/<时间戳>/REPORT.md   # 报告里的第 8 节是自动归因与优化建议
```

只想先看看现状、不确定配置对不对：

```bash
make status     # 容器 / 端点 / 关键指标 / 最近日志
make env        # 自检：哪些观测开关真的生效了
```

> ⚠️ **安全约定**：所有默认值都指向 **8100** 和容器 `vllm-probe`，与现网实例（:8000）隔离。
> 不要用默认参数去打 8000，除非你明确是在给正在跑的实验做无干扰探测（见 §9）。

---

## 2. 目录结构

```
vllm-docker-probe/
├── README.md                     ← 本文件（使用说明）
├── docs/观测字段对照表.md          ← 字段级参考：每一步有哪些指标/日志/事件
├── .env.example                  ← 全部可调参数（含多厂商示例块）
├── Makefile                      ← 快捷入口（make help）
├── vendors/                      ← 厂商适配层（Shell 端）：nvidia/amd/hygon/intel/ascend
│   ├── README.md                 ← 新增一家厂商的完整指南
│   ├── _template.sh              ← 适配模板（含每个变量说明）
│   └── <vendor>.sh               ← 设备直通/环境注入/SMI 命令/探测元数据
├── docker/
│   ├── Dockerfile                ← 探针就绪镜像（叠加在任意厂商的 vLLM 基础镜像上）
│   ├── docker-compose.yml        ← 容器壳定义（厂商中立）
│   ├── docker-compose.<vendor>.yml ← 各厂商设备直通 overlay（hygon/nvidia/amd/intel/ascend）
│   └── docker-compose.obs.yml    ← 可选：Prometheus + Grafana + OTel 叠加
├── configs/                      ← Prometheus 抓取配置、Grafana 看板、OTel 采集器
├── probe/                        ← Python 工具包（纯标准库，容器内也能跑）
│   ├── vendors.py                ← 厂商注册表（分析侧：trace 规则包/报告文案/诊断阈值）
│   ├── telemetry.py              ← GPU 遥测抽象：5 家 SMI → 统一 gpu_telem.jsonl
│   ├── suite.py                  ← 标准化对比套件（固定场景矩阵 + 跨厂商并排对比）
│   ├── metrics_probe.py          ← /metrics 解析、直方图分位、阶段表
│   ├── http_probe.py             ← 请求级打点（TTFT/ITL/TPOT/E2E）
│   ├── log_probe.py              ← 引擎日志 → 逐 step 结构化事件
│   ├── trace_probe.py            ← torch profiler trace → kernel/CPU 归因（厂商规则包）
│   ├── plugin_records.py         ← stat logger 插件 JSONL → 汇总
│   ├── report.py                 ← 中文报告 + 自动归因规则（阈值可按厂商覆盖）
│   └── cli.py                    ← 统一入口：python3 -m probe <子命令>
├── plugin_probe/                 ← vLLM stat logger 插件（可选，entry point 安装）
│   ├── pyproject.toml            ← 注册 vllm.stat_logger_plugins 入口
│   └── vllm_probe_stat_logger/   ← records.py（纯标准库抽取）+ __init__.py（StatLoggerBase 子类）
├── scripts/
│   ├── start_vllm.sh stop_vllm.sh wait_ready.sh status.sh
│   ├── collect_metrics.sh probe_http.sh watch_steps.sh
│   ├── trace_capture.sh pyspy.sh gpu_sample.sh run_suite.sh
│   ├── probe_all.sh report.sh build_image.sh
│   └── lib/common.sh             ← 配置 + 厂商检测 + docker 辅助
├── samples/                      ← 真实日志/指标/profiler/SMI 样例（测试用）
├── tests/                        ← 单元测试 + mock 服务 + 端到端自测
└── runs/<时间戳>/                 ← 每次探测的全部产物（报告就在这）
```

---

## 3. 观测档位（`PROBE_LEVEL`）

这是本工具箱的核心开关：**按需决定"探多深"**，因为探得越深、对性能的扰动越大。

| 档位 | 打开的东西 | 能回答的问题 | 额外开销 |
|---|---|---|---|
| `off` | 不加任何可选开关 | 纯基线对照 | 0 |
| `basic` | 默认周期 stats 日志 + `/metrics` | 吞吐、队列、KV 使用率的粗粒度趋势 | 极低 |
| **`steps`**（默认） | `--enable-logging-iteration-details`、`--enable-log-requests`、`--enable-per-request-metrics`、`--enable-prompt-tokens-details`、`--enable-force-include-usage`、`--max-log-len` | **每个 engine step 的批次构成与耗时**；每个请求的 queue/TTFT/decode 精确值；前缀缓存命中 token 数 | 低（逐 step 日志量约 5 行/秒） |
| `deep` | `steps` + `--kv-cache-metrics`、`--cudagraph-metrics`、`--enable-mfu-metrics`、`--per-request-spec-decode-metrics detailed` | KV block 生命周期/复用间隔、图调度命中、MFU/带宽、**逐步投机解码接受数组** | 中（采样型指标） |
| `kernel` | `deep` + `--profiler-config`（torch）+ `--enable-layerwise-nvtx-tracing` + `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` | **kernel/算子级**时间归因：GEMM、通信、Attention 各占多少；`schedule: *`、`gpu_model_runner: *` 等**引擎内部子步骤**的 CPU 占比；CPU/GPU 空洞 | 高（只为定位问题临时开，会自动关掉图模式） |

> 还有一个**与档位正交**的开关：`PROBE_PLUGIN=1` 启用 stat logger 插件（见 §6.7），
> 把每个 engine step 的结构化统计写成 JSONL，不受档位限制。

切换档位不需要改代码，两种方式：

```bash
# 临时：命令行覆盖
PROBE_LEVEL=kernel scripts/start_vllm.sh

# 常用：改 .env 里的 PROBE_LEVEL，或者用 Makefile
make up-kernel
```

> `kernel` 档会自动追加 `--enforce-eager`：因为 `--enable-layerwise-nvtx-tracing` 与 HIP/CUDA Graph 不兼容（vLLM 官方限制）。要测真实图模式，请用 `kernel` 档做算子归因、用 `deep`/`steps` 档做性能数字。

---

## 4. 「推理步骤 → 观测手段」对照表

这张表是整套工具箱的设计依据：**每一步都有对应的数据源和命令**。

| # | 推理步骤 | 观测数据源 | 采集命令 | 关键字段 |
|---|---|---|---|---|
| ① | **HTTP 接入 / 排队** | 客户端打点 + `/metrics` 直方图 | `scripts/probe_http.sh` | `headers_ms`、`queue_time_ms`、`vllm:request_queue_time_seconds`、`vllm:num_requests_waiting{reason}` |
| ② | **调度组批** | 逐 iteration 日志 | `scripts/watch_steps.sh` | `ctx_reqs/ctx_tokens/gen_reqs/gen_tokens`、`Running/Waiting` |
| ③ | **预填充 (Prefill)** | 逐 step 日志 + 直方图 + trace 注解 | `probe_http.sh` + `trace_capture.sh` | `ms/1k context token`、`vllm:request_prefill_time_seconds`、`request_prefill_kv_computed_tokens` |
| ④ | **解码 (Decode)** | ITL 直方图 + 逐 step 日志 | `watch_steps.sh` | `decode ms/token`、`vllm:inter_token_latency_seconds`、`vllm:request_time_per_output_token_seconds` |
| ⑤ | **投机解码验证** | 周期日志 + 响应体 | `make probe`（deep 档） | 平均接受长度、按位置接受率、`per_step_accepted` |
| ⑥ | **KV 缓存 / 前缀缓存** | `/metrics` + `cache_config_info` | `scripts/collect_metrics.sh` | `kv_cache_usage_perc`、`prefix_cache_hit_rate`、`kv_block_lifetime/idle/reuse_gap` |
| ⑦ | **算子 / kernel** | torch profiler trace | `scripts/trace_capture.sh` | GEMM/通信/Attention 占比、Top kernel、GPU 利用率、>1ms 空洞 |
| ⑧ | **CPU 侧（调度/后处理）** | py-spy 栈 + 日志量 | `scripts/pyspy.sh` | Python 火焰图、`hipEventSynchronize` 占比 |
| ⑨ | **硬件（显存/功耗/温度）** | 各厂商 SMI 采样（hy-smi/nvidia-smi/rocm-smi/npu-smi/xpu-smi） | `scripts/gpu_sample.sh`（或 `make gpu-sample`） | 每卡 VRAM%、功耗、温度、利用率（统一 JSONL） |
| ⑩ | **引擎内部对象（最深一层）** | stat logger 插件（`vllm.stat_logger_plugins`） | `PROBE_PLUGIN=1` + `python3 -m probe plugin` | 逐 step `SchedulerStats`/`IterationStats`、**逐请求 queue/prefill/decode/e2e/TPOT**、KV 驱逐、按位置接受数 |
| ⑪ | **CPU 侧阶段分布** | `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` → trace 注解 | `make up-kernel` + `make trace` | `schedule: allocate_slots`、`gpu_model_runner: forward/sample/postprocess`、`llm_engine step: *` |

字段级的完整对照（含每个指标的标签、日志正则、trace 事件分类）见 [`docs/观测字段对照表.md`](docs/观测字段对照表.md)。

---

## 4b. 多厂商适配层（vendors/）

### 内置厂商矩阵

| 厂商 | `VENDOR=` | 设备直通（docker run） | 可见性环境变量 | 遥测 SMI | 启动前 source | 自动探测命中条件 |
|---|---|---|---|---|---|---|
| NVIDIA | `nvidia` | `--gpus all`（需 nvidia-container-toolkit） | `CUDA_VISIBLE_DEVICES` | `nvidia-smi` | — | `nvidia-smi` 或 `/dev/nvidia0` |
| AMD | `amd` | `/dev/kfd /dev/dri` + video 组 | `HIP_VISIBLE_DEVICES` | `amd-smi`/`rocm-smi` | — | `amd-smi`/`rocm-smi` 或 `/dev/kfd` |
| Hygon | `hygon` | `/dev/kfd /dev/mkfd /dev/dri` + video 组 + `/opt/hyhal` | `HIP_VISIBLE_DEVICES` | `hy-smi`（回退 rocm-smi） | DTK / DCU env 脚本 | `hy-smi` 或 `/dev/mkfd` |
| Intel | `intel` | `/dev/dri` + video 组 | `ZE_AFFINITY_MASK` | `xpu-smi` | — | `xpu-smi` |
| Ascend | `ascend` | `/dev/davinci*` + 管理节点（按 VISIBLE_DEVICES 动态） | `ASCEND_RT_VISIBLE_DEVICES` | `npu-smi` | `set_env.sh` | `npu-smi` 或 `/dev/davinci0` |
| 通用/CI | `generic` | 不传设备参数 | — | （无） | — | 兜底（探测失败时） |

探测顺序 `ascend → nvidia → hygon → amd → intel`，可用 `.env` 的 `VENDOR=` 显式覆盖（默认 `auto`）。

### 每层的厂商适配点

| 层 | Shell 端（`vendors/<id>.sh`） | Python 端（`probe/vendors.py`） |
|---|---|---|
| 容器启动 | 设备直通参数、挂载、前置 source 脚本、可见性/分配器环境变量 | —（启动只走 Shell） |
| GPU 遥测 | `VENDOR_SMI_CMD` | `telemetry.py` 的 SMI backend（解析各家输出格式） |
| 算子归因 | — | `trace_pack` → `trace_probe.py::CATEGORY_RULE_PACKS` 规则包（cublas/hipblas/aclnn/onednn 等） |
| 报告/诊断 | — | 图模式称谓（CUDA Graph/HIP Graph/ACL Graph）、文案、`diagnose_overrides` 阈值覆盖 |

新增一家厂商：`cp vendors/_template.sh vendors/<id>.sh` 填好配置，再在
`probe/vendors.py` 注册同 id 的 `VendorProfile`（trace 规则包可选），跑
`bash tests/run_tests.sh` 会有一致性守卫强制两端对齐。完整指南见
[`vendors/README.md`](vendors/README.md)。

### 自定义挂载与数据盘

厂商无关的额外挂载用 `.env` 的 `EXTRA_MOUNTS`（空格分隔的 `host:container[:ro]`）；
`GAME_DIR` 是本站点的共享数据目录（按原路径挂进容器）。compose 场景用
`docker/docker-compose.override.yml` 追加。

---

## 5. 命令速查

### Makefile

```bash
make help        # 列出全部目标
make vendor      # 厂商检测结果 / SMI 可用性 / trace 规则包
make up          # 起可观测实例（LEVEL=steps，可覆盖：make up LEVEL=deep）
make up-kernel   # kernel 档（profiler + 逐层 NVTX）
make down        # 停实例（保留容器与日志）
make status      # 状态 + 关键指标
make env         # 观测开关自检
make probe       # 一键全链路 + 报告
make probe-deep  # 全链路 + GPU 遥测 + py-spy
make trace       # 只抓一份 torch trace 并分析
make steps       # 只做 step 级摘要
make watch       # 实时逐 step 观察
make report      # 重新生成报告
make suite       # 标准化对比套件（8 场景 → suite_results.json）
make compare     # 对比两次套件结果：make compare A=… B=…（跨厂商）
make gpu-sample  # 采样 GPU 遥测 60 秒（各厂商 SMI）
make test        # 单元测试 + mock 端到端自测
make obs-up      # 起 Prometheus + Grafana（可选观测栈）
make compose-up  # 用 compose 起容器壳（VENDOR 自动探测，可覆盖）
```

### 脚本直调（Makefile 只是薄封装）

```bash
VENDOR=nvidia scripts/start_vllm.sh --dry-run   # 查看某厂商将执行的 docker 命令
scripts/start_vllm.sh [--level steps|deep|kernel] [--dry-run] [--exec]
scripts/stop_vllm.sh  [--rm] [--kill-exec]
scripts/status.sh ; scripts/wait_ready.sh
scripts/collect_metrics.sh [--tag before] [--diff a.json b.json --elapsed 60] [--watch 5]
scripts/probe_http.sh  [--prompt-tokens 4096 --max-tokens 128 --concurrency 8 --repeat 3]
scripts/watch_steps.sh [--summary | --tail 2000 | --docker]
scripts/trace_capture.sh [--analyze-only | --skip-load]
scripts/pyspy.sh --mode dump|flame|top [--duration 30]
scripts/gpu_sample.sh --duration 60
scripts/run_suite.sh [--scenarios s02,s05 | --list]
scripts/probe_all.sh [--with-gpu-sample] [--with-pyspy] [--skip-trace]
scripts/report.sh [--open]
```

### Python 工具包

```bash
python3 -m probe env       --port 8100                 # 能力自检
python3 -m probe vendor                                # 厂商检测/能力矩阵
python3 -m probe metrics   --port 8100 --print         # 指标快照 + 阶段表
python3 -m probe http      --port 8100 --concurrency 8 # 请求级
python3 -m probe logs      --file runs/x/serve.log --live
python3 -m probe trace     --path runs/x/traces/rank0.*.pt.trace.json.gz
python3 -m probe telemetry --duration 60 --outdir runs/x   # GPU 遥测（各厂商 SMI）
python3 -m probe plugin    --file runs/x/probe_steps.jsonl # 插件 JSONL 汇总
python3 -m probe suite     --port 8100 --list              # 标准化套件矩阵
python3 -m probe compare   --a runs/a/suite_results.json --b runs/b/suite_results.json
python3 -m probe report    --outdir runs/x
```

---

## 5b. 标准化对比测试套件（suite / compare）

> 目的：**让"哪家卡/哪台机器/哪组参数更快"变成可复现、可并排看的数字**，
> 而不是各跑各的、口径不一。

### 固定 8 场景矩阵（temperature=0，确定性 prompt）

| 场景 | 负载 | 回答的问题 |
|---|---|---|
| `s01_warmup` | p512 / g8，1 次（不计入对比） | 预热 |
| `s02_chat_short_c1` | p512 / g128 ×3 | 短对话基线 |
| `s03_decode_heavy_c1` | p512 / g512 ×2 | 纯解码能力（TPOT/ITL） |
| `s04_prefill_heavy` | p16k / g32 ×2 | 预填充吞吐 |
| `s05_concurrent_c8` | p4k / g128 ×8 ×2 | 并发吞吐与排队 |
| `s06_prefix_cache` | 共享前缀 2k + p512 / g32 ×4 | 前缀缓存命中与收益 |
| `s07_long_context` | p24k / g64（可用 `SUITE_LONG_CONTEXT_TOKENS` 调整长度） | 长上下文 |
| `s08_spec_decode` | p512 / g252 ×3（可选） | 投机解码接受率 |

### 用法

```bash
make suite                          # 跑全部场景 → runs/<ts>/suite_results.json
bash scripts/run_suite.sh --scenarios s02,s05   # 只跑指定场景
python3 -m probe suite --list       # 只看矩阵

# 跨厂商/跨机器/跨参数对比（产出一张并排 Markdown 表）：
python3 -m probe compare --a runs/dcu/suite_results.json --b runs/nvidia/suite_results.json
make compare A=runs/dcu/suite_results.json B=runs/nvidia/suite_results.json OUT=COMPARE.md
```

### 结果 schema（稳定契约，跨厂商缺项为 null）

每个场景固定包含：客户端分位（TTFT/E2E/TPOT/ITL 的 p50/p90/p99）、客户端吞吐、
服务端逐请求指标（queue/TTFT/generation，需 `--enable-per-request-metrics`）、
**引擎侧 Prometheus 差分**（prompt/generation token 速率、请求速率、KV 使用率、
前缀缓存命中率增量、抢占增量）、可选的投机解码接受率。
顶部还记录厂商/硬件（一次 SMI 采样）/引擎配置（`vllm:cache_config_info` 标签）。

任何场景失败只记录错误、不中断整套（半可用的卡也能拿到能对比的部分）。

---

## 6. 每一层怎么用（含产物说明）

### 6.1 请求级：`scripts/probe_http.sh`

对每个请求逐 SSE chunk 打点，得到客户端视角的四段耗时，并抽取服务端返回的逐请求 `metrics`：

```
t_send ──► 首字节(headers_ms) ──► 首个内容块(TTFT) ──► ... ──► 最后一块(E2E)
                                        └── ITL（相邻块间隔）──┘
```

```bash
# 单请求细看（含服务端逐请求指标）
scripts/probe_http.sh --prompt-tokens 512 --max-tokens 128 --concurrency 1 --repeat 3

# 并发压一段（看 TTFT/E2E 的分位数与吞吐）
CONCURRENCY=8 REPEAT=3 PROMPT_TOKENS=4096 scripts/probe_http.sh

# 前缀缓存探测：所有请求共享一段相同前缀，观察第 2 个请求起 cached_tokens 与 TTFT
scripts/probe_http.sh --shared-prefix-tokens 2048 --repeat 5
```

产物：`http_requests.jsonl`（逐请求）、`http_summary.json`（分位数）、探测前后的 `metrics_before/after.json` 与 `metrics_diff.json`。

### 6.2 指标层：`scripts/collect_metrics.sh`

```bash
scripts/collect_metrics.sh --tag before
# ... 做一段实验 ...
scripts/collect_metrics.sh --tag after
scripts/collect_metrics.sh --diff runs/x/metrics_before.json runs/x/metrics_after.json --elapsed 60
```

`metrics --print` 会打印**推理阶段表**（队列 → 预填充 → 解码 → 运行合计 → E2E → TTFT → ITL → TPOT）的均值与 p50/p90/p99：

```
阶段                     样本     均值ms        p50        p90        p99
-------------------------------------------------------------------------
① 队列等待 queue           17        5.5       ≤300       ≤300       ≤300
② 预填充 prefill           17     3126.6     2375.0     4750.0    19150.0
③ 解码 decode              17    10561.3     6388.9    23000.0   109800.0
...
注：带 ≤ 的分位数落在该直方图的首个桶/超大跨度桶内，桶阶梯分辨率不足，
    数值只是上界；此时请以「均值」或「服务端逐请求指标」为准。
```

> **为什么会出现 `≤`**：本部署的 `vllm:request_queue_time_seconds` 桶阶梯从 `le=0.3` 起步，
> 而队列时间普遍只有几毫秒 ⇒ 所有观测都落进第一个桶，插值出来的"p50=150ms"是假精度。
> 工具箱会如实标成 `≤300`，并建议改看**服务端逐请求**的 `queue_time_ms`（那是精确值）。

### 6.3 step 级：`scripts/watch_steps.sh`

`steps` 档下每个 engine step 都会打一行，解析后就是一条结构化事件：

```
step #752     DECODE   ctx=0req/0tok gen=1req/8tok   184.79 ms kv=3.5%
[stats] running=0 waiting=0 kv=0.0% prompt=0tok/s gen=8tok/s prefix_hit=27.0%
[spec] accept_len=1.69 draft_acc=9.9% per_pos=[0.462, 0.115, 0.077, 0.038, 0.0, 0.0, 0.0]
[mfu] 0.6 TF/s/device 32.4 GB/s/device
[graph] mode=NONE count=26 unpadded=8 padded=8 pad=0
```

```bash
scripts/watch_steps.sh              # 实时跟随（Ctrl-C 结束后自动汇总）
scripts/watch_steps.sh --summary    # 只要摘要
scripts/watch_steps.sh --docker     # 直接跟 docker logs -f
```

摘要会给出：step 总数 / 纯预填充 / 纯解码 / 混合 / **空转**的构成、耗时分布、
`ms / 1k context token`、`ms / token`、**多稳态检测**（快态 vs 慢态占比 —— 本项目历史上出现过 192ms/373ms 双稳态）、
投机解码接受趋势、MFU、图调度补位比例。

### 6.4 算子级：`scripts/trace_capture.sh`

用 vLLM 自带的 torch profiler：`POST /start_profile` → 打一段负载 → `POST /stop_profile` → 落盘 → 自动分析。

```bash
make up-kernel                                  # 必须用 kernel 档启动，端点才存在
scripts/trace_capture.sh --prompt-tokens 4096 --max-tokens 64 --concurrency 8
```

真实输出示例（现网 FP8 k7 臂的一段 trace，131 MB/rank，解析约 4 秒）：

```
事件数=462041  时间窗口=7856.1ms  GPU busy=6620.9ms  利用率=84.3%  >1ms 空洞=2
GPU 叶子事件（kernel/memcpy）合计=6621.0ms  注解区间合计=7895.6ms

GPU 时间分类占比（分母=叶子 GPU 事件，故相加约为 100%）:
  量化/GEMM            4473.8 ms   67.6%  events=7698
  通信/集合通信            1464.2 ms   22.1%  events=1732
  KV/搬运               290.7 ms    4.4%  events=21743
  Attention           126.6 ms    1.9%  events=819

逐 iteration 注解归因（execute_context_X(Y)_generation_Z(W)）:
  混合 step（预填充+解码）   steps=4   GPU=5273.3 ms ctx_tokens=3640 gen_tokens=512
  解码 step（generation）   steps=8   GPU=1812.9 ms ctx_tokens=0    gen_tokens=1152
```

两点设计说明：

* **区间型事件不参与占比**：`gpu_user_annotation`（例如 `execute_context_1(455)_generation_8(64)`）
  是**覆盖整步的范围**，和它内部的 kernel 相加会重复计数。工具箱把"叶子事件（kernel/memcpy）"和
  "注解区间"分开统计，占比的分母只用叶子事件，因此能干净地闭合到 ~100%。
* **逐 iteration 注解**是本 fork 的独特信号：`execute_context_<上下文请求数>(<上下文token数>)_generation_<解码请求数>(<解码token数>)`，
  于是可以**直接把 GPU 时间拆成预填充 step 与解码 step**，并算出每 token 成本。

trace 文件保持原格式，可直接用 Chrome `chrome://tracing` 或 Perfetto 打开看完整时间线。

### 6.5 CPU 侧与硬件：`scripts/pyspy.sh`、`scripts/gpu_sample.sh`

```bash
scripts/pyspy.sh --mode dump                # 所有 vLLM 进程的栈快照
scripts/pyspy.sh --mode flame --duration 30 # 30s 火焰图（SVG，在 runs/<ts>/pyspy/）
scripts/gpu_sample.sh --duration 60         # 显存/功耗/温度采样（厂商 SMI → gpu_telem.jsonl）
```

`gpu_sample.sh` 的 SMI 命令由厂商 profile 决定（hygon=hy-smi、nvidia=nvidia-smi、
amd=amd-smi/rocm-smi、ascend=npu-smi、intel=xpu-smi），解析在
`probe/telemetry.py` 中归一化成统一 JSONL（各卡 vram_perc/power_w/temp_c/util_perc）。

用途：GPU 时间线出现空洞、或 step 时间进入"慢态"时，确认是 CPU 侧（调度/后处理/日志量）
还是硬件侧（降频、显存换出）。注意部分平台的 SMI 不提供真实利用率
（如 hy-smi 的 DCU% 常为 `N/A`），真实忙碌度以 trace 的 GPU busy/利用率 为准。

### 6.6 报告：`scripts/report.sh`

把运行目录里的所有产物合成 `<RUN_DIR>/REPORT.md`，章节固定为：

```
0. 运行概览          1. 探测开关自检       2. 请求级生命周期
3. 引擎 step 级      4. 调度/队列/KV 缓存   5. 投机解码
6. 显存/KV驻留/功耗   7. 算子/kernel 级      8. 自动归因与优化建议
9. 产物清单与复现命令
```

**第 8 节是重点**：每条结论都带触发它的证据数字和可执行建议，规则写在 `probe/report.py` 的 `diagnose()` 里，阈值集中在 `DEFAULT_THRESHOLDS`、可按厂商在 `probe/vendors.py::diagnose_overrides` 覆盖。举几个真实会触发的例子：

| 触发条件 | 结论 | 建议方向 |
|---|---|---|
| GEMM/量化算子 GPU 占比 > 45% | 访存受限型瓶颈 | 更低比特量化、增大有效 M、检查 M=1 的 padding 浪费 |
| 通信占比 > 15% | 集合通信偏重 | 查 TP/DP 切分、细碎 all-reduce 合并 |
| 慢态占比 > 20% | step 时间双稳态 | 对齐 GPU 遥测（功耗/温度降频）、KV 换出、批次构成 |
| 队列 p90 > 50ms 或出现抢占 | 调度/KV 容量不足 | 调 `--max-num-seqs`/`--max-num-batched-tokens`/`--max-model-len` |
| 前缀缓存命中率 < 20% | 前缀没被复用 | 确认 `--enable-prefix-caching`、稳定前缀前置 |
| 平均接受长度 < 1.5 | 草稿模型不匹配 | 减小 k、对齐温度、蒸馏草稿头 |
| 图模式全 NONE 且未 eager | 图未真正 replay | 查捕获桶（本项目历史结论：对 27B decode 收益仅 ~0.5%，优先级低） |

---

### 6.7 最深一层：stat logger 插件（`PROBE_PLUGIN=1`）

前面所有手段都是"从外面看"：日志是给人读的文本、`/metrics` 是聚合降采样后的数字。
而 vLLM 引擎内部本来就有 `SchedulerStats` / `IterationStats` 对象，里面带着每一步、
每个请求的**原始精确数值**。本工具箱通过 vLLM 官方插件机制
（entry point 组 `vllm.stat_logger_plugins`，基类 `StatLoggerBase`）把它们直接落成 JSONL：

```bash
PROBE_PLUGIN=1 make up-deep      # 启动时自动安装插件并开启（也支持 up / up-kernel）
# 或者写进 .env：PROBE_PLUGIN=1
python3 -m probe plugin --outdir runs/<时间戳>    # 汇总（probe_all.sh 会自动做）
```

拿到的字段（全部来自本机 0.18.1+das 的真实类定义，不是猜的）：

| 分组 | 内容 | 实现类 |
|---|---|---|
| `iter` | 逐 step 批次构成与耗时（ctx/gen 请求数与 token 数、`elapsed_ms`） | `SchedulerIterationDetails` |
| `sched` | running / waiting / skipped waiting / KV 使用率 / wave / step 计数 | `SchedulerStats` |
| `prefix` | 前缀缓存的 requests/queries/hits（含被抢占分量） | `PrefixCacheStats` |
| `spec` | drafts、草稿 token、接受 token、**按草稿位置的接受数组**、接受率与平均接受长度 | `SpecDecodingStats` |
| `evict` | KV block 驱逐事件（生命周期、驱逐前空闲、复用间隔） | `KVCacheEvictionEvent` |
| `graph` | 图调度：unpadded/padded token、补位次数、runtime mode | `CUDAGraphStat` |
| `perf` | 每 GPU 的 FLOPs / 读 / 写字节（算 MFU） | `PerfStats` |
| `reqs` | **每个完成请求的完整服务端生命周期**：finish_reason、`queue/prefill/decode/inference/e2e`、`mean_time_per_output_token`、prompt/生成/命中缓存 token、抢占次数 | `FinishedRequestStats` |
| `samples` | 本 step 观测到的 TTFT / ITL 样本（有界截断到 256） | `IterationStats` |

三个必须知道的点：

1. **插件是可选、可关闭的**：`PROBE_PLUGIN=0`（默认）不安装不加载；任何异常都被吞掉并计数，
   绝不影响推理主循环（`failures` 计数在 JSONL 旁边可查）。
2. **`iter` 分组依赖 `--enable-logging-iteration-details`**：vLLM 内部对
   `iteration_details` 做了门控（`enable_logging_iteration_details` **且** log stats 开启），
   所以建议搭配 `steps` 及以上档位；其余分组不受影响。
3. ⚠️ **不要把 `VLLM_PLUGINS` 设成只含本插件**。它是**全局插件白名单**，设了之后
   vLLM 自带的 `vllm.general_plugins`（如 LoRA filesystem / hf_hub resolver）就不再加载。
   留空（默认）即可让所有插件正常加载 —— 工具箱不会替你设置这个变量。

容量工具（跑过的真实校验）：插件装好后可用容器内自检脚本一键确认"能被 vLLM 加载器发现、
字段与当前版本一致"。设备参数直接用厂商 profile 生成（`--dry-run` 会完整打印），
因此任何厂商的镜像都适用：

```bash
# 先看某厂商的 docker 参数/环境脚本长什么样：
VENDOR=hygon scripts/start_vllm.sh --dry-run | head -40

# 通用写法（hygon 示例；其它厂商换设备参数与 prelude，其余一致）：
docker run --rm --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  --group-add video --security-opt label=disable \
  -v $PWD:/opt/vllm-probe -v /home/lyh/z100-vllm:/src:ro z100-s9-snap-s14 bash -lc '
    . /opt/dtk/env.sh >/dev/null 2>&1
    export PYTHONPATH=/src:/opt/vllm-probe PYTHONDONTWRITEBYTECODE=1
    rm -rf /tmp/plugin && cp -r /opt/vllm-probe/plugin_probe /tmp/plugin
    pip install -q --no-deps --no-build-isolation -e /tmp/plugin
    python3 /opt/vllm-probe/tests/plugin_container_selftest.py'
```

> 在 `/tmp` 里复制后再安装、并加 `PYTHONDONTWRITEBYTECODE=1`，是为了避免容器内的 root
> 在宿主机工作区留下 root 属主的 `*.egg-info` / `__pycache__`（实测踩过）。

### 6.8 看 CPU 时间花在哪个阶段（`VLLM_CUSTOM_SCOPES_FOR_PROFILING`）

kernel 档启动时会自动带上 `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`，vLLM 内部的
`record_function_or_nullcontext()` 就会把这些范围写进 torch trace：

```
schedule: allocate_slots / get_num_common_prefix_blocks / make_cached_request_data / update_after_schedule
gpu_model_runner: preprocess / forward / sample / postprocess / bookkeep / draft / eplb / ModelRunnerOutput
llm_engine step: get_output / process_outputs / record_stats / abort_requests
```

`scripts/trace_capture.sh` 跑完后的报告第 7 节会多出一张
**「Top CPU 侧范围 / 算子」**表，于是"CPU 时间到底花在调度还是采样还是后处理"
一眼可见（配合 >1ms 空洞数判断是否 CPU 侧拖累 GPU）。



## 7. 产物目录

每次运行（`start_vllm.sh` / `probe_all.sh`）都会新建 `runs/<时间戳>-<tag>/`：

| 文件 | 内容 |
|---|---|
| `run_meta.json` | 本次运行的镜像/模型/并行度/观测档位/完整启动命令 |
| `cmd.txt`、`args.txt` | 可复现的命令行（直接拷贝即可重跑） |
| `serve.log` | 引擎完整日志（逐 step 明细、stats、投机解码、MFU、图统计） |
| `env_check.json` | 观测开关自检结果 |
| `http_requests.jsonl` / `http_summary.json` | 逐请求明细 / 分位数汇总 |
| `metrics_before.json` `metrics_after.json` `metrics_diff.json` | 指标快照与差分（含 `.tsv`） |
| `steps.jsonl` / `steps_summary.json` | 逐 step 事件流 / 摘要 |
| `probe_steps.jsonl` / `probe_steps_summary.json` | stat logger 插件的逐 step 结构化记录 / 汇总（`PROBE_PLUGIN=1` 时） |
| `traces/` | torch profiler 原始 trace（`rank*.pt.trace.json.gz`）、`profiler_out_*.txt` |
| `trace_summary.json` / `key_averages.json` | 算子级归因结果 |
| `gpu_telem.jsonl` / `gpu_telem_raw.txt` | GPU 遥测采样（各厂商 SMI，统一 JSONL + 原始快照） |
| `suite_results.json` / `suite/<场景>/` | 标准化套件结果（可跨厂商对比）与逐场景原始请求数据 |
| `pyspy/` | py-spy 栈快照与火焰图 |
| **`REPORT.md`** | **中文报告（最终产物）** |

---

## 8. 可选观测栈（Prometheus + Grafana + OTel）

```bash
make compose-check      # 校验 compose 语法
make obs-up             # 起 prometheus(9090) + grafana(3000) + otel-collector(4317/4318)
# 浏览器打开 http://127.0.0.1:3000 ，看板「vLLM 推理全流程观测」已自动导入
make obs-down
```

看板按推理链路分成 6 组：请求生命周期（队列/预填充/解码/TTFT/ITL/TPOT 分位）、调度与队列
（`num_requests_waiting_by_reason` 分解）、KV 与前缀缓存、吞吐与 MFU、投机解码接受率、配置信息。
每个面板在描述里都标注了它需要哪个 vLLM 开关（例如 KV block 面板需要 `--kv-cache-metrics`）。

OpenTelemetry（可选，用于跨进程/跨服务链路追踪）：

```bash
scripts/start_vllm.sh   # 在 .env 里追加：
#   EXTRA_VLLM_ARGS=... --otlp-traces-endpoint http://otel-collector:4317
```

只要给了 `--otlp-traces-endpoint`，vLLM 就会为每个请求产出 `llm_request` span
（`vllm/v1/engine/output_processor.py::do_tracing`，本版本已确认存在），属性包括
`gen_ai.latency.time_to_first_token / e2e / time_in_queue / time_in_model_prefill /
time_in_model_decode`、`gen_ai.usage.prompt_tokens / completion_tokens`、`gen_ai.request.id`。
客户端没有拿到 trace id 的响应头，用响应头 **`X-Request-Id`** 与 span 的 `gen_ai.request.id`
对齐即可（vLLM 是「入站 traceparent」模式，不向外回传 trace id）。

> ⚠️ **不要指望 `--collect-detailed-traces`**：本版本（0.18.1+das）里它只做了参数校验，
> 两个消费者属性 `collect_model_forward_time` / `collect_model_execute_time` 在全树
> 没有任何读取方，实测 `/metrics` 中也不存在 `vllm:model_forward_time_milliseconds`。
> 想要 model forward / execute 的细分，请用第 6.4 节的 torch profiler。

---

## 9. 与现网实例共存 / 对已运行实例做探测

**默认姿态：另起一个探测实例（:8100），不要碰现网 :8000。**
27B/TP4 的显存占用很高，两个实例同时在 4 张卡上跑会互相挤兑，探测数据也会失真。

如果你确实想探测**已经在跑的**实例（例如 `z100-vllm`），有两种无重启方式：

```bash
# 方式 A：只读探测（完全不动服务），采集指标 + 解析现有日志
python3 -m probe metrics --port 8000 --outdir runs/live-$(date +%H%M%S)
python3 -m probe logs --file /mnt/game/dflash/serve-perf.log --summary runs/live/steps_summary.json

# 方式 B：用 probe_all 对现有实例做一轮（会发推理请求，注意会污染正在做的 A/B 实验）
RUN_DIR=runs/live bash scripts/probe_all.sh --port 8000 --skip-trace
```

如果要复用现网容器启动方式（`docker exec` 而不是新建容器）：

```bash
MODE=exec EXEC_CONTAINER=z100-vllm scripts/start_vllm.sh --level deep
scripts/stop_vllm.sh --kill-exec        # 结束（按进程名强杀，谨慎）
```

> 但请注意：`MODE=exec` 会**替换掉**容器里正在跑的那个实例，请确认没有正在进行的实验。

---

## 10. 常见问题

**Q1：日志里没有 `Iteration(...)` 明细 / 报告第 3 节是空的？**
需要 `steps` 及以上档位启动（会带 `--enable-logging-iteration-details`）。
用 `make env` 或 `python3 -m probe env` 自检，它会逐项告诉你哪些能力已生效。

**Q2：`POST /start_profile` 返回 404？**
profiler 端点只有在启动时带了 `--profiler-config '{"profiler":"torch",...}'` 才会注册。
用 `make up-kernel`（`PROBE_LEVEL=kernel`）启动即可。

**Q3：trace 文件在哪？多大？**
写在 `<RUN_DIR>/traces/`（容器内是 `/traces`），命名为 `rank<N>.<时间戳>.pt.trace.json.gz`。
实测本环境一个 rank 约 8.7 MB（解压后 131 MB），rank0 解析约 4 秒。只想压缩体积可以用
`torch_profiler_use_gzip`（默认已开）。

**Q4：`metrics` 显示的分位数带 `≤`？**
说明该直方图的桶阶梯太粗（见 §6.2）。以「均值」为准，或改用服务端逐请求指标。

**Q5：`py-spy` 报权限错误？**
容器需要 `--cap-add SYS_PTRACE`（`start_vllm.sh` 与 compose 已加）。若仍失败，检查宿主
`/proc/sys/kernel/yama/ptrace_scope`。

**Q6：显存没释放 / 卡被占用？**
```bash
scripts/stop_vllm.sh --rm       # 删掉容器
scripts/stop_vllm.sh --kill-exec  # 或者按进程名清理容器内残留进程，并等显存下降
make gpu-sample                 # 采样各卡 VRAM%（或直接用本机 SMI：hy-smi / nvidia-smi）
```

**Q7：`--enable-layerwise-nvtx-tracing` 会不会影响性能数字？**
会。它要求关掉图模式，只适合做"算子归因"，不要用它取性能基线。

**Q8：启动后日志里报 `ImportError: librocm_smi64.so.2: cannot open shared object file`？**
容器内必须先 source DTK 环境脚本才能 `import torch`。`start_vllm.sh` 已经在服务进程前自动执行
`. /opt/dtk/env.sh` + `. /src/tools/dcu/env.sh`（与现网 `s19_dflash_arm.sh` 一致），
并在之后固定 `PYTHONPATH=/src:/opt/vllm-probe:...`。若你的镜像里路径不同，改 `.env` 里的
`DTK_ENV_FILE` / `DCU_ENV_FILE` / `PYTHONPATH_IN_CONTAINER` 即可。
（注意：`docker exec` 的会话默认没有这套环境，所以手动在容器里跑 `python3 -c "import vllm"` 也会报同样的错，这不代表服务起不来。）

**Q9：`PROBE_PLUGIN=1` 开了，但 `probe_steps.jsonl` 是空的 / 报告里没有插件小节？**
按顺序查：① 容器内是否装上了插件（启动日志里会有 `stat logger 插件已安装(JSONL)`）；
② `VLLM_PROBE_LOG` 是否指向 `/logs/probe_steps.jsonl`；
③ `iter` 分组为空是正常的——它需要 `--enable-logging-iteration-details`（`steps` 及以上档位），
其余分组（sched/prefix/spec/graph/reqs）不受影响。用 `tests/plugin_container_selftest.py`
可以一键确认插件与当前 vLLM 版本是否兼容。

**Q10：能不能不构建镜像？**
可以。基础镜像里已经有 vLLM/torch/py-spy/opentelemetry，`probe/` 是纯标准库实现，
`start_vllm.sh` 会把项目目录挂到容器 `/opt/vllm-probe` 并设置 `PYTHONPATH`，因此**容器内外都能直接跑**。
`scripts/build_image.sh` 只是为了做一个"自带工具包"的镜像，便于分发。

---

## 11. 已验证环境与自测

```bash
make test          # = bash tests/run_tests.sh
```

自测包含三层，全部可在**无 GPU、无 Docker** 的机器上跑：

1. **单元测试**：真实日志样例（`samples/serve.log`）、真实 `/metrics` 抓取（`samples/metrics.txt`）、
   合成 Chrome trace、mock 服务下的请求级打点、报告生成与自动归因、
   **文档/看板 ↔ 真实指标名一致性守卫**（防止引用不存在的指标导致面板静默空白）、
   插件记录抽取（鸭子类型假对象，验证"字段改名只丢该项、不抛异常"）、
   **厂商注册表一致性守卫**（Shell `vendors/*.sh` ↔ Python `probe/vendors.py`）、
   **五家 SMI 输出解析**（`samples/smi/` 样例文本）、标准化套件 schema。
2. **脚本语法检查**：全部 `scripts/*.sh`、`vendors/*.sh` 过 `bash -n`。
3. **mock 端到端**：起一个 mock vLLM（含 `/metrics`、SSE 流式、`/start_profile`），
   跑通 `env → vendor → metrics → http → telemetry → diff → trace_capture → logs → plugin → suite → compare → probe_all → REPORT.md` 全链路。

插件还有一层**只在容器内可跑**的自检（验证 entry point 能被 vLLM 加载器发现、
字段与当前版本一致，见 §6.7 的命令）——本机已实测 18/18 项通过。

本工具箱开发时用以下**真实数据**验证过（不是模拟数据）：

* 现网 `serve-perf.log`（769 个 iteration、46 条 stats、42 条投机解码、61 行图统计）→ step 摘要与 `ms/token` 数值自洽；
* 现网 `/metrics`（104 个指标族、577 条序列）→ 阶段表、KV block 指标、前缀缓存命中率；文档/看板引用的 37+32 个指标名全部实测存在；
* 现网 torch trace（`rank0…pt.trace.json.gz`，131 MB 解压 / 462,041 事件）→ GEMM 67.6%、通信 22.1%、GPU 利用率 84.3%、>1ms 空洞 2 个；
* 现网 `profiler_out_0.txt` key-averages 表 → Self CUDA 合计 20.9 s、Top 算子与 trace 结果一致；
* 容器内 `vllm serve --help` 全量 flag 校验（生成的启动命令 `exit=0`、无 unrecognized 参数）；
* 容器内对真实 `SchedulerStats`/`IterationStats` 等类做 introspection，插件字段名据此确定。

---

## 12. 注意事项

1. **探测本身有开销**：`steps` 档写日志、`deep` 档采样指标、`kernel` 档开 profiler 都会影响性能。
   取"性能基线"时用 `PROBE_LEVEL=off|basic`；要"归因"时再用高挡位，并在报告里注明档位。
2. **`/src` 是只读挂载（可选）**：`SRC_DIR` 未设置时不挂载、直接用镜像里自带的 vLLM；
   设置后探测不会修改 vLLM 源码，镜像里的 vLLM 与 checkout 可能不同步，
   报告里的 `run_meta.json` 记录了实际使用的镜像与命令。
3. **一次只跑一个实例**：显存有限，探测实例与现网实例不要同时驻留在同一组卡上。
4. **报告阈值是可调的**：默认阈值在 `probe/report.py::DEFAULT_THRESHOLDS`
   （来自本项目历史结论，例如"图模式收益仅 ~0.5%"），厂商专属覆盖在
   `probe/vendors.py::VendorProfile.diagnose_overrides`；换模型/换卡后请同步调整。
5. **日志量**：`steps` 档约 5 行/秒；长时间高频压测建议配合 `watch_steps.sh --tail` 做分段汇总。
