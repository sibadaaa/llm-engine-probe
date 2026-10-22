# 物理机（host）启动模板

一套**不要求改核心脚本**的物理机启动模板：每个模板是一份 `.env` 片段，按顺序叠在本机 `.env`
之上，再由现有 `scripts/start_vllm.sh --host` 启动。Docker 模式用
[`docker/docker-compose.<厂商>.yml`](../docker/) 选择厂商；物理机模式用这里的
`vendor/<厂商>.env` 对齐同一件事。

> 2026-10-21 实测补了一处 host 模式兼容修复：`serve_launch.sh` 在 source 厂商 prelude
> （DTK 的 `env.sh` 等）期间临时 `set +u`，避免 `set -u` 下实例启动即退出。模板本身仍不要求改核心。

> 为什么独立成一层：物理机换厂商/换模型/换并行度时，真正变的是十来个 `.env` 变量。
> 模板把这十来个值固化成可复用、可组合、可审查的文件，同时**不侵入**
> `scripts/start_vllm.sh` 与 `scripts/lib/common.sh`（它们仍只认 `.env` + 显式环境变量）。

---

## 1. 快速开始

```bash
# ① 列出全部模板
hosts/vllm-host.sh list

# ② 看叠加后的生效配置（不启动、不写运行目录）
hosts/vllm-host.sh show vendor/hygon scenario/tp4-prod spec/dflash

# ③ 预检（等价 make host-check，但带上模板变量）
hosts/vllm-host.sh check vendor/hygon scenario/tp4-prod

# ④ 先看将执行的命令（不启动）
hosts/vllm-host.sh dry-run vendor/hygon scenario/tp4-prod spec/dflash

# ⑤ 启动 → 探测 → 报告 → 停止
hosts/vllm-host.sh up    vendor/hygon scenario/tp4-prod spec/dflash
hosts/vllm-host.sh status vendor/hygon scenario/tp4-prod
hosts/vllm-host.sh probe  vendor/hygon scenario/tp4-prod
hosts/vllm-host.sh down   vendor/hygon scenario/tp4-prod
```

模板可以写短名（`hygon`）或全名（`vendor/hygon`）。短名在 `hosts/*/` 下唯一时可用，
有歧义会报错并列出候选。

底层脚本参数放在 `--` 之后透传：

```bash
hosts/vllm-host.sh up    vendor/hygon -- --skip-check
hosts/vllm-host.sh check vendor/hygon -- --json --skip-gpu --skip-vllm-import
hosts/vllm-host.sh up-kernel vendor/hygon scenario/tp4-prod   # kernel 档
```

`HOST_TEMPLATES` 可提供默认模板链（空格或逗号分隔），适合 systemd/脚本固化：

```bash
HOST_TEMPLATES="vendor/hygon scenario/tp4-prod spec/dflash" hosts/vllm-host.sh up
```

---

## 2. 叠加规则（重要）

优先级从高到低：

1. **命令行显式导出的变量**：`TP=2 hosts/vllm-host.sh up ...`（最高，模板不会覆盖它）；
2. **模板链**：按书写顺序，后面的覆盖前面的；
3. **仓库根 `.env`**：本机基线，提供 `HOST_PYTHON`、模型盘路径、缓存目录等机器相关值。

因此推荐顺序是 **厂商 → 场景 → 量化/投机解码 → 部署形态**：

```bash
hosts/vllm-host.sh up vendor/hygon scenario/tp4-prod spec/dflash
# 显式改并行度：模板里的 TP=4 会被 2 覆盖，其它值不变
TP=2 hosts/vllm-host.sh up vendor/hygon scenario/tp4-prod
```

`hosts/vllm-host.sh show` 会打印模板链与最终生效值；`dry-run` 会打印完整
`serve_launch.sh`（含 prelude、环境变量、vLLM 命令行），用于启动前复核。

> **追加型键的注意点**：`EXTRA_VLLM_ARGS` / `EXTRA_ENV` 的「末尾追加」型模板
> （`quant/*`、`spec/mtp`、`deploy/multinode-worker`）在你**显式导出**同名变量时不生效——
> 显式值优先级最高，模板追加被整体丢弃（`spec/mtp` 意味着投机解码静默不生效）。
> `hosts/vllm-host.sh show` / `dry-run` / `up` 会对此给出 ⚠️ 提示；如需追加，
> 请把对应参数合并进显式值。

> 模板只改「配置变量」，不会改 `start_vllm.sh` 的观测档位逻辑：
> `PROBE_LEVEL` 仍可按 `LEVEL=deep`/`up-kernel` 覆盖。

---

## 3. 模板目录与索引

```
hosts/
├── vllm-host.sh                 # 统一入口（list/show/check/dry-run/up/...）
├── lib/env_overlay.sh           # 叠加与校验实现（只被入口 source）
├── _template.env                # 新增模板的骨架（含编写约定）
├── vendor/*.env                 # 厂商：VENDOR + prelude + 可见卡默认值
├── scenario/*.env               # 场景：模型/并行/长度/吞吐/端口
├── quant/*.env                  # 量化：FP8 / INT8 W8A8 / W4A16 / AWQ
├── spec/*.env                   # 投机解码：dflash / mtp / eagle / ngram
├── deploy/                      # 部署形态：systemd / 外部实例 / PD / 多机（实验性）
└── engine/                      # 引擎组件：ENGINE 切换 + 引擎语义的模型/并行参数
```

### 3.1 厂商模板（`vendor/`）

| 模板 | 用途 | 关键设置 |
|---|---|---|
| [`vendor/nvidia.env`](vendor/nvidia.env) | NVIDIA CUDA 物理机 | `VENDOR=nvidia`；生成 `CUDA_VISIBLE_DEVICES`；可选 prelude |
| [`vendor/amd.env`](vendor/amd.env) | AMD ROCm 物理机 | `VENDOR=amd`；生成 `HIP_VISIBLE_DEVICES`；ROCm 环境脚本 |
| [`vendor/hygon.env`](vendor/hygon.env) | 海光 DCU（本站现网） | `VENDOR=hygon`、DTK prelude（存在才注入）、`HSA_FORCE_FINE_GRAIN_PCIE` |
| [`vendor/intel.env`](vendor/intel.env) | Intel GPU（XPU） | `VENDOR=intel`；生成 `ZE_AFFINITY_MASK`；oneAPI 环境脚本 |
| [`vendor/ascend.env`](vendor/ascend.env) | 昇腾 NPU（vllm-ascend） | `VENDOR=ascend`；生成 `ASCEND_RT_VISIBLE_DEVICES`；CANN `set_env.sh` |

厂商模板不设置 `HOST_PYTHON`（每台机器的 venv/conda 不同），只给出注释示例。
也**不直接设置卡号**（`VISIBLE_DEVICES`）：卡号沿用 `.env` 或命令行显式值，
厂商 profile（`vendors/<id>.sh` 的 `VENDOR_DEVICE_VIS_ENV`）再据此生成对应的可见性环境变量。

### 3.2 场景模板（`scenario/`）

| 模板 | 用途 | 默认 |
|---|---|---|
| [`scenario/single-gpu-dev.env`](scenario/single-gpu-dev.env) | 单卡冒烟/调试 | `Qwen3-0.6B`、TP=1、4K 上下文、`PORT=8101` |
| [`scenario/tp4-prod.env`](scenario/tp4-prod.env) | 对齐现网生产臂 | 27B FP8、TP=4、32K、eager、`PORT=8100` |
| [`scenario/tp8-throughput.env`](scenario/tp8-throughput.env) | 8 卡高吞吐 | TP=8、batched-tokens 32K、`MAX_NUM_SEQS=256`、`PORT=8102` |
| [`scenario/long-context-128k.env`](scenario/long-context-128k.env) | 长上下文 | 27B FP8、TP=4、`MAX_MODEL_LEN=131072`、`PORT=8103` |

场景模板会**重置**投机解码与 `EXTRA_VLLM_ARGS`，保证单独使用时行为可预测；
要投机解码请在其后再叠 `spec/*.env`。

### 3.3 量化模板（`quant/`）

| 模板 | 用途 | 说明 |
|---|---|---|
| [`quant/fp8.env`](quant/fp8.env) | FP8 权重（27B-FP8） | vLLM 自动识别；显式 `--quantization fp8` 便于复核 |
| [`quant/int8-w8a8.env`](quant/int8-w8a8.env) | INT8 W8A8（Channel-INT8） | `--quantization compressed-tensors`，`DTYPE=float16` |
| [`quant/w4a16.env`](quant/w4a16.env) | W4A16 权重（TinyLlama 样例） | compressed-tensors 4bit group128 |
| [`quant/awq.env`](quant/awq.env) | AWQ 权重（需自备） | `--quantization awq`，模型路径为占位 |

量化模板只改模型/精度/量化参数，并行度仍由场景模板决定。

### 3.4 投机解码模板（`spec/`）

| 模板 | 用途 | 说明 |
|---|---|---|
| [`spec/dflash.env`](spec/dflash.env) | DFlash2 草稿（现网） | `method=dflash`、k=7、草稿 `bfloat16` |
| [`spec/mtp.env`](spec/mtp.env) | 目标模型自带 MTP 层 | 通过 `--speculative-config` 透传，不加载独立草稿模型 |
| [`spec/eagle.env`](spec/eagle.env) | EAGLE/EAGLE3 草稿 | 草稿路径为占位，需换成本机权重 |
| [`spec/ngram.env`](spec/ngram.env) | 无草稿模型的 n-gram 投机 | `DRAFT_MODEL=[ngram]`，只吃 prompt 重复 |

> **MTP 的特殊之处**：`start_vllm.sh` 只有在 `DRAFT_MODEL` 非空时才生成
> `--speculative-config`。MTP 的草稿权重就在目标 checkpoint 里（现网 27B-FP8 含
> `mtp.*`），所以 `spec/mtp.env` 把 `DRAFT_MODEL` 清空，改用
> `EXTRA_VLLM_ARGS` 直接透传 JSON。副作用：`deep` 档的逐请求投机解码指标不会开启
> （`SPEC_DECODE_ON=0`），需要那些指标时请用 dflash/eagle 草稿模型。

### 3.5 部署形态（`deploy/`，实验性）

| 文件 | 用途 |
|---|---|
| [`deploy/systemd/vllm-host-probe.service`](deploy/systemd/vllm-host-probe.service) | systemd 常驻示例（oneshot + RemainAfterExit + 限制项） |
| [`deploy/systemd/README.md`](deploy/systemd/README.md) | 安装/启用/排错 |
| [`deploy/external-probe.sh`](deploy/external-probe.sh) | 只读探测**别人启动的**实例（systemd/手工） |
| [`deploy/pd-prefill.env`](deploy/pd-prefill.env) / [`pd-decode.env`](deploy/pd-decode.env) / [`pd-run.sh`](deploy/pd-run.sh) | PD 分离骨架（prefill/decode 两个实例） |
| [`deploy/pd-README.md`](deploy/pd-README.md) | PD 前提、KV connector、启动顺序 |
| [`deploy/multinode-head.env`](deploy/multinode-head.env) / [`multinode-worker.env`](deploy/multinode-worker.env) / [`multinode-run.sh`](deploy/multinode-run.sh) | 多机 TP 骨架（head/worker） |
| [`deploy/multinode-README.md`](deploy/multinode-README.md) | 多机前提、`--nnodes/--node-rank`、版本检查 |

PD/多机模板**默认拒绝直接启动**：脚本先检查必需变量（KV connector、节点数、head 地址），
并把最终命令打印出来；确认无误后再按 README 打开闸门。原因见各 README 的「为什么默认不启动」。
deploy 模板**保留上游** `EXTRA_VLLM_ARGS`（worker 只追加 `--headless`），不会重置场景/厂商参数。

### 3.8 引擎模板（`engine/`）

引擎模板切换 `ENGINE`（引擎组件 = 提供服务进程的软件，与厂商 `VENDOR` 正交；注册表在
`engines/<id>.sh` ↔ `probe/engines.py`，契约见
[`docs/design/component-interface/设计.md`](../docs/design/component-interface/设计.md) §4/§9）。

| 模板 | 用途 | 关键设置 |
|---|---|---|
| [`engine/vllm.env`](engine/vllm.env) | 显式声明 vLLM（占位语义，与不设置等价） | `ENGINE=vllm` |
| [`engine/strata.env`](engine/strata.env) | 切换到 Strata（MoE 专家分层推理引擎，结构适配） | `ENGINE=strata`、`MODEL` 换成 pack 名、`HOST_ENGINE_PATTERN`（真机核对项见 `docs/design/strata-engine/施工图.md` §4） |
| [`engine/sglang.env`](engine/sglang.env) | 切换到 SGLang（高性能 OpenAI 兼容推理引擎，结构适配） | `ENGINE=sglang`、`MODEL` 身份记录、推荐入口含 `--enable-metrics`（默认关）、`HOST_SERVE_ENTRYPOINT` 覆盖拉起命令 |
| [`engine/llamacpp.env`](engine/llamacpp.env) | 切换到 llama.cpp（llama-server GGUF 单节点推理，结构适配） | `ENGINE=llamacpp`、`MODEL` 身份记录、推荐入口含 `--metrics`（默认关）、无跨卡 TP、`HOST_SERVE_ENTRYPOINT` 覆盖拉起命令 |
| [`engine/ollama.env`](engine/ollama.env) | 切换到 Ollama（本地模型管理/推理服务，OpenAI 兼容层探测） | `ENGINE=ollama`、`MODEL`/`SERVED_NAME` 用 tag 名（需先 `ollama pull`）、`EXTRA_ENV` 传 `OLLAMA_HOST` 对齐 `PORT`、无指标面 |

**链序约定**：引擎模板放最后——`vendor → scenario → quant/spec（vLLM 专属）→ engine`。
引擎模板负责把模型/并行参数换成该引擎的语义（如 Strata 的 `MODEL` 是 pack 名不是权重路径），
因此要能覆盖前面模板的 vLLM 语义取值：

```bash
hosts/vllm-host.sh up engine/strata                     # 单模板
hosts/vllm-host.sh up vendor/nvidia engine/strata       # 显式指定厂商做遥测
```

> 注意：非 vLLM 引擎在 `RUNTIME=docker` 下会被启动链显式拒绝（本仓库 `docker/` 暂不做
> 非 vLLM 引擎适配——多数引擎其实有官方镜像，拒绝属 `docs/design/README.md` §5 红线 3
> 的仓库策略，非引擎能力缺失）；外部已运行的引擎实例请用
> `deploy/external-probe.sh` 只读探测。Strata 的完整使用说明（外部只读/host 拉起、
> 压测语义差异、排障）见 [`../docs/strata-接入指南.md`](../docs/strata-接入指南.md)。

---

## 4. 与 Docker 模式的对照

| Docker 里的做法 | 物理机模板的等价做法 |
|---|---|
| `docker-compose.nvidia.yml` / `.amd.yml` / `.hygon.yml` / `.intel.yml` / `.ascend.yml` | `vendor/<厂商>.env` |
| `IMAGE` 决定 vLLM 版本 | `HOST_PYTHON` 所在环境决定版本（`.env` 配置） |
| `-e CUDA_VISIBLE_DEVICES=...` 等 | 厂商模板设 `VENDOR`；可见性变量由 `vendors/<id>.sh` 按 `VISIBLE_DEVICES` 生成 |
| `-v <模型盘>:<容器路径>` | 模板直接写宿主机 `MODEL` 路径 |
| `--ulimit` / `--shm-size` | 宿主机 ulimit / `/dev/shm`；systemd 模板里写 `LimitMEMLOCK` 等 |
| `docker compose --profile obs` | 不适用；外部 Prometheus 抓宿主机 `:PORT/metrics` |

---

## 5. 新增一个模板

1. `cp hosts/_template.env hosts/<类别>/<名字>.env`；
2. 按 [`_template.env`](_template.env) 的注释填写：文件头三行（用途/适用/组合）、覆盖键、风险；
3. 只使用 `.env.example` / `scripts/lib/common.sh` 里已有的键；新 vLLM 参数用
   `EXTRA_VLLM_ARGS`（按空白切分，**值里不能有空格**）或 `EXTRA_ARGS_FILE`（每行一个参数，
   JSON 用它）；
4. 跑 `bash -n hosts/<类别>/<名字>.env`、`hosts/vllm-host.sh show <名字>`、
   `hosts/vllm-host.sh dry-run <名字>`；
5. 在 [`hosts/README.md`](README.md) 的索引表加一行。

**引擎模板（`engine/`）特殊约定**：引擎模板负责把模型/并行参数换成引擎语义，放链序最后；
新引擎的接入三步见 `docs/design/README.md` §4（profile + 模板 + 两行索引）。

模板文件是 shell 片段，会被 `source`。**不要**在模板里执行命令、`cd`、写文件或改
`HOST_PYTHON`/`RUNS_DIR` 这类机器相关路径——那会让 `show`/`dry-run` 产生副作用。

---

## 6. 校验与自测

```bash
bash tests/host_templates_selftest.sh   # 模板语法/键名/叠加/ dry-run 全量自测（无需 GPU/vLLM）
bash tests/run_tests.sh                 # 全量测试（会包含上面的模板自测）
```

自测覆盖：

* 每个 `hosts/*/*.env` 都能 `bash -n`，且键名属于 `common.sh` / `.env.example` 的已知集合；
* `list` / `show` / 短名解析 / 歧义报错；
* 显式环境 > 模板链 > `.env` 的优先级；
* 每个模板单独 `dry-run` 能生成 `serve_launch.sh`，且关键参数（TP、模型、投机解码、
  量化、可见卡）出现在命令行里；
* `scripts/` 无模板引入的额外改动（自测白名单当前放行：`start_vllm.sh` 的 host prelude
  `set +u` 兼容修复与引擎 caps 门禁收拢、`lib/common.sh` 的引擎组件层与就绪等待共享函数、
  `stop_vllm.sh` / `status.sh` / `pyspy.sh` / `wait_ready.sh` 的引擎感知改动
  （见 `docs/design/component-interface/施工图.md` 与 F3 评审修复）；
  其它 `scripts/` 改动会判失败，扩展白名单须同步更新此处描述）。

---

## 7. 已知限制

* 模板里的模型路径按本站点 `/mnt/game/models/<模型>` 写；换机器请改模板或
  `MODEL=/你的路径 hosts/vllm-host.sh ...`。`check` 会报模型/驱动问题，`dry-run` 不会。
* 模板不改 `.env`，也不建立「模板注册表」；可用模板 = `hosts/*/*.env` 的文件集合。
* 模板只适用于 `RUNTIME=host`；Docker 模式请继续用 `docker/docker-compose.*.yml`。
* 同一端口同时只能有一个实例；场景模板给了 8101/8102/8103 便于并存调试，但仍受显存约束。
* PD/多机模板是**骨架**，需要按你的 KV connector、网卡、vLLM 版本补齐；不要直接上生产。
* 引擎模板（`engine/`）里的非 vLLM 条目（如 `engine/strata.env`）是**结构适配**：
  进程正则、版本命令、就绪指标等取值为占位，真机核对项见
  [`docs/design/strata-engine/施工图.md`](../docs/design/strata-engine/施工图.md) §4。
