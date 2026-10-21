# PD 分离（prefill / decode）实验性骨架

> **默认拒绝启动。** [`pd-run.sh`](pd-run.sh) 不带闸门变量时只打印
> 「需要补齐的变量 + KV connector 要求 + 两个实例的命令预览」并退出 2；
> 只有 `PD_ENABLE=1` **且** `KV_CONNECTOR` 非空才继续。
> 这是骨架，不是生产方案；先按本文确认前提，再打开闸门。

PD（Prefill/Decode Disaggregation）把一次请求拆成两段：

```text
客户端 ──> PD 代理/路由 ──> prefill 实例（算 prompt KV，kv_producer）
                        └─> decode  实例（读 KV，逐 token 生成，kv_consumer）
```

两个实例各自只提供**半套**能力：prefill 不长期保存解码状态，decode 不做
长 prompt 预填充。**没有代理/路由时，客户端直接打任一实例都不等于 PD**。

---

## 1. 文件

| 文件 | 作用 |
|---|---|
| [`pd-prefill.env`](pd-prefill.env) | prefill 实例的模型/并行/端口/观测档位 |
| [`pd-decode.env`](pd-decode.env) | decode 实例的模型/并行/端口/观测档位 |
| [`pd-run.sh`](pd-run.sh) | 闸门、参数校验、生成 `--kv-transfer-config`、依次调用 `hosts/vllm-host.sh up` |

内部变量（`PD_ENABLE`、`KV_CONNECTOR`、`PREFILL_PORT`、`DECODE_PORT`、
`PROXY_ADDR`、运行目录）只在 `pd-run.sh` 里，**不要写进 `.env` 模板**。
两个 `.env` 只使用 `.env.example` / `scripts/lib/common.sh` 已有的键。

默认端口：

| 角色 | 端口 | 说明 |
|---|---|---|
| prefill | `8200` | `PREFILL_PORT` 可覆盖 |
| decode | `8201` | `DECODE_PORT` 可覆盖 |
| PD 代理 | 由你决定（示例 `8100`/`8000`） | 本仓库**不启动**代理 |

---

## 2. 前提（缺一不可）

### 2.1 vLLM 版本与 KV connector

* vLLM 必须支持 `--kv-transfer-config`，且你的版本内置了要用的 connector；
  先自查：

  ```bash
  ${HOST_PYTHON:-python3} -c \
    'from vllm.entrypoints.cli.main import main; main()' serve --help \
    | grep -E 'kv-transfer|kv_connector'
  ```

* `KV_CONNECTOR` 必须是该版本认识的类名，常见写法：
  `NixlConnector` / `P2pConnector` / `MooncakeConnector` 等
  （具体以 `vllm serve --help` 与 vLLM 源码中的 connector 注册表为准）。
* **connector 的额外字段本脚本不猜**。`pd-run.sh` 只生成最小配置：

  ```json
  {"kv_connector":"<KV_CONNECTOR>","kv_role":"kv_producer"}   # prefill
  {"kv_connector":"<KV_CONNECTOR>","kv_role":"kv_consumer"}   # decode
  ```

  connector 需要的 `kv_ip`/`kv_port`/`kv_buffer_device`/backend/extra_config、
  或 NIXL 的 side-channel 环境变量，请用 `EXTRA_VLLM_ARGS` / `EXTRA_ENV`
  显式传入；已经在 `EXTRA_VLLM_ARGS` 或 `EXTRA_ARGS_FILE` 里写了
  `--kv-transfer-config` 时，`pd-run.sh` 不再自动追加，`kv_role` 由你保证。

### 2.2 网卡 / 互联

* prefill 与 decode 之间必须能建立 KV 传输通道。常见要求：
  * 同机：回环 + 共享内存/本地 TCP；注意两实例的显存与端口不能冲突；
  * 跨机：IB/RDMA（推荐）或 connector 支持的 TCP；需要驱动/固件/UCX/NIXL 版本匹配；
  * 防火墙放行 connector 的 side-channel 端口与 `KV_CONNECTOR` 配置里的
    `kv_ip:kv_port`。
* **厂商支持要单独确认**：本站点的海光 DCU（ROCm/HIP）不能假设 CUDA 的
  NIXL/IB 方案可以直接照搬；先确认 connector 在该后端有实现，再做同机小规模验证。

### 2.3 两个实例的一致性

* 同一个 `MODEL`（每个节点上的路径都必须存在；NFS 也行）；
* `SERVED_NAME` 一致（本骨架默认都是 `qwen3.8-27b`）；
* `MAX_MODEL_LEN`、`TP`、KV cache dtype、tokenizer 一致；TP 不一致时
  KV layout 往往对不上，骨架默认两边都 TP=4；
* `DRAFT_MODEL`/`SPEC_*` 已在模板里清空：PD 骨架不叠 `spec/*.env`。

### 2.4 代理 / 压测客户端

* 需要 PD 代理/路由把 prompt 发给 prefill、把生成阶段发给 decode；
  可用 vLLM 仓库的 disagg 示例代理、`vllm-router` 或自建；
* 探针/套件是按**单端口**设计的：`probe_all.sh`/`run_suite.sh` 一次只打一个
  `--port`。PD 的端到端测试要把代理端口作为目标，并分别对两个实例做探针；
* 本仓库不会自动聚合 PD 两段的报告，也不会替你启动代理。

---

## 3. 先 dry-run（默认行为）

```bash
cd /path/to/llm-engine-probe

# ① 单独看两个模板的叠加结果（不启动）
hosts/vllm-host.sh show deploy/pd-prefill
hosts/vllm-host.sh show deploy/pd-decode

# ② 看 pd-run.sh 的完整预览；不带闸门 → 打印命令并退出 2
hosts/deploy/pd-run.sh
echo "退出码：$?"     # 2

# ③ 只补 PD_ENABLE、不补 KV_CONNECTOR → 仍然退出 2，并提示缺 KV_CONNECTOR
PD_ENABLE=1 hosts/deploy/pd-run.sh

# ④ 想看真实 connector 的最终命令，但仍不启动：只给 KV_CONNECTOR，不给 PD_ENABLE
KV_CONNECTOR=NixlConnector hosts/deploy/pd-run.sh
```

预览里会给出两条最终命令（`PORT=... EXTRA_VLLM_ARGS=... hosts/vllm-host.sh up ...`）
以及自动生成的 `--kv-transfer-config`，确认无误后再进入下一步。

---

## 4. 打开闸门

```bash
# 最小启用（connector 额外参数用 EXTRA_ENV 补）
PD_ENABLE=1 KV_CONNECTOR=NixlConnector \
  EXTRA_ENV="VLLM_NIXL_SIDE_CHANNEL_HOST=10.0.0.1 VLLM_NIXL_SIDE_CHANNEL_PORT=5600" \
  hosts/deploy/pd-run.sh
```

`pd-run.sh` 会：

1. 校验 `KV_CONNECTOR` 字符、两个端口、`EXTRA_ARGS_FILE`；
2. 依次执行（**只调用入口脚本，不绕过模板层**）：

   ```bash
   PORT=8200 EXTRA_VLLM_ARGS='... --kv-transfer-config {...kv_producer...}' \
     hosts/vllm-host.sh up deploy/pd-prefill -- --run-dir runs/pd-prefill
   PORT=8201 EXTRA_VLLM_ARGS='... --kv-transfer-config {...kv_consumer...}' \
     hosts/vllm-host.sh up deploy/pd-decode  -- --run-dir runs/pd-decode
   ```

3. 两个实例各自用固定运行目录，便于分别 `down`（`down` 默认只取最新目录，
   PD 必须显式给 `--run-dir`）：

   ```bash
   hosts/vllm-host.sh down deploy/pd-prefill -- --run-dir runs/pd-prefill
   hosts/vllm-host.sh down deploy/pd-decode  -- --run-dir runs/pd-decode
   ```

> 同机跑两个实例会抢显存：模板默认都写 `VISIBLE_DEVICES=0,1,2,3`。
> 同机调试请把 `pd-decode.env` 改成 `4,5,6,7`，或分成两台机器各用 `0-3`。
> 两台机器各自有仓库和 `.env`，`HOST_PYTHON`/模型路径/缓存目录要各自配好。

---

## 5. 启动后怎么验证

```bash
# 分别看两个实例（各自是独立进程/运行目录）
PORT=8200 hosts/vllm-host.sh status deploy/pd-prefill
PORT=8201 hosts/vllm-host.sh status deploy/pd-decode

# 分别做单实例探针（能验证实例本身，但不等于 PD 端到端）
PORT=8200 hosts/vllm-host.sh probe deploy/pd-prefill
PORT=8201 hosts/vllm-host.sh probe deploy/pd-decode

# 看 KV 传输是否真的建立：两边 serve.log 里应有 connector 初始化/握手/传输记录
grep -nE 'kv[-_ ]?transfer|kv_connector|NIXL|producer|consumer' \
  runs/pd-prefill/serve.log runs/pd-decode/serve.log

# 端到端：客户端必须打你的代理（示例）
PROBE_ENDPOINT=chat PORT=<代理端口> scripts/probe_all.sh --runtime host --port <代理端口>
```

只看到两个 `/health` 200 **不代表 PD 成功**：如果 `--kv-transfer-config` 没生效，
两个实例会各自独立推理，客户端却以为在做 PD。务必用代理发真实请求，并确认
prefill 的 KV 真的被 decode 读到（日志/connector 指标/时延形态）。

---

## 6. 排错

| 现象 | 先查 |
|---|---|
| `pd-run.sh` 退出 2 | 预期：没设 `PD_ENABLE=1` 或 `KV_CONNECTOR` 为空；看预览补齐 |
| 启动时报 `unrecognized arguments: --kv-transfer-config` | vLLM 版本不支持；先按 §2.1 自查版本 |
| connector 初始化失败/一直重试 | `kv_ip`/`kv_port`/side-channel 端口/网卡/IP 不通；防火墙；RDMA/UCX 驱动 |
| 两个实例都 `/health` 200，但请求异常 | 没接代理，或代理没把 prefill 的 KV 交给 decode；看 §5 |
| decode OOM / KV 不足 | decode 的 `GPU_MEM_UTIL`、`MAX_MODEL_LEN`、KV dtype 与 prefill 不一致 |
| `down` 停错实例 | PD 必须给 `--run-dir`（`down` 默认取最新目录） |
| 同机两实例抢卡 | 改 `VISIBLE_DEVICES` 或拆到两台机器 |

---

## 7. 仍然缺的真实环境信息

本骨架刻意不猜下面这些；补齐前不要上生产：

1. **vLLM 版本 + connector 名称/实现**：`--kv-transfer-config` 是否可用、
   `NixlConnector`/`P2pConnector`/`MooncakeConnector` 在你的版本里是否注册、
   `kv_role` 语义是否就是 producer/consumer。
2. **connector 的完整 JSON/环境变量**：`kv_ip`、`kv_port`、`kv_buffer_device`、
   `kv_connector_extra_config`、NIXL side-channel host/port 等；本脚本只生成最小两项。
3. **网络事实**：prefill/decode 是同机还是跨机；IB/RDMA 设备名、IP/网段、
   UCX/NIXL 版本、防火墙规则；海光 DCU 后端是否支持所选 connector。
4. **代理/路由实现与端口**：用 vLLM disagg 示例、`vllm-router` 还是自建；
   它的鉴权/超时/重试策略；压测客户端最终打哪个地址。
5. **两实例的资源规划**：每台机器的卡数/`TP`/`GPU_MEM_UTIL`/`MAX_MODEL_LEN`/
   KV dtype；同机时 `VISIBLE_DEVICES` 怎么分。
6. **观测聚合**：两个实例各有一份 `serve.log`/`REPORT.md`，本仓库暂不生成
   “PD 端到端”报告；需要你在代理层另外加指标/压测。
