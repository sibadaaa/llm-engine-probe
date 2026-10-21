# 多机 TP（vLLM `--nnodes`）实验性骨架

> **默认拒绝启动。** [`multinode-run.sh`](multinode-run.sh) 不带闸门变量时只打印
> 「需要补齐的变量 + 节点参数 + 命令预览」并退出 2；只有 `MULTINODE_ENABLE=1`
> **且** `HEAD_ADDR` 非空才继续，并在启动前用 `HOST_PYTHON` 检查
> `vllm serve --help` 是否真的支持 `--nnodes`。

这不是 PD（prefill/decode 分离）。多机 TP 是**一个模型实例跨多台机器做张量并行**：
每台机器跑一个 vLLM 进程，rank 0（head）提供 OpenAI API，其它 rank（worker）以
`--headless` 加入同一个分布式实例。

---

## 1. 文件

| 文件 | 作用 |
|---|---|
| [`multinode-head.env`](multinode-head.env) | head（rank 0）的模型/并行/端口/观测档位 |
| [`multinode-worker.env`](multinode-worker.env) | worker（rank>=1）的配置；`EXTRA_VLLM_ARGS="--headless"` |
| [`multinode-run.sh`](multinode-run.sh) | 闸门、版本检查、生成节点参数、调用 `hosts/vllm-host.sh up` |

内部变量（`MULTINODE_ENABLE`、`NODES`、`NODE_RANK`、`HEAD_ADDR`、`HEAD_PORT`、
`MULTINODE_SKIP_VERSION_CHECK`）只在 `multinode-run.sh` 里，**不要写进 `.env` 模板**。
两个 `.env` 只使用 `.env.example` / `scripts/lib/common.sh` 已有的键。

默认端口：

| 用途 | 端口 | 说明 |
|---|---|---|
| head OpenAI API | `8300` | `multinode-head.env` 的 `PORT`；探针/客户端打这里 |
| head 分布式 master | `29500` | `HEAD_PORT`；所有节点必须一致，且网络可达 |
| worker 本地 PORT | `8301` | worker 是 `--headless`，**不监听 HTTP**；只用于本机端口检查/日志定位 |

`multinode-run.sh` 会把下面四项追加到模板的 `EXTRA_VLLM_ARGS`：

```text
--nnodes <NODES> --node-rank <NODE_RANK> \
--master-addr <HEAD_ADDR> --master-port <HEAD_PORT>
```

模板里已有的额外参数会保留；如果你在 `EXTRA_VLLM_ARGS`/`EXTRA_ARGS_FILE` 里
重复写了这四项，脚本会报错退出 2，要求删掉（避免两套地址/rank 打架）。

---

## 2. 版本前提

* vLLM 必须支持多机 TP：`vllm serve --help` 里有
  `--nnodes`、`--node-rank`、`--master-addr`、`--master-port`；
  worker 还需要 `--headless`（本骨架的 worker 模板依赖它）。
* 先自查（每台机器都查，`HOST_PYTHON` 不同）：

  ```bash
  ${HOST_PYTHON:-python3} -c \
    'from vllm.entrypoints.cli.main import main; main()' serve --help \
    | grep -E -- '--nnodes|--node-rank|--master-addr|--master-port|--headless'
  ```

* `multinode-run.sh` 在启动前会自动做同一项检查；查不到时**拒绝启动**并提示。
  如果已手工确认版本（例如用了自定义 `HOST_SERVE_ENTRYPOINT`），可显式跳过：

  ```bash
  MULTINODE_SKIP_VERSION_CHECK=1 ...
  ```

* **不要假设 CUDA 版本支持就等于海光 DCU/ROCm 版本支持**：请在实际运行的
  vLLM 构建里确认 `--nnodes`/`--headless` 与分布式后端（NCCL/RCCL/HCCL）可用。

---

## 3. 每台机器各自配置

多机不是“只配 head”。**每个节点都要有同一份仓库、各自的 `.env`/`HOST_PYTHON`**：

| 项 | head | worker |
|---|---|---|
| `HOST_PYTHON` | 本机装 vLLM 的解释器 | 本机装 vLLM 的解释器（不一定和 head 同路径） |
| `HOST_PRELUDE_FILES` | 本机厂商环境（DTK/CANN 等） | 同左，按本机路径 |
| `MODEL` | 同一模型目录 | **必须同一个模型**（本地盘/NFS 均可，但内容一致） |
| `SERVED_NAME` | 一致 | 一致 |
| `TP` | **跨节点总数** | 与 head 完全一致 |
| `VISIBLE_DEVICES` | 本机可见卡 | 本机可见卡 |
| `GPU_MEM_UTIL`/`MAX_MODEL_LEN`/KV dtype | 一致 | 一致 |
| `NODE_RANK` | `0` | `1..NODES-1`，每台不同 |
| `NODES`/`HEAD_ADDR`/`HEAD_PORT` | 所有节点一致 | 所有节点一致 |

**`TP` 是跨节点总数**：示例是两节点各 4 卡，所以两个模板都写 `TP=8`，
每节点 `VISIBLE_DEVICES=0,1,2,3`。如果你的机器是 8 卡/节点、两节点，
就写 `TP=16`、`VISIBLE_DEVICES=0,...,7`。卡数/TP 不一致会导致 worker 加入失败。

模型路径也要注意：`hosts/vllm-host.sh show` 只检查本机路径是否存在（不存在只告警），
worker 加载不到模型时会在 `serve.log` 里报错。

---

## 4. 先 dry-run（默认行为）

```bash
cd /path/to/vllm-docker-probe

# ① 看两个模板叠加结果（不启动）
hosts/vllm-host.sh show deploy/multinode-head
hosts/vllm-host.sh show deploy/multinode-worker

# ② 不带闸门：打印预览并退出 2
hosts/deploy/multinode-run.sh
echo "退出码：$?"     # 2

# ③ 只开闸门、不给 HEAD_ADDR：仍然退出 2（在版本检查之前）
MULTINODE_ENABLE=1 hosts/deploy/multinode-run.sh

# ④ 想看真实 HEAD_ADDR 的命令，但仍不启动：只给 HEAD_ADDR/NODES/NODE_RANK，不开闸门
NODES=2 NODE_RANK=0 HEAD_ADDR=10.0.0.10 hosts/deploy/multinode-run.sh
```

预览里会显示将要执行的 head/worker 命令、追加的节点参数、以及版本检查方式。
确认 `HEAD_ADDR`/`NODES`/`NODE_RANK`/`TP` 后再启动。

---

## 5. 启动顺序：先 head，再 worker

### head（rank 0）

```bash
cd /path/to/vllm-docker-probe
MULTINODE_ENABLE=1 \
NODES=2 NODE_RANK=0 \
HEAD_ADDR=10.0.0.10 HEAD_PORT=29500 \
  hosts/deploy/multinode-run.sh --head
```

`multinode-run.sh` 最终执行：

```bash
EXTRA_VLLM_ARGS='--nnodes 2 --node-rank 0 --master-addr 10.0.0.10 --master-port 29500' \
  hosts/vllm-host.sh up deploy/multinode-head -- --run-dir runs/multinode-head
```

`up` 会等 `/health` 就绪；**只有所有 worker 都加入后 head 才会完全就绪**，
所以大模型 + 多机时 `TimeoutStartSec`/`WARMUP_TIMEOUT` 要给足（模板默认 1800s）。

### worker（rank 1..N-1，每台分别执行）

```bash
cd /path/to/vllm-docker-probe
MULTINODE_ENABLE=1 \
NODES=2 NODE_RANK=1 \
HEAD_ADDR=10.0.0.10 HEAD_PORT=29500 \
  hosts/deploy/multinode-run.sh --worker
```

worker 最终执行：

```bash
EXTRA_VLLM_ARGS='--headless --nnodes 2 --node-rank 1 --master-addr 10.0.0.10 --master-port 29500' \
  hosts/vllm-host.sh up deploy/multinode-worker -- --run-dir runs/multinode-worker --no-wait
```

* `--headless` 来自 `multinode-worker.env`，不要删；
* `--no-wait` 由脚本追加：worker 没有 `/health`，等就绪会一直超时；
* worker 启动后看本机 `runs/multinode-worker/serve.log`，确认它开始连接
  `HEAD_ADDR:HEAD_PORT`，并在 head 日志里看到该 rank 加入。

> 端口/防火墙：head 的 `8300`（API）与 `29500`（master）要对 worker/客户端放行；
> worker 的 `8301` 只是本地占位，不需要对外开放。

---

## 6. 状态 / 日志 / 停止

```bash
# head：有 HTTP，可 status/probe
PORT=8300 hosts/vllm-host.sh status deploy/multinode-head
PORT=8300 hosts/vllm-host.sh probe  deploy/multinode-head

# worker：headless，没有 /health，不能 probe；只能看日志
tail -f runs/multinode-worker/serve.log

# head 上确认所有 rank 都加入了
tail -f runs/multinode-head/serve.log
```

停止（**每台机器各自停自己的**；`down` 默认只取最新运行目录，这里显式给 `--run-dir`）：

```bash
# head 机器
hosts/vllm-host.sh down deploy/multinode-head -- --run-dir runs/multinode-head

# 每台 worker 机器
hosts/vllm-host.sh down deploy/multinode-worker -- --run-dir runs/multinode-worker
```

建议先停 worker，再停 head；反过来停 head 会让 worker 反复重连/报错，
最后仍需在 worker 上 `down` 清理。

---

## 7. 排错

| 现象 | 先查 |
|---|---|
| `unrecognized arguments: --nnodes` | vLLM 版本不支持多机；换版本，别用 `MULTINODE_SKIP_VERSION_CHECK` 硬绕 |
| `unrecognized arguments: --headless` | worker 版本不支持/参数名不同；按版本改 worker 模板的 `EXTRA_VLLM_ARGS` |
| worker 一直重连/超时 | head 没先起；`HEAD_ADDR` 不可路由；`HEAD_PORT` 被防火墙挡；`NODES/NODE_RANK` 不一致 |
| head 永远等不到 `/health` | worker 没全部加入；看 head 日志缺哪个 rank；看 worker 是否加载模型失败 |
| worker 加载模型失败 | 模型路径在本机不存在/不一致；`HOST_PYTHON` 不对；显存不足 |
| `NODE_RANK` 重复 | 每台 worker 必须不同；脚本只校验本机 `0 <= rank < NODES`，跨机重复需人工保证 |
| OOM / 通信初始化失败 | `TP` 与总卡数不符；`VISIBLE_DEVICES`/驱动/RCCL/NCCL 环境；`GPU_MEM_UTIL` 太高 |
| `down` 停错实例 | 显式 `--run-dir`；head/worker 目录不同 |
| 端口被占用 | head `8300`/`29500`、worker `8301`；同机冒烟时尤其容易撞 |

---

## 8. 仍然缺的真实环境信息

本骨架刻意不猜下面这些；补齐前不要上生产：

1. **实际 vLLM 构建**：是否支持 `--nnodes`/`--node-rank`/`--master-addr`/
   `--master-port`/`--headless`；海光 DCU（ROCm/HIP）与 CUDA 版本差异。
2. **拓扑**：几台机器、每台几张卡、`TP` 总数、`VISIBLE_DEVICES` 分配；
   是否同交换机/NUMA；是否需要设置 `NCCL_*`/`RCCL_*`/`HCCL_*` 环境变量。
3. **网络事实**：`HEAD_ADDR` 用哪个网段/IP、`HEAD_PORT` 是否放行、
   是否有 IB/RDMA 与对应驱动；DNS/hostname 是否可用。
4. **每节点环境**：各自的 `HOST_PYTHON`（venv/conda 路径）、`HOST_PRELUDE_FILES`
   （DTK/CANN 脚本）、模型路径（本地盘还是 NFS）与读写权限。
5. **版本一致性**：所有节点的 vLLM/torch/通信库版本、模型权重/tokenizer 完全一致。
6. **观测**：只有 head 有 HTTP `/metrics`；worker 的进程/显存/日志要单独采集，
   本仓库暂不生成跨机聚合报告。
