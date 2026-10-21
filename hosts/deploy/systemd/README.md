# systemd 常驻示例（host 模式）

[`vllm-host-probe.service`](vllm-host-probe.service) 是把 `hosts/vllm-host.sh up`
固化成 systemd 服务的**示例**，用于开机自启/统一限制项。它和
[`docs/物理机模式.md`](../../../docs/物理机模式.md) §10 的写法一致，只是入口从
`make up-host` 换成了带模板链的 `hosts/vllm-host.sh up`。

> **systemd 只写文件、不安装。** 本目录不会自动 `systemctl enable/start`；
> 安装、启用、排错都由你手动执行。
>
> **先手工验证，再交给 systemd。** systemd 的环境比登录 shell 小得多
> （没有 `~/.bashrc`、没有 venv 激活、没有 DTK/Ascend 的 profile），
> 先在 shell 里用同一组模板跑通，再改 unit 里的路径/用户/解释器。

---

## 1. 安装前：手工验证同一组模板

```bash
cd /path/to/llm-engine-probe

# ① 只看叠加结果，不启动
hosts/vllm-host.sh show vendor/hygon scenario/tp4-prod spec/dflash

# ② 可选：预检（需要 GPU/驱动；失败先修环境）
hosts/vllm-host.sh check vendor/hygon scenario/tp4-prod

# ③ 手工启动（阻塞到 /health 就绪或超时）
HOST_TEMPLATES="vendor/hygon scenario/tp4-prod spec/dflash" \
  hosts/vllm-host.sh up

# ④ 看状态/日志，然后停掉
HOST_TEMPLATES="vendor/hygon scenario/tp4-prod spec/dflash" \
  hosts/vllm-host.sh status
tail -f runs/<时间戳>/serve.log
HOST_TEMPLATES="vendor/hygon scenario/tp4-prod spec/dflash" \
  hosts/vllm-host.sh down
```

手工验证时最容易漏的三件事（systemd 里必须显式写）：

* `HOST_PYTHON`：systemd 不激活 venv/conda，`python3` 通常 import 不到 vLLM；
* `HOST_PRELUDE_FILES`：海光 DTK、昇腾 CANN 等环境脚本不会自动 source；
* 模型/缓存路径：`.env` 里的路径对 `User=` 必须可读，缓存目录必须可写。

---

## 2. 修改 unit 里的占位符

复制前先编辑 [`vllm-host-probe.service`](vllm-host-probe.service)，把这几处改成你的机器：

| 占位符 | 改成 |
|---|---|
| `User=me` | 运行 vLLM 的非 root 用户（能访问 GPU 设备/模型盘） |
| `WorkingDirectory=/home/me/llm-engine-probe` | 你的仓库绝对路径 |
| `ExecStart=/home/me/.../hosts/vllm-host.sh up` | 同一路径下的入口脚本 |
| `ExecStop=/home/me/.../hosts/vllm-host.sh down` | 同上 |
| `Documentation=file:///home/me/...` | 可选，改成你的 README 路径 |
| 注释掉的 `Environment=HOST_PYTHON=...` | 取消注释并填真实解释器 |
| 注释掉的 `Environment=HOST_PRELUDE_FILES=...` | 厂商环境脚本，按需 |

`Environment="HOST_TEMPLATES=vendor/hygon scenario/tp4-prod spec/dflash"` 是默认模板链，
按你的厂商/场景/投机解码改；模板顺序是**后面覆盖前面**。

> `ExecStart` 不要写 shell 管道/`&&`/变量展开：systemd 不经过 shell。
> 需要复杂逻辑时，把参数放进 `HOST_TEMPLATES`，入口脚本负责其余叠加。
> 路径中有空格时，systemd 的 `ExecStart` 需要用引号包住整个可执行文件路径。

---

## 3. 安装 / 启用 / 启动

```bash
# 安装（文件名必须与 unit 里的 [Unit] 名称无关，但建议保持一致）
sudo install -m 0644 hosts/deploy/systemd/vllm-host-probe.service \
  /etc/systemd/system/vllm-host-probe.service

# 重新加载 unit（每次改 unit 都要做）
sudo systemctl daemon-reload

# 开机自启（还不启动）
sudo systemctl enable vllm-host-probe

# 启动：会阻塞到 /health 就绪；大模型可能 5~15 分钟（TimeoutStartSec=1200）
sudo systemctl start vllm-host-probe

# 一条命令 enable + start
sudo systemctl enable --now vllm-host-probe
```

> `systemctl start` 的返回要等 `ExecStart` 结束。`hosts/vllm-host.sh up` 在
> `/health` 就绪后返回，所以这里是“启动完成”而不是“进程常驻前台”。
> 端口被占用、模型路径错误、`HOST_PYTHON` 不对时，start 会失败并返回非 0。

---

## 4. 状态 / 日志

```bash
systemctl status vllm-host-probe
systemctl is-active vllm-host-probe

# 跟踪启动/停止输出（hosts/vllm-host.sh 与 start_vllm.sh 的日志）
journalctl -u vllm-host-probe -f
journalctl -u vllm-host-probe -n 200 --no-pager

# 服务起来后，vLLM 自己的引擎日志仍在运行目录里
tail -f runs/<时间戳>/serve.log
hosts/vllm-host.sh status vendor/hygon scenario/tp4-prod
```

`Type=oneshot` + `RemainAfterExit=yes` 的正常状态是：

```text
Active: active (exited) since ...
```

`active (exited)` 不是“进程已退出”，而是 systemd 记住了 oneshot 成功；
真正的 vLLM 进程在后台（PID 记录在 `<RUN_DIR>/serve.pid`）。

---

## 5. 停止 / 禁用 / 卸载

```bash
# 停止：ExecStop=hosts/vllm-host.sh down，优雅终止进程组
sudo systemctl stop vllm-host-probe

# 停止后不再开机自启
sudo systemctl disable vllm-host-probe

# 卸载示例
sudo rm -f /etc/systemd/system/vllm-host-probe.service
sudo systemctl daemon-reload
sudo systemctl reset-failed vllm-host-probe
```

> `stop` 默认还会等显存回落（最长约 60s），所以 `TimeoutStopSec=90`。
> 同机还有别的显存占用、或只是假实例调试时，可在 unit 里加
> `Environment=PROBE_SKIP_VRAM_WAIT=1` 跳过等待（docs §5.2）。

`systemctl restart` = `stop` + `start`，会新建一个运行目录；上一次的
`runs/<时间戳>/` 会保留，便于对比。

---

## 6. 修改模板/环境后

```bash
sudoedit /etc/systemd/system/vllm-host-probe.service   # 改 Environment=...
sudo systemctl daemon-reload
sudo systemctl restart vllm-host-probe
systemctl show vllm-host-probe -p Environment           # 确认 systemd 看到了变量
```

模板文件（`hosts/**/*.env`）改了**不需要** `daemon-reload`，但需要
`systemctl restart` 才会用新模板重建实例（`up` 每次生成新的 `serve_launch.sh`）。

---

## 7. 排错

| 现象 | 先查什么 |
|---|---|
| `status=203/EXEC` | `ExecStart` 路径不存在/不可执行；`chmod +x hosts/vllm-host.sh`；shebang 是否 `#!/usr/bin/env bash` |
| `status=1/FAILURE`，很快退出 | `journalctl -u vllm-host-probe -n 200`；用 `User=` 同一用户手工跑 ③ 的命令 |
| `start` 卡到超时 | 大模型冷启动慢：调大 `TimeoutStartSec`（如 1800/2400）；同时看 `serve.log` 是否在加载权重 |
| 一直等不到 `/health` | `HOST_PYTHON` 能否 `import vllm`；`HOST_PRELUDE_FILES` 是否漏了 DTK/CANN；模型路径是否存在；端口是否被占用 |
| `端口 ... 已被占用` | `ss -ltnp \| grep :8100`；停掉旧实例，或改模板/`.env` 的 `PORT` |
| `stop` 很慢/超时 | 显存回落等待：设 `PROBE_SKIP_VRAM_WAIT=1`；或手工 `hosts/vllm-host.sh down ...` 观察 |
| 服务 active 但 vLLM 已死 | 本模板 `Restart=no`，**不会自动拉起**（见 unit 注释）；要自动重启请把 `serve_launch.sh` 作为 `ExecStart` 或套 supervisor |
| `unit not found` | 文件名与 `systemctl` 用的名字不一致；先 `daemon-reload` |
| 用户看不到 GPU | `id` 检查 `video`/`render` 组；设备权限由 udev/驱动决定，与 systemd 无关 |
| `HOST_TEMPLATES` 没生效 | `systemctl show ... -p Environment`；含空格的多个模板必须整体用双引号 |
| 想复用上一次运行目录 | `up` 默认每次新建；需要固定目录时在 unit 的 `ExecStart` 后加参数不方便，建议手工 `hosts/vllm-host.sh up ... -- --run-dir runs/xxx` 验证，再决定是否改 unit |

---

## 8. 已知限制

* **不会自动重启**：`Type=oneshot` + `RemainAfterExit=yes` 只保证 systemd 记录
  “已启动”并在 stop 时调用 `down`；vLLM 崩溃后 systemd 不会拉起。这是刻意的，
  与 docs §10 一致——探针侧只负责生成配置/做探测，不抢进程管理器的职责。
* **只管理本工具启动的实例**：`down` 按运行目录里的 `serve.pid`/`serve.pgid`
  停止；systemd 手工改过启动命令、或用外部脚本起的实例，不在本 unit 管理范围内。
* **systemd 环境最小**：不要依赖登录 shell 的 PATH/venv/profile；所有环境变量
  写进 unit 或 `.env`。
* **需要 root 安装**：写 `/etc/systemd/system/`、`daemon-reload`、`enable` 需要 sudo；
  运行实例本身建议用普通用户（`User=`）。
* **端口/显存互斥**：同一端口只能一个实例；同机同时跑 PD/多机调试实例时，
  先看 [`../pd-README.md`](../pd-README.md) 与 [`../multinode-README.md`](../multinode-README.md)。
