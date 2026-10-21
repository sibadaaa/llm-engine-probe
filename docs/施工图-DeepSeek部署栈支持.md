# 施工图 · DeepSeek 部署栈支持（DeepEP / DeepInfer / deepseek-recipe）

> 状态：**待施工**（本文档是给实施 agent 的完整施工说明，当前仓库未包含任何对应实现）。
> 编写日期：2026-10-21，基于当时 HEAD 的代码结构；文中 `file:line` 为编写时位置，施工时以符号/段落定位为准。
>
> **范围裁定（已与需求方确认）**：
> 1. **Docker 模式支持暂缓**——不动 `docker/` 目录；`ENGINE=deepinfer` 仅在 `RUNTIME=host` 下允许，docker 模式显式报错。
> 2. **deepseek-recipe 只做接入指南文档**，不做代码集成（不写 shim 服务）。
> 3. **全部为结构适配**（同 `vendors/` 里 amd/intel/ascend 的做法）：无实测硬件（DeepInfer/DeepEP 仅支持 NVIDIA SM100/Hopper+ 与 Ascend 950），验收标准 = dry-run + 自测通过。
> 4. 探针产物 schema、报告结构、套件行为**保持不变**；`ENGINE=vllm`（默认）路径行为逐字节不变。

---

## 0. 执行顺序与任务索引

| 阶段 | 内容 | 任务 | 依赖 |
|---|---|---|---|
| 1 | DeepEP / DeepEP-Ascend（纯模板层 + 探针小改） | T1.1–T1.5 | 无 |
| 2 | DeepInfer 引擎适配（`ENGINE` 配置流） | T2.1–T2.7 | 无（与阶段 1 可并行，但都改 hosts/README 与 selftest，建议顺序执行） |
| 3 | deepseek-recipe 接入指南（纯文档） | T3.1 | 阶段 1/2 完成后写，便于互相链接 |

每阶段完成后跑一次 `bash tests/run_tests.sh`（全绿再进入下一阶段）。

---

## 1. 背景速查（三个项目的事实核对清单）

### 1.1 DeepInfer（github.com/deepseek-ai/DeepInfer，Apache-2.0）

* DeepSeek 生产推理框架，**主要基于 vLLM 改造**（另借鉴 SGLang / TensorRT-LLM）。
* **当前开源版只有离线推理**：`python examples/offline_inference.py`；无 OpenAI HTTP 服务、无 KV cache 系统、无 PD 分离、无副本负载均衡。
* 支持模型：`DeepSeek-V4.1-Flash`（HF: `deepseek-ai/DeepSeek-V4.1-Flash-1017`；可用 `DEEPINFER_HF_MODEL_DIR=/path` 指定本地权重）。
* 特性（影响探针预期的）：Attention DP / balanced CP、MoE EP + EPLB、DSpark 与置信度调度验证、SWA bounded replay、异步调度（零开销 batch scheduler）、batch-invariant 推理、确定性采样、structured outputs。
* 硬件 / 构建：NVIDIA SM100（CUDA 13）或 Ascend 950（CANN 9.2）；Python **3.12**、GCC ≥ 13；构建用 `DEEPINFER_ACCELERATOR=nvidia|ascend ./build.sh`（ascend 用 `requirements-ascend.txt`）。
* **对接本工具的落点**：引擎身份、启动命令（host 模式）、`DEEPINFER_ACCELERATOR` 环境变量、CANN prelude（Ascend）、Python 3.12 解释器（`HOST_PYTHON`）。

### 1.2 deepseek-recipe（github.com/deepseek-ai/deepseek-recipe，MIT）

* Rust 库 + Python 绑定（`pip install deepseek-recipe`，Python 3.10+）：把 Messages / Chat Completions / Responses 请求 → Conversation → DeepSeek V4/V4.1 prompt（`DeepseekV41Encoding().render_conversation(...)`），并解析输出回协议响应。
* 关键 API（Python）：`ChatCompletionRequest` / `ConversionOptions`（`with_model_alias` 注册别名）/ `resolve_max_tokens`（非思考 8192、思考 65536、思考 max 131072）。
* 支持流式输出、图片（V4.1，OpenCV 预处理；V4 不支持图片）、工具调用、`json_schema/regex` 结构化输出、logprobs 透传（库不算概率）、`reasoning_tokens` 统计（需挂 tokenizer）。
* 附带 `server-py`（FastAPI）/ `server-rs`（Axum）**mock 推理**示例——是真服务前端的骨架参考。
* **对接本工具的落点**：只写接入指南文档（T3.1），说明如何把 DeepInfer 离线引擎 + deepseek-recipe 拼成探针可打的 HTTP 服务。

### 1.3 DeepEP / DeepEP-Ascend（github.com/deepseek-ai/DeepEP，MIT）

* MoE 专家并行 all-to-all 通信库（dispatch/combine、FP8 dispatch、NVLink 冗余专家交换）。
* **DeepEP-Ascend**：同一 API 跑在 Ascend 950（独立仓库 deepseek-ai/DeepEP-Ascend）。
* 安装：`git submodule update --init --recursive && bash install.sh`（host 侧 C++ 扩展；kernel 由 DeepJIT 运行时编译，需运行期保留 CUDA toolkit 与 host 编译器）。
* 依赖（NVIDIA 路径）：CUDA ≥ 13.1、PyTorch ≥ 2.10、NCCL ≥ 2.32.3（`pip install "nvidia-nccl-cu13>=2.32.3" --no-deps` 或 `EP_NCCL_ROOT_DIR` 指定）；NVLink（节点内）+ RDMA（跨节点）；Hopper+。
* 关键运行时环境变量（模板要透传的）：`EP_BUFFER_DEBUG`、`EP_SUPPRESS_NCCL_CHECK`、`EP_JIT_CACHE_DIR`（默认 `$HOME/.dj`，多机建议指到共享盘）、`EP_JIT_DEBUG`、`EP_OVERRIDE_RDMA_SL` / `EP_DEFAULT_RDMA_SL`、`EP_DISABLE_GIN`（仅 NVLink 场景）、`EP_REUSE_NCCL_COMM`。
* vLLM 侧启用方式（以本地 vLLM 版本帮助为准）：`--all2all-backend deepep`，配套通常有 `--deepep-mode auto|normal|low_latency`、`--enable-expert-parallel`、DP/EP 相关参数。**施工时先在目标 vLLM 源码里 grep `deepep` 核对可用 flag**（本仓库对接的 vLLM fork 版本不一）。

---

## 2. 现有架构速查（集成点与守卫机制）

施工 agent **必须理解以下机制**，否则会踩坑：

1. **模板叠加层**（`hosts/lib/env_overlay.sh`）：
   * 优先级：显式环境 > 模板链（后者覆盖前者）> 仓库根 `.env`；模板在 `set -a` 下 source，**不得有副作用**（不执行命令/cd/写文件）。
   * **追加型键** `EXTRA_VLLM_ARGS` / `EXTRA_ENV`：模板用 `EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:+$EXTRA_VLLM_ARGS }--xxx yyy"` 形式在末尾追加（参考 `hosts/quant/fp8.env:25`）；若调用者显式导出了同名变量，模板追加会被整体丢弃，`host_template_apply`（`env_overlay.sh:153`）对此打 ⚠️。新模板一律遵守该写法。
   * `host_template_show()` 的展示键数组在 `env_overlay.sh:206-214`——新增配置键要补进去（T2.3）。
2. **键名白名单守卫**（`tests/host_templates_selftest.sh:164-201`）：模板里出现的 `KEY=` 会与「`.env.example` 的 KEY= + `scripts/lib/common.sh` 的 `${KEY:=`」并集比对。**新增 `ENGINE` / `DEEPINFER_ACCELERATOR` 等键必须同时写进 `.env.example` 和 `common.sh`**，否则自测报"未知键"。
3. **模板清单契约**（`tests/host_templates_selftest.sh:89-95` `EXPECTED_TEMPLATES` 数组）：新增模板目录必须同步追加（T1.5 / T2.6）。
4. **scripts/ 改动白名单守卫**（`tests/host_templates_selftest.sh:444-466`）：用 `git status --porcelain -- scripts/` 检查未提交改动，当前只放行 `scripts/start_vllm.sh`。**阶段 2 改 `scripts/lib/common.sh` 前必须先扩这个白名单**（T2.1），否则守卫误报。注意：若改动已 commit，`git status` 不再显示——白名单仍建议扩（防后续返工），并在 PR 说明里注明。
5. **dry-run 断言表**（`tests/host_templates_selftest.sh:375+` `DRY_EXPECT_DESC` / `DRY_EXPECT_DESC2`）：每个模板可声明"serve_launch.sh 必含片段"断言，新模板要加（T1.5 / T2.6）。
6. **host 模式启动链**（`scripts/start_vllm.sh`）：
   * 参数拼装段 ≈ L140-250（观测档位 flag → 投机解码 → NVTX → 插件 → `EXTRA_VLLM_ARGS`/`EXTRA_ARGS_FILE`）；
   * 环境变量段 ≈ L252-279（厂商可见性/分配器/`VENDOR_EXTRA_ENV` → 通用 → `EXTRA_ENV`）；
   * 入口段 ≈ L284-289：host 用 `host_serve_entrypoint()`（`common.sh` 内，逻辑：`HOST_SERVE_ENTRYPOINT` 非空则用之，否则 `$HOST_PYTHON -c 'from vllm.entrypoints.cli.main import main; main()'`）；
   * `run_meta.json` ≈ L316-354（内嵌 `py - <<PY`）；host launcher 生成 ≈ L363-409（`serve_launch.sh`：prelude `set +u` 包裹 → `export K=V` → `exec`）。
7. **hostcheck 结构**（`probe/hostcheck.py`）：检查项注册表 `_CHECKS`（L862-877）；`_check_vllm`（L485）；import 探测在 `_probe_host_python` 的内嵌 Python 脚本（L294 起，`__CHECK_VLLM__` 占位替换）；结果 tracker 的 `add/skip` 带 hint 文案。新增检查项照 `_check_vllm` 的模式写。
8. **trace kernel 归因**（`probe/trace_probe.py`）：`BASE_RULES`（L115-131，跨厂商通用，**已含 `deepep` 关键字**在"通信/集合通信"）+ `CATEGORY_RULE_PACKS`（L135-164，按厂商 id 追加正则；未识别厂商合并为超集 `_ALL_PACKS`）。
9. **厂商层与引擎层正交**：`probe/vendors.py` 注册表、`vendors/*.sh` 一律**不改**；DeepInfer/DeepEP 属引擎/部署栈维度。
10. **测试入口**：`bash tests/run_tests.sh` 跑全部（含 `test_vendors.py` 的 shell↔Python 一致性守卫、`host_templates_selftest.sh`）；探针纯标准库，容器内可跑。

---

## 3. 阶段 1 · DeepEP / DeepEP-Ascend

### T1.1 新增 `hosts/ep/deepep.env`（NVIDIA）

全文草稿（可直接落盘，按仓库模板头注释规范）：

```bash
# 用途：vLLM 启用 DeepEP 专家并行 all-to-all（--all2all-backend deepep），MoE/EP 场景通信归因与调优。
# 适用：NVIDIA Hopper+ 物理机；vLLM 已编译/安装 deep_ep（DeepEP V2）；NVLink 节点内 + 跨节点 RDMA。
# 组合：hosts/vllm-host.sh up vendor/nvidia scenario/tp8-throughput ep/deepep
# ============================================================================
# 说明：
#   * 本模板只追加 DeepEP 相关 flag 与 EP_* 环境，不改并行度/模型；
#     vLLM 各 fork 的 DeepEP flag 名有差异，启动前请用 `dry-run` 复核，
#     并在目标 vLLM 源码里核对（常见：--all2all-backend deepep / --deepep-mode /
#     --enable-expert-parallel / --data-parallel-size）。
#   * DeepEP kernel 由 DeepJIT 运行时编译：首启编译耗时数分钟，且运行期需要
#     CUDA toolkit + host 编译器在 PATH；JIT 缓存建议指到共享盘（见 EXTRA_ENV）。
#   * EP dispatch/combine 需要 GPU SM（零 SM RDMA EP 不支持），不要与
#     --enforce-eager 之外的降 SM 手段叠加评估。
#
# 覆盖键：EXTRA_VLLM_ARGS、EXTRA_ENV（均只在末尾追加，不清空已有值）。
# 风险：
#   1) 显式导出 EXTRA_VLLM_ARGS/EXTRA_ENV 时本模板追加被整体丢弃（vllm-host.sh 会 ⚠️）；
#   2) RDMA 环境（SL/VL、自适应路由、PCI atomic）需按 DeepEP README 的网络配置调好，
#      通信类异常表现为 dispatch/combine 卡死或带宽远低于预期；
#   3) 与 custom all reduce 叠加时注意 all2all 后端一致性（vLLM 会各自选择后端）。
# ============================================================================

# --- DeepEP 启用（末尾追加）---
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:+$EXTRA_VLLM_ARGS }--all2all-backend deepep --enable-expert-parallel"
# deepep-mode 按场景取舍（prefill 用 normal、decode 用 low_latency、auto 兜底）：
# EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:+$EXTRA_VLLM_ARGS }--deepep-mode auto"

# --- DeepEP 运行时环境（末尾追加；EP_* 变量须在 import deep_ep 前生效）---
EXTRA_ENV="${EXTRA_ENV:+$EXTRA_ENV }EP_SUPPRESS_NCCL_CHECK=0 EP_BUFFER_DEBUG=0"
# JIT 缓存指到共享盘可避免每机重复编译（路径按站点改）：
# EXTRA_ENV="${EXTRA_ENV:+$EXTRA_ENV }EP_JIT_CACHE_DIR=/mnt/game/deepep-jit-cache"
```

### T1.2 新增 `hosts/ep/deepep-ascend.env`（Ascend 950）

```bash
# 用途：vLLM + DeepEP-Ascend（Ascend 950 上的同 API 专家并行通信库）。
# 适用：Ascend 950 物理机 + CANN 9.2+；deep_ep 已按 DeepEP-Ascend 安装；vllm-ascend 路径。
# 组合：hosts/vllm-host.sh up vendor/ascend scenario/tp8-throughput ep/deepep-ascend
# ============================================================================
# 说明：DeepEP-Ascend 与 NVIDIA 版同 API（EPBuffer 等）；HCCL 承担 all-to-all。
#   CANN prelude 由 vendor/ascend.env 的 HOST_PRELUDE_FILES 注入，本模板不重复注入。
#
# 覆盖键：EXTRA_VLLM_ARGS、EXTRA_ENV（均只在末尾追加）。
# 风险：
#   1) --all2all-backend deepep 在 vllm-ascend 上的可用性随 fork 版本差异较大，
#      启动前务必 dry-run + 在 vllm-ascend 源码核对；
#   2) kernel 档（torch profiler）在 Ascend 是实验性（见 vendor/ascend.env），建议 steps/deep；
#   3) HCCL 网络相关变量（如 HCCL_OP_EXPANSION_MODE 等站点特定项）请用 EXTRA_ENV 显式追加。
# ============================================================================

EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:+$EXTRA_VLLM_ARGS }--all2all-backend deepep --enable-expert-parallel"
EXTRA_ENV="${EXTRA_ENV:+$EXTRA_ENV }EP_BUFFER_DEBUG=0"
```

### T1.3 `probe/hostcheck.py`：新增 `deep_ep` 检查项（warn-only）

* 在 `_probe_host_python` 的内嵌 Python 脚本里追加 `deep_ep` 探测（与 `import vllm` 同一次子进程完成）：

  ```python
  info["deep_ep"] = None; info["deep_ep_error"] = None
  try:
      import deep_ep
      info["deep_ep"] = str(getattr(deep_ep, "__version__", "unknown"))
  except Exception as exc:
      info["deep_ep_error"] = "%s: %s" % (type(exc).__name__, exc)
  ```
* 新增 `_check_deepep(ctx, tr)`：未安装 → `tr.add("deep_ep", True, "deep_ep 未安装（仅 DeepEP/EP 场景需要；模板 ep/deepep 启用时才必须）")`；已安装 → 上报版本，hint 提示 `EP_JIT_CACHE_DIR` 首启 JIT 编译耗时。**该项永远不 fail**（warn-only），不加入 `--skip-*` 参数。
* 注册进 `_CHECKS` tuple（放 `vllm` 之后）；`collect_hostcheck` 结果 JSON 里的 `python_info` 段带上 `deep_ep` 字段。
* 注意 `_Context` 的字段透传链（`collect_hostcheck` → `_probe_host_python`）里版本字段白名单（≈ L955-960 的 `py_info.get(...)`）要补 `deep_ep`。

### T1.4 `probe/trace_probe.py`：DeepEP kernel 归因细化

* `BASE_RULES` 的"通信/集合通信"已含 `deepep`，**保持不动**；
* 在 `CATEGORY_RULE_PACKS` 中补充 DeepEP V2 / Ascend 的 kernel 名模式（先在真机 trace 里核对，以下为初始值，按实际 kernel 名调整）：
  * `"nvidia"`：`"通信/集合通信": r"deep_ep::|ep::dispatch|ep::combine|internode|intranode|clean_low_latency"`（DeepEP V1 的 kernel 名 `internode_ll` 等如仍在 fork 中出现可保留）；
  * `"ascend"`：`"通信/集合通信": r"deepep|deep_ep"`（HCCL 已有 `hccl|hcom`）。
* 报告文案无需改（归因到既有"通信/集合通信"分类即可）。

### T1.5 索引与自测更新

* `hosts/README.md`：新增 §3.6「EP 通信模板（`ep/`）」索引表（两个模板各一行：用途、关键设置）；§3 目录树补 `ep/` 行；"新增一个模板"章节的类别列表同步。
* `tests/host_templates_selftest.sh`：
  * `EXPECTED_TEMPLATES` 追加 `ep/deepep ep/deepep-ascend`；
  * `DRY_EXPECT_DESC` 为两个模板加断言：serve_launch.sh 必含 `--all2all-backend deepep` 与 `EP_`；
  * 若断言表按"每个模板必须有表项"强校验，无表项会 fail——确认后补齐。

---

## 4. 阶段 2 · DeepInfer 引擎适配（`ENGINE` 配置流，host only）

设计原则：**分支收拢，不重写**。`ENGINE=vllm`（默认）的所有现有行为不变；`ENGINE=deepinfer` 走精简分支。文件名 `start_vllm.sh` 不改（避免脚本链/文档大面积 churn），在文件头注释注明"虽名 start_vllm，经 ENGINE 支持 DeepInfer"。

### T2.1 配置键落地（三处同步，缺一会被键名白名单守卫拦下）

* `scripts/lib/common.sh`：在 `RUNTIME` 段（≈ L118-129）追加：

  ```bash
  # --- 引擎：vllm（默认）| deepinfer（DeepSeek DeepInfer，仅 host 模式） ---
  : "${ENGINE:=vllm}"
  # DeepInfer 构建目标（DEEPINFER_ACCELERATOR=nvidia|ascend），透传给构建/运行环境
  : "${DEEPINFER_ACCELERATOR:=}"
  # 引擎进程清理/查找正则；留空 → 按 ENGINE 取默认（见 engine_process_pattern）
  : "${HOST_ENGINE_PATTERN:=}"
  ```
  并把三者加入 L134 的 export 列表。在 `host_serve_entrypoint()` 附近新增 `engine_process_pattern()`：`HOST_ENGINE_PATTERN` 非空 → 用之；否则 `ENGINE=deepinfer` → `deepinfer|DeepInfer|EngineCore|APIServer`（**结构占位，真机核对后修正**），`ENGINE=vllm` → 现有 `HOST_VLLM_PATTERN` 默认值。`HOST_VLLM_PATTERN` 保留兼容（deepinfer 下被 `HOST_ENGINE_PATTERN` 覆盖语义要在 `.env.example` 写清楚）。
* `.env.example`：紧跟 `RUNTIME=docker` 段后新增「0b. 引擎」小节，内容含上表三键 + 说明（deepinfer 仅 host；Python 3.12 / GCC≥13 / CUDA 13 / CANN 9.2；模型 DeepSeek-V4.1-Flash；DEEPINFER_HF_MODEL_DIR；观测档位降级说明）。
* `hosts/lib/env_overlay.sh` 的 `host_template_show()` keys 数组补 `ENGINE DEEPINFER_ACCELERATOR HOST_ENGINE_PATTERN`。

### T2.2 `scripts/start_vllm.sh`：引擎分支收拢

1. **引擎合法性检查**（读入配置后立刻）：
   ```bash
   case "$ENGINE" in
     vllm) ;;
     deepinfer)
       if ! runtime_is_host; then
         die "ENGINE=deepinfer 暂只支持 RUNTIME=host（DeepInfer 无官方镜像；docker 支持在施工图里列为暂缓项）"
       fi
       case "${DEEPINFER_ACCELERATOR:-nvidia}" in nvidia|ascend) ;; *)
         die "DEEPINFER_ACCELERATOR 只能是 nvidia|ascend，当前：$DEEPINFER_ACCELERATOR" ;; esac
       ;;
     *) die "未知 ENGINE=$ENGINE（可用：vllm|deepinfer）" ;;
   esac
   ```
2. **参数拼装段**（≈ L140-250）用 `if [[ "$ENGINE" == "vllm" ]]` 收拢以下 vLLM 专属块：观测档位 flag（`--enable-logging-iteration-details` 等）、投机解码段、layerwise NVTX、profiler 相关、stat logger 插件段。`ENGINE=deepinfer` 时：
   * `PROBE_LEVEL` 不是 `off|basic` → `warn "DeepInfer 暂不支持逐 step 观测（结构适配），档位降为 basic"` 并置 `LEVEL=basic`；
   * `PROBE_PLUGIN=1` → `warn "stat logger 插件是 vLLM 专属，DeepInfer 下忽略"`；
   * `DRAFT_MODEL/SPEC_METHOD` 非空 → `die`（DeepInfer 无 vLLM 投机解码 flag；如将来支持再放开）。
3. **环境变量段**（≈ L252-279）：通用项（`TOKENIZERS_PARALLELISM`、`PYTHONUNBUFFERED`、`PYTHONPATH`、`EXTRA_ENV`）保留给两引擎；`VLLM_*`、`VLLM_CACHE_ROOT`、插件、`VLLM_NO_USAGE_STATS` 收进 vllm 分支；deepinfer 分支追加 `add_env "DEEPINFER_ACCELERATOR=${DEEPINFER_ACCELERATOR:-nvidia}"` 与 `[[ -n "${DEEPINFER_HF_MODEL_DIR:-}" ]] && add_env "DEEPINFER_HF_MODEL_DIR=..."`（该键同样要进 T2.1 三处白名单，或在模板里用 `EXTRA_ENV` 透传——**推荐后者，少一个白名单键**）。
4. **入口段**（≈ L284-289）：host 且 `HOST_SERVE_ENTRYPOINT` 为空时按引擎给默认：
   ```bash
   if [[ "$ENGINE" == "deepinfer" ]]; then
     # 结构占位：DeepInfer 开源版为离线推理入口；HTTP 服务化见 docs/deepseek-recipe-接入指南.md
     SERVE_ENTRYPOINT_ACTUAL="${HOST_PYTHON} -m deepinfer.entrypoints.offline"
   fi
   ```
   （占位命令**必然无法直接起服务**——这是结构适配的预期行为；`dry-run` 只验证拼装。真机接入时用 `HOST_SERVE_ENTRYPOINT` 覆盖，见 §7。）
5. **`run_meta.json`**：`meta` 增加键 `"engine": os.environ.get("ENGINE", "vllm")`、`"accelerator"`（deepinfer 时为 `DEEPINFER_ACCELERATOR`，vllm 时空串）。
6. **host launcher / stop**：`stop_vllm.sh` 与 `status.sh` 里所有用 `HOST_VLLM_PATTERN` 的位置改走 `engine_process_pattern()`（grep `HOST_VLLM_PATTERN` 全量替换点核对一遍）。

### T2.3 `probe/config.py`：engine 字段

* `ProbeConfig` 增加 `engine: str = "vllm"`；`from_env()` 读 `env_str("ENGINE", "vllm")`；`describe()` 末尾拼 `engine=...`。`dump()` 经 `asdict` 自动带上，产物（http_summary / suite / config.json）即含引擎身份。

### T2.4 `probe/hostcheck.py`：引擎感知

* `collect_hostcheck(...)` 增加 `engine: str = "vllm"` 参数（调用方 `scripts/host_check.sh` / `probe/cli.py` 从 `ENGINE` 环境取）；写入结果 JSON 顶层。
* `_check_vllm`：`engine == "deepinfer"` 且 import 失败 → 降为 `tr.add("vllm", True, "未 import 到 vllm（ENGINE=deepinfer 时非必需；DeepInfer 环境自带内核）")`；import 成功 → 正常上报版本并注明"DeepInfer 基于 vLLM"。
* 新增 `_check_engine(ctx, tr)`：deepinfer → 在 import 探测脚本里追加 `import deepinfer`（同 T1.3 的模式，`info["deepinfer"]` / `info["deepinfer_error"]`），失败则 **fail**（引擎主检查），hint 指向 Python 3.12 / `DEEPINFER_ACCELERATOR` / build.sh；vllm → `tr.skip("engine", "ENGINE=vllm，无独立引擎包")`。注册进 `_CHECKS`（`vllm` 之前）。

### T2.5 探针/报告/状态的最小适配

* `probe/report.py`：报告头部元信息表加一行「引擎」；DeepInfer 时在"自动归因"节前加一句固定说明：逐 step / 投机解码 / stat-logger 数据不可用（结构适配）。
* `scripts/status.sh`：与 `probe/cli.py`（若有版本探测输出）里的 `vllm_version` 标签旁并列 `engine` 字段。
* `probe/metrics_probe.py` / `http_probe.py`：**不改逻辑**（`/metrics` 404 已有容错）；只确认 engine 字段经 `config.dump()` 落产物即可。

### T2.6 模板层：`hosts/engine/`

新增两个模板（全文骨架；模型路径为占位，同 `quant/awq.env` 的约定）：

**`hosts/engine/deepinfer.env`**
```bash
# 用途：切换引擎到 DeepInfer（DeepSeek 官方推理框架，NVIDIA SM100 路径，结构适配）。
# 适用：NVIDIA SM100/Hopper+ 物理机；Python 3.12 venv 已按 DEEPINFER_ACCELERATOR=nvidia 完成源码构建。
# 组合：hosts/vllm-host.sh up engine/deepinfer scenario/tp8-throughput
# ============================================================================
# 覆盖键：ENGINE、DEEPINFER_ACCELERATOR、HOST_ENGINE_PATTERN、HOST_SERVE_ENTRYPOINT（留注释）、
#         MODEL/SERVED_NAME（占位，按站点改）。
# 风险：
#   1) DeepInfer 开源版为离线推理，HTTP 服务化见 docs/deepseek-recipe-接入指南.md；
#      直接 up 会以占位入口启动失败，属预期——请先用 dry-run 复核命令拼装；
#   2) 观测档位降为 off/basic（steps/deep/kernel 依赖 vLLM flag）；
#   3) HOST_PYTHON 必须指向 3.12 解释器（.env 或显式覆盖，本模板不设机器相关路径）。
# ============================================================================
ENGINE=deepinfer
DEEPINFER_ACCELERATOR=nvidia
MODEL=/mnt/game/models/DeepSeek-V4.1-Flash-1017
SERVED_NAME=deepseek-v4.1-flash
# 进程清理正则（结构占位，真机核对后修正）：
HOST_ENGINE_PATTERN="deepinfer|DeepInfer|EngineCore|APIServer"
# 服务化后建议用 HOST_SERVE_ENTRYPOINT 覆盖占位入口（见接入指南文档）：
# HOST_SERVE_ENTRYPOINT="python -m deepinfer.serve ..."
```

**`hosts/engine/deepinfer-ascend.env`**：同上，改 `DEEPINFER_ACCELERATOR=ascend`，风险注记加"CANN 9.2 prelude 由 vendor/ascend.env 注入；CANN 未 source 时 `import torch_npu` 失败"。

### T2.7 索引与自测更新

* `hosts/README.md`：新增 §3.7「引擎模板（`engine/`）」索引表；§1 快速开始与 §3 目录树补行；§7 已知限制补"DeepInfer 模板为结构适配"。
* `tests/host_templates_selftest.sh`：
  * **`scripts/` 改动白名单（L451-453 的 grep -vE）追加 `scripts/lib/common.sh`**（T2.1 改了它）；
  * `EXPECTED_TEMPLATES` 追加 `engine/deepinfer engine/deepinfer-ascend`；
  * `DRY_EXPECT_DESC`：engine/deepinfer 断言 serve_launch.sh 含 `DEEPINFER_ACCELERATOR=nvidia` 与 `ENGINE=deepinfer`；
  * 新增一条单测式检查：`ENGINE=deepinfer RUNTIME=docker scripts/start_vllm.sh --dry-run` 必须**失败**且报错文案含 "host"（引擎合法性守卫）。
* `tests/test_hostcheck.py`：补 deep_ep / engine 检查项的单测（mock py_info，断言 warn-only 与 fail 路径）。

---

## 5. 阶段 3 · deepseek-recipe 接入指南（纯文档）

### T3.1 新增 `docs/deepseek-recipe-接入指南.md`

大纲（章节即验收项）：

1. **为什么需要前端**：DeepInfer 开源版=离线推理；本工具探针面向 OpenAI 兼容 HTTP；deepseek-recipe 负责协议转换（Messages/Chat Completions/Responses → V4.1 prompt）。
2. **最小可用的 FastAPI shim 骨架**（≈60 行代码块）：`pip install deepseek-recipe fastapi uvicorn`；`ChatCompletionRequest → convert(ConversionOptions) → DeepseekV41Encoding().render_conversation → tokenizer 编码 → DeepInfer 离线引擎 → 输出解析 → SSE 回放`；流式实现参考上游 `docs/streaming.md` 与 `server-py/`。
3. **与探针的对接面**（逐条）：
   * 端点：`/v1/chat/completions`、`/v1/completions`（可只实现前者并设 `PROBE_ENDPOINT=chat`）、`/v1/models`（`status.sh` 会探测）、`/metrics`（尽力而为，404 可接受）；
   * 鉴权：带 `--api-key` 时探针自动加 Bearer（`VLLM_API_KEY`）；
   * 端口：绑定 `PORT`（默认 8100）；外部已起服务用 `VLLM_BASE_URL=http://127.0.0.1:8100 scripts/probe_http.sh` 或 `deploy/external-probe.sh` 只读探测；
   * 流式 delta：`probe/http_probe.py:280` 附近已兼容 content / DeepSeek 系 `reasoning_content` 等三种 delta 键形。
4. **已知语义差异与坑**：`resolve_max_tokens` 的三档默认；`reasoning_tokens` 需挂 tokenizer；logprobs 由调用方回填；V4 不支持图片（会被替换为 `[Unsupported Image]`）；structured output 的约束执行方是后端而非 recipe；`model` 名与 `ConversionOptions.with_model_alias`。
5. **与 DeepInfer/DeepEP 的组合拓扑图**（Mermaid 或 ASCII）：recipe(front) → DeepInfer(engine, MoE EP via DeepEP/DeepEP-Ascend)。
6. **真机验证清单**：列出探针全链路（http → suite → report → compare）在该服务上的逐项预期。
* `README.md` §「后续计划支持多引擎」句尾与 `hosts/README.md` §3.7 加链接。

---

## 6. 测试与验收清单（无硬件）

```bash
# ① 全量自测（含模板契约、厂商一致性、键名白名单、scripts 改动白名单）
bash tests/run_tests.sh

# ② 引擎守卫：docker 模式必须拒绝
ENGINE=deepinfer RUNTIME=docker scripts/start_vllm.sh --dry-run   # 预期 die，文案含 host

# ③ DeepInfer host dry-run（结构拼装验证）
ENGINE=deepinfer DEEPINFER_ACCELERATOR=nvidia RUNTIME=host \
  RUNS_DIR=/tmp/opencode/di-runs scripts/start_vllm.sh --dry-run
#   serve_launch.sh 预期含：ENGINE/DEEPINFER_ACCELERATOR 导出、无 vLLM 专属 flag、basic 档降级提示

# ④ 模板叠加 dry-run
hosts/vllm-host.sh show engine/deepinfer
hosts/vllm-host.sh dry-run engine/deepinfer ep/deepep
#   预期命令行含 --all2all-backend deepep --enable-expert-parallel 且 EP_ 环境在 export 列表

# ⑤ 探针无回归
python3 -m probe vendor
python3 -m probe hostcheck --skip-gpu --json   # deep_ep 检查项出现且 warn-only
python3 -m probe --help                        # config engine 字段不出现在非预期位置

# ⑥ vLLM 默认路径逐字节不变：任选一个现有模板 dry-run，与施工前基线 diff 为空
hosts/vllm-host.sh dry-run vendor/hygon scenario/tp4-prod spec/dflash > /tmp/after.txt
#   （施工前先留基线 /tmp/before.txt；ENGINE 不设置时 diff 必须为空）
```

**红线**：⑥ 必须为空 diff；`tests/test_vendors.py` 一致性守卫不允许因本次改动出现新增键；除 §3/§4 列出的文件外不得改动其它文件（尤其 `vendors/`、`docker/`、`probe/vendors.py`）。

---

## 7. 真机接入 checklist（后续，无硬件期不做）

1. 在目标 vLLM fork 源码核对 DeepEP flag 名与默认值，回填 `hosts/ep/*.env` 注释与 `DRY_EXPECT_DESC` 断言；
2. 用真实 DeepInfer 启动命令替换 `hosts/engine/*.env` 的 `HOST_SERVE_ENTRYPOINT` 占位，并核对 `HOST_ENGINE_PATTERN`（`ps -ef | grep -i deepinfer`）；
3. 按接入指南搭 recipe 前端，用 `deploy/external-probe.sh` 走通 http_probe → suite → report → compare 全链路；
4. 抓一次 kernel trace，核对 `trace_probe.py` 中 DeepEP/Ascend kernel 正则的实际命中，回填模式；
5. 首启 JIT 编译耗时确认后，把 `EP_JIT_CACHE_DIR` 共享盘建议写回模板注释；
6. 视需要再评估 Docker 支持（DeepInfer 官方镜像 / 源码编译镜像）。

## 8. 明确不做（本期）

* `docker/` 任何改动（DeepInfer/DeepEP 镜像、compose overlay）；
* deepseek-recipe 的代码集成（shim 服务代码）；
* DeepInfer 的 PD/多机模板（其开源版尚不含 disagg/LB）；
* `vendors/` 厂商适配层与 `probe/vendors.py` 的任何改动；
* 真实硬件调参（EP 冗余专家、EPLB、RDMA SL/VL 等）。
