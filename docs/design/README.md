# 设计与施工图 · 引擎组件即插即用体系（docs/design/）

> 状态：**待施工**。本目录是给实施 agent 的完整设计说明与施工图；除本目录外不涉及任何代码改动。
> 编写日期：2026-10-22，基于当时 HEAD（`5dd4d9e`）的代码结构。文中 `file:line` 为编写时位置，
> 施工时以符号/段落定位为准（锚点偏移不算错，引用错符号才算）。
>
> 全部文档使用中文；术语与既有《[../施工图-DeepSeek部署栈支持.md](../施工图-DeepSeek部署栈支持.md)》保持一致
> （`ENGINE`、`RUNTIME=host`、`hosts/` 模板层、`vendors/` 厂商层）。

---

## 0. 一句话背景

本仓库是**大模型推理引擎探测工具箱**：探针面（HTTP / 指标 / 日志 / trace / 报告 / 套件）目前
隐含假设"被探测的引擎就是 vLLM"。现在要适配 [Strata](https://github.com/Niko1221/Strata)
（消费级 PC 上的 MoE 专家分层推理引擎，OpenAI 兼容 API，`/metrics` 直接以 vLLM 指标名输出
Prometheus 文本），并借此把"引擎身份"从硬编码改为**即插即用的组件接口**，让未来的其它推理
引擎（DeepInfer、SGLang 系 fork、llama-server 系……）按同一契约接入。

## 1. 概念速查（四个正交维度）

| 维度 | 配置键 | 取值 | 谁负责 |
|---|---|---|---|
| 运行方式 | `RUNTIME` | `docker` \| `host` | `scripts/start_vllm.sh`、docker compose |
| **引擎组件**（本期新增） | `ENGINE` | `vllm`（默认）\| `strata` \| 未来… | `engines/*.sh`（shell 端注册表）+ `probe/engines.py`（Python 端注册表） |
| 硬件厂商 | `VENDOR` | nvidia/hygon/amd/intel/ascend/generic | `vendors/*.sh` + `probe/vendors.py`（**本期不动**） |
| 部署形态 | hosts 模板链 | vendor/ scenario/ quant/ spec/ deploy/ **engine/**（新增类目） | `hosts/lib/env_overlay.sh` |

边界裁定：**"引擎组件"= 提供服务进程的软件**（替换 serve 入口）；DeepEP 这类只追加
flag/env 的**增强型组件不走本接口**，继续用 `hosts/<类目>/*.env` 模板（见
《../施工图-DeepSeek部署栈支持.md》阶段 1）。

## 2. 特性分解与目录索引

每个功能一个文件夹，设计与施工图都放在文件夹内：

| 目录 | 功能 | 内容 | 依赖 |
|---|---|---|---|
| [`component-interface/`](component-interface/) | **F1 · 引擎组件即插即用接口** | [设计.md](component-interface/设计.md)（概念模型、双端契约、能力枚举、守卫规范）<br>[施工图.md](component-interface/施工图.md)（任务 T1.x） | 无（一切的地基） |
| [`strata-engine/`](strata-engine/) | **F2 · Strata 引擎适配**（首个落地组件） | [事实核对.md](strata-engine/事实核对.md)（上游版本/API/安装/指标/坑位清单）<br>[施工图.md](strata-engine/施工图.md)（任务 T2.x） | F1 的注册表骨架（阶段 1） |
| [`observability-compat/`](observability-compat/) | **F3 · 观测面组件兼容**（探针层的引擎感知） | [设计.md](observability-compat/设计.md)（metrics/http/report/trace/status 按能力协商）<br>[施工图.md](observability-compat/施工图.md)（任务 T3.x） | F1 的能力枚举定型 |

## 3. 施工顺序

```text
F1 阶段1（Python 注册表 + 守卫）  ──┐
F1 阶段2（shell 注册表 + 配置键）──┼──> F2（strata 组件落地：engines/strata.sh + hosts/engine/strata.env）
F1 阶段3（启动链分支收拢）        ──┤         │
F1 阶段4（hostcheck 引擎感知）    ──┘         ▼
                                    F3（观测面兼容：metrics/report/trace/status 按能力协商）
```

* F1 内部阶段按序执行（每阶段后跑 `bash tests/run_tests.sh`，全绿再进下一阶段）。
* F2 依赖 F1 阶段 1–2（注册表可注册即可动手），与 F1 阶段 3–4 可并行但建议串行（都碰
  `scripts/` 白名单与 selftest）。
* F3 依赖 F1 的能力枚举定型；其中"Strata 指标核对"子项依赖 F2 的组件文件就位。
* 每个特性文件夹内的施工图都自带独立验收清单；全仓红线见 §5。

## 4. 新引擎接入成本（设计目标）

接口定型后，接入一个新引擎 = **三份新文件 + 两行索引**，不碰任何既有脚本逻辑：

```text
engines/<id>.sh              # shell 端组件 profile（入口/进程正则/能力位）
probe/engines.py             # Python 端注册表追加一条 EngineProfile
hosts/engine/<id>.env        # 引擎模板（模型/并行参数的引擎语义）
tests/host_templates_selftest.sh   # EXPECTED_TEMPLATES + 断言表各加一行
hosts/README.md §3           # 索引表加一行
```

`start_vllm.sh` / `probe/*.py` 的既有分支**不新增** `ENGINE == "<id>"` 判断——一切差异走
能力位（caps）协商。这是"即插即用"的验收标准。

## 5. 全局红线（三个施工图共同遵守）

1. **vLLM 默认路径逐字节不变**：不设置 `ENGINE`（或 `ENGINE=vllm`）时，任何 dry-run 产物与
   施工前基线 diff 为空；探针产物 schema、报告结构、套件行为不变。
2. **`vendors/` 厂商层与 `probe/vendors.py` 一律不动**（引擎与厂商正交）。
3. **`docker/` 一律不动**（Strata/DeepInfer 均无官方镜像，docker 支持列为暂缓项）。
4. 键名白名单纪律：新增配置键（`ENGINE`、`EXTRA_ENGINE_ARGS`、`HOST_ENGINE_PATTERN` 等）
   必须同时进 `.env.example` 与 `scripts/lib/common.sh`（`tests/host_templates_selftest.sh`
   的键名守卫会拦）；`hosts/README.md` §3 索引即契约，新模板必须登记。
5. `scripts/` 改动白名单：`tests/host_templates_selftest.sh` 末段检查 `git status --porcelain
   -- scripts/`，只放行 `scripts/start_vllm.sh`；施工要动 `scripts/lib/common.sh` 前先扩该
   `grep -vE` 白名单（或先 commit）。
6. 探针保持纯标准库（≥3.10），禁止第三方依赖；容器内也要能跑。
7. 默认端口仍是 8100、容器名仍是 `vllm-probe`（刻意避开生产 :8000）；Strata 自身默认 8080
   仅作为其引擎默认值出现在注释里，不改变本仓库 `PORT` 语义。
8. 全量测试不需要 GPU/Docker/vLLM：`bash tests/run_tests.sh` 全绿是任何施工的验收底线。

## 6. 与既有文档的关系

* 《[../施工图-DeepSeek部署栈支持.md](../施工图-DeepSeek部署栈支持.md)》（DeepEP / DeepInfer /
  deepseek-recipe）**保持不动**。其阶段 2 的 `ENGINE=vllm|deepinfer` 硬编码方案与 F1 接口的
  关系：若 F1 先施工，DeepInfer 直接按 F1 契约注册（其 T2.1/T2.2 改走注册表）；若该图先施工，
  F1 的施工图含一个"把 ENGINE 分支收拢进注册表"的迁移任务。两图互不阻塞。
* [README.md](../../README.md) §2（目录结构）与 [hosts/README.md](../../hosts/README.md) §3
  在各施工图内列出了需要同步的索引行，施工时按图更新。
* [docs/观测字段对照表.md](../观测字段对照表.md) 若因 F3 增加 Strata 指标字段，按其现有格式
  追加章节。
