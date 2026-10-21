# AGENTS.md

面向 agent 的仓库速览。文档、CLI 输出、报告、注释**全部为中文**，新代码与文档保持中文。

## 定位与结构

- 大模型推理引擎探测工具箱（主 vLLM，多厂商）：Bash 脚本（`scripts/`、`vendors/`、`hosts/`）+ 纯标准库 Python 包 `probe/`（≥3.10，**禁止引入第三方依赖**，容器内也要能跑）。
- 双运行方式 `RUNTIME=docker|host`（默认 docker），共享同一套探针/产物；物理机指南见 `docs/物理机模式.md`。
- 配置流向：`.env`（模板 `.env.example`）→ 环境变量 → CLI 参数，键定义集中在 `scripts/lib/common.sh`。
- 产物落 `runs/<时间戳>/`（REPORT.md、suite_results.json、trace 等）；`make clean-runs` 清理。

## 常用命令

```bash
bash tests/run_tests.sh        # 全量验证：unittest + bash -n + mock 服务端到端 + host/模板自测
python3 -m unittest tests.test_vendors -v   # 单个测试模块（discover -s tests）
python3 -m probe <subcommand>  # CLI：env/metrics/http/logs/trace/report/plugin/vendor/telemetry/suite/compare/hostcheck
scripts/start_vllm.sh --dry-run           # 验证启动拼装，不起实例
hosts/vllm-host.sh dry-run vendor/hygon scenario/tp4-prod spec/dflash   # 模板叠加 dry-run
```

全量测试**不需要 GPU/Docker/vLLM**（用 `tests/mock_vllm_server.py` + `samples/` 夹具）；任何脚本/probe 改动都以 `run_tests.sh` 绿为验收标准。

## 高频踩坑

- **默认端口 8100、容器名 `vllm-probe` 刻意避开现网生产实例（:8000）**：不要把默认值改成 8000 或对 8000 发探测请求。
- **`scripts/` 改动白名单守卫**：`tests/host_templates_selftest.sh` 末段用 `git status --porcelain -- scripts/` 检查未提交改动，白名单目前只放行 `scripts/start_vllm.sh`；改 `scripts/lib/common.sh` 等其它脚本前先扩该 `grep -vE` 白名单（或先 commit）。
- **模板键名白名单**：`hosts/*/*.env` 里出现的 `KEY=` 必须已存在于 `.env.example` 或 `common.sh`，否则被静默忽略且自测报"未知键"。
- **自测禁止写 `runs/`**：模板自测会把 `RUNS_DIR` 重定向到临时目录， leakage 会被判 fail。

## 厂商抽象层（`vendors/` ↔ `probe/vendors.py`）

- 两端键集合必须完全一致（`tests/test_vendors.py` 强制）；`vendors/*.sh` 会在子 shell 中 source 读元数据，**必须无副作用**（只允许变量赋值/函数定义）。
- 新增厂商三步流程见 `vendors/README.md`（shell profile → `VendorProfile` → 必要时 telemetry Backend + `CATEGORY_RULE_PACKS` 规则包）。

## hosts/ 物理机启动模板

- 模板是 `set -a` 下 source 的 shell 片段：只写 `KEY=VALUE` 和无副作用判断；**禁止执行命令、cd、写文件**（`show`/`dry-run` 会加载每个模板）。
- 追加型键写法（不要整体覆盖）：`EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:+$EXTRA_VLLM_ARGS }--flag val"`；优先级为 显式环境 > 模板链 > `.env`，显式导出同名键会整体丢弃模板追加（`env_overlay.sh` 会打 ⚠️）。
- 新增模板必做三件事：① 键进白名单（见上）；② `tests/host_templates_selftest.sh` 的 `EXPECTED_TEMPLATES` 数组加一行；③ `hosts/README.md` §3 索引表加一行（README 承诺即契约，缺文件会被判 fail）。可在 selftest 的 `DRY_EXPECT_DESC` 表加 dry-run 必含片段断言。

## 其他约定

- trace kernel 归因：跨厂商 `BASE_RULES` + 按厂商 `CATEGORY_RULE_PACKS`（`probe/trace_probe.py`）；未识别厂商合并全部包为超集。
- `README.md` 是权威使用文档（很长，按目录结构 §2 定位章节）；字段级参考 `docs/观测字段对照表.md`。
- `docs/施工图-DeepSeek部署栈支持.md` 是 DeepEP / DeepInfer / deepseek-recipe 支持的**待实施**施工图——当前仓库没有对应实现，不要把它当已完成功能引用。
