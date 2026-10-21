# vendors/ · 厂商适配层

本目录是 llm-engine-probe 的**厂商抽象层**（Shell 端）。每个文件描述一家
显卡生态在「容器启动 / 设备直通 / 环境注入 / 遥测命令」上的差异，Python 端
的分析适配（trace 规则包、报告文案、遥测解析）在 `probe/vendors.py` 注册。

内置适配：`nvidia`、`amd`、`hygon`、`intel`、`ascend`，另有：

* `generic` —— 无设备直通的兜底（CI / 无 GPU 环境，`VENDOR=generic` 强制选择）；
* `_template.sh` —— 新增厂商的模板（复制改名后填写）。

## 使用

```bash
# 自动探测（默认）：VENDOR=auto，在 .env 里一般无需设置
VENDOR=auto scripts/start_vllm.sh --dry-run

# 显式指定（覆盖探测结果）
VENDOR=nvidia make up

# 查看当前检测结果与能力矩阵
python3 -m probe vendor
```

自动探测顺序：`ascend → nvidia → hygon → amd → intel`，命中条件为
profile 里的 `VENDOR_DETECT_SMI`（命令在 PATH）或 `VENDOR_DETECT_DEVICE`
（设备节点存在），都不命中则回退 `generic`。

## 适配一家新厂商（三步）

1. **Shell 端**：`cp vendors/_template.sh vendors/<id>.sh`，填写全部必填项
   （设备直通参数、可见性环境变量、SMI 命令、探测元数据）。每个变量的含义
   见模板内注释；动态设备数量用 `vendor_dynamic_docker_args()`。
2. **Python 端**：在 `probe/vendors.py` 的 `VENDOR_REGISTRY` 注册同 id 的
   `VendorProfile`（label / smi_cmd / trace_pack / 报告文案）；若需要专属
   kernel 分类规则，在 `probe/trace_probe.py` 的 `CATEGORY_RULE_PACKS` 中
   增加同名规则包。
3. **遥测**：若厂商 SMI 输出格式不在现有 backend 覆盖范围内，在
   `probe/telemetry.py` 中新增 Backend（参考 `HyBackend`），并在
   `telemetry.parse_smi_output()` 中注册。

完成后运行：

```bash
bash tests/run_tests.sh          # 含厂商注册表一致性守卫（shell ↔ Python）
python3 -m probe vendor --vendor <id>   # 人工核对展示信息
VENDOR=<id> scripts/start_vllm.sh --dry-run   # 核对 docker 参数
```

## 一致性约定

* `probe/vendors.py` 注册表键集合必须与 `vendors/*.sh`（去掉 `_` 前缀文件）
  完全一致，由 `tests/test_vendors.py` 强制校验；
* profile 文件在子 shell 中会被 source 用于读取探测元数据，
  **必须无副作用**（只允许变量赋值/函数定义）；
* 所有 `VENDOR_*` 变量在 `set -u` 下都可能为空，读取时用 `${VAR:-}` 风格。
