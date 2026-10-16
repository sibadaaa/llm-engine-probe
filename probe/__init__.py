"""vLLM 推理全流程探测工具包。

本包只依赖 Python 标准库，因此既可以在宿主机运行，也可以直接挂载进
容器内运行（容器里通常没有 requests / pandas 等第三方包）。
所有模块都可以独立作为脚本使用，也可以统一通过 ``python3 -m probe``
调用（见 ``probe/cli.py``）。

模块划分（对应推理链路上的不同层级）：

========================  ====================================================
模块                      观测层级
========================  ====================================================
``vendors``               厂商注册表：trace 规则包 / 报告文案 / 诊断阈值覆盖
``metrics_probe``         HTTP ``/metrics`` 层：引擎聚合指标、直方图分位数
``http_probe``            请求层：队列 → 预填充(TTFT) → 解码(ITL/TPOT) → 收尾
``log_probe``             引擎 step 层：逐 iteration 日志、调度/统计/投机解码
``trace_probe``           算子/kernel 层：torch profiler trace（Chrome trace）
``telemetry``             GPU 遥测层：各厂商 SMI → 统一 JSONL（显存/功耗/温度）
``plugin_records``        引擎内部层：stat logger 插件 JSONL → 逐 step/逐请求结构化数据
``suite``                 对比层：标准化场景矩阵 → 可跨厂商对比的 suite_results
``report``                汇总层：把以上产物合成中文 Markdown 报告
``config``                配置层：环境变量 / 命令行参数
========================  ====================================================

多厂商：NVIDIA / AMD / Hygon / Intel / Ascend 内置适配，扩展方式见
``vendors/README.md``；已有真实部署验证的环境是 Hygon DCU(DTK) + vLLM 0.18.x。
"""

__version__ = "2.0.0"

__all__ = [
    "config",
    "metrics_probe",
    "http_probe",
    "log_probe",
    "trace_probe",
    "telemetry",
    "plugin_records",
    "suite",
    "vendors",
    "report",
]
