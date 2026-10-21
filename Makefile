# ============================================================================
# llm-engine-probe 快捷入口
#   默认目标：help
#   所有脚本都可以单独调用；这里只是把常用参数固化下来。
#
#   运行方式由 RUNTIME 决定：docker（默认，容器内跑）| host（物理机/裸机直接跑）。
#   * .env 里写 RUNTIME=host 后，下面的 make 目标（up/down/status/probe/...）全部自动走 host；
#   * 想临时切一次：make up RUNTIME=host（或直接用 *-host 目标，见下）；
#   * Makefile 不硬编码 RUNTIME，未显式指定时让脚本自己读 .env 的 RUNTIME
#     （这就是 RUNTIME_ARG 的作用）。
# ============================================================================
SHELL := /usr/bin/env bash
PROJECT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PORT ?= 8100
LEVEL ?= steps
RUN_DIR ?=
# 未显式指定 RUNTIME（命令行/环境）时不要覆盖 .env：只在显式给出时注入前缀。
# 这样 .env 里写 RUNTIME=host 后，make up / make probe 等默认目标也会走物理机模式。
RUNTIME_ARG = $(if $(RUNTIME),RUNTIME=$(RUNTIME) ,)

# compose 需要根目录的 .env（compose 默认只读 compose 文件所在目录的 .env）
ifeq ($(wildcard .env),)
COMPOSE_ENV :=
else
COMPOSE_ENV := --env-file .env
endif
COMPOSE := docker compose $(COMPOSE_ENV)
# 厂商 overlay（compose-up 用；默认按 vendors/ 自动探测）
VENDOR_DETECTED := $(shell bash -c 'source scripts/lib/common.sh >/dev/null 2>&1; printf "%s" "$$VENDOR"' 2>/dev/null)
VENDOR ?= $(if $(VENDOR_DETECTED),$(VENDOR_DETECTED),generic)

.DEFAULT_GOAL := help

.PHONY: help build up up-kernel up-host down down-rm down-host restart restart-host status status-host env vendor metrics http steps watch trace trace-host pyspy gpu-sample probe probe-deep probe-host suite suite-host compare report test host-check obs-up obs-down obs-logs clean-runs compose-check compose-up compose-down

help: ## 显示全部可用目标
	@echo "llm-engine-probe 常用目标（RUNTIME=docker|host，默认 docker）："
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "典型流程（Docker）：  make build → make up → make probe → 看 runs/<时间戳>/REPORT.md"
	@echo "典型流程（物理机）：  make host-check → make up-host → make probe-host → make compare A=… B=…"
	@echo "运行方式：             .env 里 RUNTIME=host 后全部目标自动走物理机；临时切：make up RUNTIME=host 或 make up-host"

build: ## 构建探针就绪镜像（叠加在已验证的基础镜像上；仅 Docker 模式）
	$(RUNTIME_ARG)bash scripts/build_image.sh

up: ## 启动可观测的 vLLM 实例（LEVEL=steps|deep|kernel|basic|off）
	$(RUNTIME_ARG)LEVEL=$(LEVEL) bash scripts/start_vllm.sh --level $(LEVEL) $(if $(RUN_DIR),--run-dir $(RUN_DIR))

up-kernel: ## 以算子级档位启动（torch profiler + 逐层 NVTX，会关闭图模式）
	$(RUNTIME_ARG)LEVEL=kernel bash scripts/start_vllm.sh --level kernel $(if $(RUN_DIR),--run-dir $(RUN_DIR))

up-host: ## 物理机（裸机）直接启动，不依赖 Docker（前置：make host-check）
	RUNTIME=host LEVEL=$(LEVEL) bash scripts/start_vllm.sh --host --level $(LEVEL) $(if $(RUN_DIR),--run-dir $(RUN_DIR))

down: ## 停止实例（保留容器与日志）
	$(RUNTIME_ARG)bash scripts/stop_vllm.sh

down-rm: ## 停止并删除容器
	$(RUNTIME_ARG)bash scripts/stop_vllm.sh --rm

down-host: ## 停止物理机实例（保留日志；等价 RUNTIME=host scripts/stop_vllm.sh）
	RUNTIME=host bash scripts/stop_vllm.sh

restart: down up ## 重启（先停后起）

restart-host: down-host up-host ## 重启物理机实例

status: ## 查看实例状态与关键指标
	$(RUNTIME_ARG)bash scripts/status.sh --port $(PORT)

status-host: ## 物理机实例状态（运行方式/PID/端点/关键指标/日志）
	RUNTIME=host bash scripts/status.sh --port $(PORT) $(if $(RUN_DIR),--run-dir $(RUN_DIR))

env: ## 探测开关自检（哪些观测能力已生效）
	python3 -m probe env --port $(PORT) --outdir $${PROBE_OUTDIR:-runs/latest}

vendor: ## 厂商检测结果（SMI 可用性 / trace 规则包 / 镜像提示）
	python3 -m probe vendor

metrics: ## 采集一次 /metrics 快照并打印阶段表
	$(RUNTIME_ARG)bash scripts/collect_metrics.sh --port $(PORT) $(if $(RUN_DIR),--run-dir $(RUN_DIR))

http: ## 请求级探测（并发/长度用环境变量覆盖：CONCURRENCY=8 PROMPT_TOKENS=4096）
	$(RUNTIME_ARG)bash scripts/probe_http.sh --port $(PORT) $(if $(RUN_DIR),--run-dir $(RUN_DIR))

steps: ## 解析日志得到逐 step 统计摘要
	$(RUNTIME_ARG)bash scripts/watch_steps.sh --summary $(if $(RUN_DIR),--run-dir $(RUN_DIR))

watch: ## 实时跟随逐 step 日志（Ctrl-C 结束）
	$(RUNTIME_ARG)bash scripts/watch_steps.sh $(if $(RUN_DIR),--run-dir $(RUN_DIR))

trace: ## 用 /start_profile 抓一份 torch trace 并分析（需 kernel 档启动）
	$(RUNTIME_ARG)bash scripts/trace_capture.sh --port $(PORT) $(if $(RUN_DIR),--run-dir $(RUN_DIR))

trace-host: ## 物理机实例上抓 torch trace 并分析（需 up-host LEVEL=kernel）
	RUNTIME=host bash scripts/trace_capture.sh --port $(PORT) $(if $(RUN_DIR),--run-dir $(RUN_DIR))

pyspy: ## 抓 Python 栈快照（定位 CPU 侧瓶颈）
	$(RUNTIME_ARG)bash scripts/pyspy.sh --mode dump $(if $(RUN_DIR),--run-dir $(RUN_DIR))

gpu-sample: ## 采样 GPU 遥测 60 秒（显存/功耗/温度，各厂商 SMI）
	$(RUNTIME_ARG)bash scripts/gpu_sample.sh --duration 60 $(if $(RUN_DIR),--run-dir $(RUN_DIR))

probe: ## 一键全链路探测并生成报告
	$(RUNTIME_ARG)bash scripts/probe_all.sh --port $(PORT) --level $(LEVEL) $(if $(RUN_DIR),--run-dir $(RUN_DIR))

probe-deep: ## 全链路 + GPU 遥测 + py-spy
	$(RUNTIME_ARG)bash scripts/probe_all.sh --port $(PORT) --level $(LEVEL) --with-gpu-sample --with-pyspy $(if $(RUN_DIR),--run-dir $(RUN_DIR))

probe-host: ## 物理机实例一键全链路（RUNTIME=host probe）
	RUNTIME=host bash scripts/probe_all.sh --port $(PORT) --level $(LEVEL) $(if $(RUN_DIR),--run-dir $(RUN_DIR))

suite: ## 标准化对比套件（8 场景 → suite_results.json，可跨厂商对比）
	$(RUNTIME_ARG)bash scripts/run_suite.sh --port $(PORT) $(if $(RUN_DIR),--run-dir $(RUN_DIR))

suite-host: ## 物理机实例跑标准化对比套件
	RUNTIME=host bash scripts/run_suite.sh --port $(PORT) $(if $(RUN_DIR),--run-dir $(RUN_DIR))

compare: ## 对比两次套件结果：make compare A=runs/a/suite_results.json B=runs/b/suite_results.json
	python3 -m probe compare --a "$(A)" --b "$(B)" $(if $(OUT),--out $(OUT))

report: ## 生成/刷新中文 Markdown 报告
	$(RUNTIME_ARG)bash scripts/report.sh $(if $(RUN_DIR),--run-dir $(RUN_DIR))

test: ## 跑单元测试 + mock 服务器端到端 + host 模式自测
	bash tests/run_tests.sh

host-check: ## 物理机环境预检（GPU 驱动/Python/vLLM/系统限制；JSON=1 输出 JSON）
	$(RUNTIME_ARG)bash scripts/host_check.sh $(if $(JSON),--json) $(if $(RUN_DIR),--outdir $(RUN_DIR))

obs-up: ## 启动观测栈（Prometheus + Grafana + OTel Collector）
	$(COMPOSE) -f docker/docker-compose.yml -f docker/docker-compose.obs.yml --profile obs up -d

obs-down: ## 停止观测栈
	$(COMPOSE) -f docker/docker-compose.yml -f docker/docker-compose.obs.yml --profile obs down

obs-logs: ## 观测栈日志
	$(COMPOSE) -f docker/docker-compose.yml -f docker/docker-compose.obs.yml --profile obs logs -f --tail 50

compose-check: ## 校验 compose 文件语法（含全部厂商 overlay）
	$(COMPOSE) -f docker/docker-compose.yml config -q && echo "docker-compose.yml OK"
	$(COMPOSE) -f docker/docker-compose.yml -f docker/docker-compose.obs.yml --profile obs config -q && echo "obs 叠加 OK"
	@for v in hygon nvidia amd intel ascend; do \
	  $(COMPOSE) -f docker/docker-compose.yml -f docker/docker-compose.$$v.yml config -q \
	    && echo "$$v overlay OK" || exit 1; \
	done

compose-up: ## 用 compose 起容器壳（VENDOR 自动探测，可覆盖：make compose-up VENDOR=nvidia）
	$(COMPOSE) -f docker/docker-compose.yml -f docker/docker-compose.$(VENDOR).yml up -d vllm
	@echo "容器壳已起；下一步：EXEC_CONTAINER=vllm-probe-c scripts/start_vllm.sh --exec"

compose-down: ## 停掉 compose 容器壳
	$(COMPOSE) -f docker/docker-compose.yml -f docker/docker-compose.$(VENDOR).yml down

clean-runs: ## 删除运行产物（保留最近一次）
	@ls -1d runs/*/ 2>/dev/null | sort | head -n -1 | xargs -r rm -rf
	@echo "已清理旧的 runs/ 目录"
