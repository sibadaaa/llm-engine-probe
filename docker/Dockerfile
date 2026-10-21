# 探针就绪镜像：在已验证的 vLLM 基础镜像上叠加探测工具。
#
# 为什么需要它：基础镜像里通常已经有 vLLM / torch / py-spy，
# 但没有本项目的 probe 工具包。本镜像把工具包放到 /opt/vllm-probe，
# 于是「容器内」也能直接 `python3 -m probe ...`。
#
# 多厂商：基础镜像由 BASE_IMAGE 决定（NVIDIA/AMD/Hygon/Intel/Ascend 均可）：
#   scripts/build_image.sh                                   # 用 .env 的 IMAGE 作为基础镜像
#   BASE_IMAGE=vllm/vllm-openai:latest scripts/build_image.sh --tag vllm-probe:cuda
#   BASE_IMAGE=rocm/vllm:latest scripts/build_image.sh --tag vllm-probe:rocm
#   BASE_IMAGE=quay.io/ascend/vllm-ascend:latest scripts/build_image.sh --tag vllm-probe:ascend
#   BASE_IMAGE=z100-s9-snap-s14 scripts/build_image.sh --tag vllm-probe:dtk
#
# 构建：（用法见上）

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="llm-engine-probe" \
      org.opencontainers.image.description="vLLM 推理全流程探测工具镜像（多厂商：NVIDIA/AMD/Hygon/Intel/Ascend）" \
      org.opencontainers.image.source="llm-engine-probe" \
      vllm.probe.base_image="${BASE_IMAGE}"

ARG PIP_PACKAGES=""
ARG PIP_INDEX_URL=""
# 置 1 时把 stat logger 插件装进镜像（也可以运行时再装：start_vllm.sh 的 PROBE_PLUGIN=1）
ARG INSTALL_PROBE_PLUGIN=0

# 探针工具包（纯标准库实现，复制即可用，不依赖 pip）
COPY probe /opt/vllm-probe/probe
COPY plugin_probe /opt/vllm-probe/plugin_probe
COPY scripts /opt/vllm-probe/scripts
COPY vendors /opt/vllm-probe/vendors
COPY .env.example /opt/vllm-probe/.env.example
RUN chmod +x /opt/vllm-probe/scripts/*.sh /opt/vllm-probe/vendors/*.sh 2>/dev/null || true

ENV PYTHONPATH=/opt/vllm-probe \
    PYTHONUNBUFFERED=1
WORKDIR /opt/vllm-probe

# 可选：额外 pip 包；py-spy 缺失时尝试补装（无网络时忽略失败，不影响镜像可用性）
RUN set -eux; \
    if [ -n "$PIP_PACKAGES" ]; then \
      if [ -n "$PIP_INDEX_URL" ]; then \
        pip install --no-cache-dir -i "$PIP_INDEX_URL" $PIP_PACKAGES; \
      else \
        pip install --no-cache-dir $PIP_PACKAGES; \
      fi; \
    fi; \
    if ! command -v py-spy >/dev/null 2>&1; then \
      if [ -n "$PIP_INDEX_URL" ]; then \
        pip install --no-cache-dir -i "$PIP_INDEX_URL" py-spy || true; \
      else \
        pip install --no-cache-dir py-spy || true; \
      fi; \
    fi; \
    if [ "$INSTALL_PROBE_PLUGIN" = "1" ]; then \
      pip install --no-cache-dir --no-deps --no-build-isolation /opt/vllm-probe/plugin_probe; \
    fi; \
    python3 -c "import sys; sys.path.insert(0, '/opt/vllm-probe'); import probe; print('probe toolchain', probe.__version__)"

# 容器默认只做「壳」，vLLM 由 scripts/start_vllm.sh（docker run/exec）或
# docker-compose 里的同名服务拉起，便于反复开关服务而不重启容器。
CMD ["bash", "-lc", "sleep infinity"]
