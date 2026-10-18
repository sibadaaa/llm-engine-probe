#!/usr/bin/env bash
# 构建"探针就绪"镜像：在已验证的 vLLM 基础镜像上叠加探测工具与依赖。
#
# 用法：
#   scripts/build_image.sh                      # 默认 TAG=vllm-probe:latest，基础镜像取 .env 的 IMAGE
#   BASE_IMAGE=z100-s9-snap-s14 scripts/build_image.sh
#   BASE_IMAGE=vllm/vllm-openai:latest scripts/build_image.sh --tag vllm-probe:cuda
#   BASE_IMAGE=quay.io/ascend/vllm-ascend:latest scripts/build_image.sh --tag vllm-probe:ascend
#   scripts/build_image.sh --tag my/probe:v1 --no-cache
#
# 说明：基础镜像里通常已经带了 vLLM / torch / py-spy，本镜像只是：
#   1) 把 probe 工具包复制到 /opt/vllm-probe（容器内也能直接跑）
#   2) 可选安装额外 pip 包（PIP_PACKAGES，默认空）
#   3) 打上版本/来源标签，方便回溯

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

TAG="${TAG:-vllm-probe:latest}"
NO_CACHE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="${2:?}"; shift 2 ;;
    --no-cache) NO_CACHE="--no-cache"; shift ;;
    --base) BASE_IMAGE="${2:?}"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

# 物理机模式不使用镜像：给出明确指引，避免用户误以为要构建/拉取镜像。
if runtime_is_host; then
  die "物理机模式不需要镜像；直接 scripts/start_vllm.sh --host（先 scripts/host_check.sh 预检）"
fi

BASE_IMAGE="${BASE_IMAGE:-$IMAGE}"
require_cmd docker
[[ -f "$PROJECT_ROOT/docker/Dockerfile" ]] || die "找不到 docker/Dockerfile"
[[ -n "$BASE_IMAGE" ]] || die "基础镜像为空：请在 .env 里设置 IMAGE，或用 BASE_IMAGE=... 指定（厂商 $VENDOR 提示：$VENDOR_IMAGE_HINT）"

if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  warn "本地没有基础镜像 $BASE_IMAGE，尝试拉取（内网 harbor 需要先 docker login）"
  docker pull "$BASE_IMAGE" || die "拉取失败；请检查镜像名或先手动 docker pull"
fi

log "构建 $TAG（BASE_IMAGE=$BASE_IMAGE, 厂商=$VENDOR, PIP_PACKAGES=${PIP_PACKAGES:-<空>}）"
docker build $NO_CACHE \
  -f "$PROJECT_ROOT/docker/Dockerfile" \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "PIP_PACKAGES=${PIP_PACKAGES:-}" \
  --build-arg "PIP_INDEX_URL=${PIP_INDEX_URL:-}" \
  -t "$TAG" \
  "$PROJECT_ROOT"

ok "构建完成：$TAG"
echo
echo "  之后在 .env 里设置："
echo "    IMAGE=$TAG"
echo "  然后：scripts/start_vllm.sh"
