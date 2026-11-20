#!/usr/bin/env bash
# Serve thinkingmachines/Inkling-2-Mini on one RTX PRO 6000 Blackwell (SM120) with vLLM,
# with the DSpark speculator bundled in the checkpoint.
# The vllm/vllm-openai:inkling-2-mini image ships its vllm/_C kernels for SM100 only;
# the one file in ../patch is bind-mounted read-only over the image so the KV-cache
# write uses vLLM's Triton kernel on SM12x (see README "The patch").
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
P="$REPO/patch/vllm"
D=/usr/local/lib/python3.14/dist-packages/vllm
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e}  # tag inkling-2-mini
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
PORT=${PORT:-8000}
GPU=${GPU:-0}

docker rm -f inkling-2-mini >/dev/null 2>&1 || true
docker run -d --name inkling-2-mini --gpus "\"device=$GPU\"" --ipc=host -p "$PORT:8000" \
  -e HF_HUB_OFFLINE=1 \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -v "$P/v1/attention/backends/fa_utils.py:$D/v1/attention/backends/fa_utils.py:ro" \
  --entrypoint vllm "$IMAGE" \
  serve thinkingmachines/Inkling-2-Mini --revision 1fb4134b9a74f9681054f4ef82e4bf1f458a2337 \
    --host 0.0.0.0 --port 8000 --served-model-name inkling-2-mini \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
    --max-model-len 524288 \
    --reasoning-parser inkling \
    --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
    --enable-prompt-tokens-details "$@"

echo "Started. Follow logs with: docker logs -f inkling-2-mini   (ready in ~4 min)"
