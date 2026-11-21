#!/usr/bin/env bash
# Serve thinkingmachines/Inkling-2-Mini on one RTX PRO 6000 Blackwell (SM120) with vLLM,
# with the DSpark speculator bundled in the checkpoint.
# The vllm/vllm-openai:inkling-2-mini image ships its vllm/_C kernels for SM100 only,
# so one file in ../patch moves the KV-cache write to vLLM's Triton kernel on SM12x.
# The other files are the V1 speed-ups and the TP2 embedding fix (see README
# "The patch"). All nine are bind-mounted read-only over the image.
#   ./run.sh           one GPU (GPU=0|1), 512K context
#   TP=2 ./run.sh      both GPUs, 1M context
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
P="$REPO/patch/vllm"
D=/usr/local/lib/python3.14/dist-packages/vllm
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e}  # tag inkling-2-mini
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
PORT=${PORT:-8000}
GPU=${GPU:-0}
TP=${TP:-1}
# vLLM defaults to 1024 concurrent requests on a card this size and preallocates
# block tables for all of them; see README "The request limit".
MAX_SEQS=${MAX_SEQS:-64}
if [ "$TP" = 2 ]; then
  # The custom all-reduce is in the SM100-only _C extension; NCCL works.
  GPUS=all; CTX=1048576; UTIL=0.95; EXTRA=(--disable-custom-all-reduce)
else
  GPUS="\"device=$GPU\""; CTX=524288; UTIL=0.95; EXTRA=()
fi

MOUNTS=()
for f in v1/attention/backends/fa_utils.py model_executor/layers/vocab_parallel_embedding.py \
         v1/worker/extensible_kv_cache.py \
         models/inkling2/nvidia/dspark.py models/inkling2/nvidia/model.py \
         models/inkling2/nvidia/loop.py models/inkling2/nvidia/sconv.py \
         models/inkling2/nvidia/ops/softcap.py models/inkling2/nvidia/ops/loop_fused.py; do
  MOUNTS+=(-v "$P/$f:$D/$f:ro")
done

docker rm -f inkling-2-mini >/dev/null 2>&1 || true
docker run -d --name inkling-2-mini --gpus "$GPUS" --ipc=host -p "$PORT:8000" \
  -e HF_HUB_OFFLINE=1 \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  "${MOUNTS[@]}" \
  --entrypoint vllm "$IMAGE" \
  serve thinkingmachines/Inkling-2-Mini --revision 1fb4134b9a74f9681054f4ef82e4bf1f458a2337 \
    --host 0.0.0.0 --port 8000 --served-model-name inkling-2-mini \
    --tensor-parallel-size "$TP" --gpu-memory-utilization "$UTIL" \
    --max-model-len "$CTX" --max-num-seqs "$MAX_SEQS" \
    --reasoning-parser inkling \
    --speculative-config '{"method":"dspark","num_speculative_tokens":4,"draft_sample_method":"probabilistic"}' \
    --enable-prompt-tokens-details "${EXTRA[@]}" "$@"

echo "Started. Follow logs with: docker logs -f inkling-2-mini   (ready in ~4 min)"
