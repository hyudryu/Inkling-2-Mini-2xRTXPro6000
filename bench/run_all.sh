#!/usr/bin/env bash
# Runs all benchmark phases while logging GPU power once per second.
#   POOL=<KV pool tokens from the server log> OUT=results/<name> ./run_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL=${MODEL:-inkling-2-mini}
POOL=${POOL:?set POOL to the "GPU KV cache size" from the server log}
OUT=${OUT:?set OUT to a results directory, e.g. results/v1}
PREFILL_SIZES=${PREFILL_SIZES:-1024,4096,8192,16384,32768,65536,131072}
CONTEXTS=${CONTEXTS:-8192,16384,32768,65536,131072}
LEVELS=${LEVELS:-1,2,4,8,16,32}
mkdir -p "$HERE/$OUT" && cd "$HERE/$OUT"
B="python3 -I $HERE/bench.py --model $MODEL"
nvidia-smi -i ${GPU:-0} --query-gpu=timestamp,index,power.draw,temperature.gpu,clocks.sm --format=csv,noheader,nounits -lms 1000 > power.csv &
PW=$!
trap 'kill $PW 2>/dev/null' EXIT
: > phases.txt
# JIT warm-up so the prefill sweeps measure warm kernels (cold numbers are a startup artifact)
echo "warmup start $(date +%s)" >> phases.txt
$B --only prefill --prefill-sizes 4096,65536 --seed-offset 1 --out results-warmup.json
echo "warmup end $(date +%s)" >> phases.txt
for r in 1 2 3 4 5; do
  echo "prefill$r start $(date +%s)" >> phases.txt
  $B --only prefill --prefill-sizes "$PREFILL_SIZES" --seed-offset $((r + 1)) --out results-prefill-warm$r.json
  echo "prefill$r end $(date +%s)" >> phases.txt
done
echo "decode start $(date +%s)" >> phases.txt
curl -s "${URL:-http://localhost:8000}/metrics" | grep "^vllm:spec_decode" > spec-before-decode.txt || true
$B --only decode --levels "$LEVELS" --out results-decode.json
curl -s "${URL:-http://localhost:8000}/metrics" | grep "^vllm:spec_decode" > spec-after-decode.txt || true
echo "decode end $(date +%s)" >> phases.txt
echo "context start $(date +%s)" >> phases.txt
$B --only context --levels "$LEVELS" --contexts "$CONTEXTS" --pool-tokens "$POOL" --seed-offset 7 --out results-context.json
echo "context end $(date +%s)" >> phases.txt
