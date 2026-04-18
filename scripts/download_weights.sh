#!/usr/bin/env bash
# Download TRIBE v2 model weights via the tribe-server download endpoints.
# tribe-server must be running at localhost:8081 before running this script.
# best.safetensors (FmriEncoder) is NOT downloaded via the API — it requires
# running: cd ../tribe-playground && python3 convert_ckpt.py

set -euo pipefail

BASE="http://localhost:8081"

check_server() {
    curl -sf "$BASE/health" > /dev/null || {
        echo "ERROR: tribe-server not running at $BASE"
        echo "Start it with: cd ../tribe-playground && cargo run --release -p tribe-server"
        exit 1
    }
}

start_download() {
    local model="$1"
    echo "Starting download: $model" >&2
    local job_id
    job_id=$(curl -sf -X POST "$BASE/api/download/$model" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
    echo "  Job ID: $job_id" >&2
    echo "  Poll: curl $BASE/api/jobs/$job_id" >&2
    echo "$job_id"
}

poll_until_done() {
    local job_id="$1"
    local model="$2"
    echo "Waiting for $model (job $job_id)..."
    while true; do
        local status
        status=$(curl -sf "$BASE/api/jobs/$job_id" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
        echo "  [$model] status: $status"
        if [ "$status" = "done" ]; then break; fi
        if [ "$status" = "failed" ]; then echo "ERROR: $model download failed"; exit 1; fi
        sleep 10
    done
    echo "  $model: done"
}

check_server

echo "=== Step 1: Convert checkpoint to best.safetensors ==="
echo "Run manually if not already done:"
echo "  cd ../tribe-playground && python3 convert_ckpt.py"
echo ""

echo "=== Step 2: Download LLaMA text encoder (~6GB) ==="
llama_job=$(start_download "llama")
poll_until_done "$llama_job" "llama"

echo "=== Step 3: Download Wav2Vec2Bert audio encoder ==="
w2v_job=$(start_download "wav2vec")
poll_until_done "$w2v_job" "wav2vec"

echo "=== Step 4: Download CLIP visual encoder ==="
clip_job=$(start_download "clip")
poll_until_done "$clip_job" "clip"

echo ""
echo "All weights downloaded. Restart tribe-server to load them."
echo "Then re-run the sweep: cargo run -p sweep"
