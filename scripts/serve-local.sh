#!/usr/bin/env sh
# Serve the smoke-test model (configs/smoke-local.yaml) with llama.cpp, OpenAI-compatible on :8080.
# Downloads the pinned GGUF on first use. Thinking is off, as for every model in the benchmark.
set -eu
REPO=ggml-org/Qwen3-4B-GGUF
REVISION=2f3b082b1356a6123f7ed71e65aea340da25d53c
FILE=Qwen3-4B-Q4_K_M.gguf
SHA256=ab27b9bfa375a178d6cba48f3ad892b94b7739659dcc7aae8058ce0ffed6b328
MODEL="data/models/$FILE"

if [ ! -f "$MODEL" ]; then
  mkdir -p data/models
  curl -L --fail -o "$MODEL.part" "https://huggingface.co/$REPO/resolve/$REVISION/$FILE"
  mv "$MODEL.part" "$MODEL"
fi
echo "$SHA256  $MODEL" | shasum -a 256 -c -

exec llama-server -m "$MODEL" --alias qwen3-4b-q4_k_m --host 127.0.0.1 --port 8080 \
  --ctx-size 16384 --parallel 4 --kv-unified --reasoning off
