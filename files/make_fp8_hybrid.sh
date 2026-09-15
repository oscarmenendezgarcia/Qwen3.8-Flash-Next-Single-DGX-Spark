#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# make_fp8_hybrid.sh — convert a checkpoint's dense side layers to blockwise fp8.
#
# NVIDIA's checkpoint leaves the GDN in/out projections, the QSA q/k/v/o and the
# shared experts in bf16 (9.99 GiB against the Mia mirror's 3.69). Those layers
# are read in full on every decoded token, so they dominate decode bandwidth
# while the routed experts stay sparse and 4-bit. Converting them recovers ~12%
# of decode at no measured quality cost -- see
# docs/checkpoint-quantization-and-spanish-2026-09-14.md section 6.
#
# Writes a sibling snapshot "<rev>-fp8hybrid"; the published one is untouched.
# Point start.sh at it with TP1_SNAPSHOT (refs/main always wins the automatic
# resolution, so a variant beside it can never be picked on its own).
#
#   ./files/make_fp8_hybrid.sh nvidia/Qwen3.8-Flash-Next-NVFP4
#   TP1_MODEL_ID=... TP1_SNAPSHOT=<rev>-fp8hybrid VLLM_FP8_HYBRID=1 ./start.sh
#
# Serving it also needs VLLM_FP8_HYBRID=1 in EXTRA_DOCKER_ARGS: without the shim
# that files/patch_modelopt_mxfp8.py appends, ModelOpt sends these layers down
# the excluded bf16 path and they do not load.
set -euo pipefail

MODEL_ID="${1:-nvidia/Qwen3.8-Flash-Next-NVFP4}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

REPO_DIR="$HF_CACHE/hub/models--${MODEL_ID//\//--}"
[[ -d "$REPO_DIR/snapshots" ]] || err "not downloaded: $REPO_DIR"

REV="$(cat "$REPO_DIR/refs/main" 2>/dev/null || true)"
[[ -n "$REV" && -d "$REPO_DIR/snapshots/$REV" ]] \
    || REV="$(ls -1 "$REPO_DIR/snapshots" | grep -v -- '-fp8hybrid' | head -1)"
[[ -n "$REV" ]] || err "no source snapshot under $REPO_DIR/snapshots"

SRC="$REPO_DIR/snapshots/$REV"
DST="$REPO_DIR/snapshots/${REV}-fp8hybrid"
info "source: $REV"
info "target: ${REV}-fp8hybrid"

if [[ -f "$DST/.prepared" ]]; then
    info "already prepared, nothing to do"; exit 0
fi

# cp -a keeps the entries as symlinks into ../../blobs, so only the shards the
# converter rewrites cost disk (~71 GiB here, not another full checkpoint).
rm -rf "$DST"
cp -a "$SRC" "$DST"
# The index is rewritten in place, so it must be a real file, not a blob symlink.
cp --remove-destination "$(readlink -f "$DST/model.safetensors.index.json")" \
   "$DST/model.safetensors.index.json"
chmod u+w "$DST/model.safetensors.index.json"

info "converting (one shard in RAM at a time; capped at 14g)..."
docker run --rm --name fp8convert --memory 14g --cpus 8 \
    -v "$HF_CACHE:/hf" -v "$SCRIPT_DIR:/tools:ro" \
    --entrypoint python3 "$IMAGE" \
    -u /tools/fp8_convert.py "/hf/hub/models--${MODEL_ID//\//--}/snapshots/${REV}-fp8hybrid"

# Two clean-ups the converter does not do, both of which break the launch:
#
#   1. It leaves the originals as "<shard>.bf16.bak". Here those are symlinks
#      into ../../blobs, and start.sh sizes the snapshot with `du -L`, which
#      follows them -- the checkpoint measures 146 GiB instead of 121 and the
#      memory guard aborts before loading anything.
#   2. The shards it writes are owned by root (the converter runs in the
#      container) and are not world-readable, so the serving container's own
#      loader cannot open them.
docker run --rm -v "$HF_CACHE:/hf" --entrypoint bash "$IMAGE" -c "
  cd '/hf/hub/models--${MODEL_ID//\//--}/snapshots/${REV}-fp8hybrid'
  rm -f *.bf16.bak
  chmod 644 *.safetensors model.safetensors.index.json
  touch .prepared"

n=$(python3 -c "import json;print(sum(1 for k in json.load(open('$DST/model.safetensors.index.json'))['weight_map'] if k.endswith('weight_scale_inv')))")
[[ "$n" -gt 0 ]] || err "conversion produced no blockwise-fp8 tensors"
info "done: $n weight_scale_inv tensors, $(du -sh -L "$DST" | cut -f1) snapshot"

# PLE_GIB is the one value start.sh cannot infer, and getting it wrong is not a
# warning: start.sh subtracts it from the checkpoint to size the weights, so the
# stock 26.82 against a 47.68 GiB table makes the weights look 21 GiB larger than
# they are and the memory guard aborts before loading anything. Print it measured
# rather than leave it to be looked up.
# Size the n-gram embedding, not "everything with ple in the name". Two ways to
# get this wrong, both verified here against the packed table start.sh builds
# (47.68 GiB): sizing the file that holds it adds the MTP head NVIDIA ships in
# the same shard (50.03), and summing every PLE tensor adds ple.key_proj and
# ple.value_proj, which stay on the GPU and are not offloaded (47.75).
PLE_MEASURED=$(python3 - "$DST" <<'PLEPY'
import json, os, struct, sys
snap = sys.argv[1]
idx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
total = 0
for shard in sorted({f for k, f in idx.items() if "ngram_embedding" in k.lower()}):
    with open(f"{snap}/{shard}", "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    for name, meta in header.items():
        if name != "__metadata__" and "ngram_embedding" in name.lower():
            a, b = meta["data_offsets"]
            total += b - a
print(f"{total / 1073741824:.2f}")
PLEPY
)

echo
info "Put these in .env (see .env.gb10.sample for the rest):"
echo
echo "    TP1_MODEL_ID=$MODEL_ID"
echo "    TP1_SNAPSHOT=${REV}-fp8hybrid"
echo "    PLE_GIB=$PLE_MEASURED"
echo "    EXTRA_DOCKER_ARGS=\"-e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_FP8_HYBRID=1\""
echo
info "A bigger PLE table also wants more host reserve for its page cache:"
info "HOST_RESERVE_GIB=28 was measured comfortable here for a 47.68 GiB table"
info "(residency 3.7-5.1 GiB while serving). Raise it if the table is larger."
