#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# patch_qwen_tool_parser.sh — stop the Qwen parser eating answers that mention
# a tool marker.
#
# The parser enters a tool preamble the moment it sees the opener, so a reply
# that merely quotes `<tool_call>` -- reviewing agent code, explaining the Qwen
# format, a fenced example -- loses that text and everything after it. It fails
# silently: finish_reason is "stop" and no error is raised. Reproduced on this
# box: a 339-character answer came back as "La etiqueta ".
#
# Fix is vllm-project/vllm#56661, still open upstream. The two patches here are
# blazux's adaptation of it to this image (Apache-2.0), vendored under
# files/patches/. Run before start.sh mounts the results.
#
#   ./files/patch_qwen_tool_parser.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
PKG="/usr/local/lib/python3.12/dist-packages"
OUT="$SCRIPT_DIR/parser"
FILES=(vllm/parser/engine/parser_engine_config.py
       vllm/parser/engine/streaming_parser_engine.py
       vllm/parser/nemotron_v3.py
       vllm/parser/qwen3.py)

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

# Originals are kept beside the patched copies so a re-run starts from a clean
# base; the patches are --forward, but applying them twice to the same file
# would otherwise depend on that flag rather than on state we control.
if [[ ! -d "$OUT/orig" ]]; then
    info "Extracting parser sources from image..."
    tmp=$(docker create "$IMAGE" /bin/true)
    for f in "${FILES[@]}"; do
        mkdir -p "$OUT/orig/$(dirname "$f")"
        docker cp "$tmp:$PKG/$f" "$OUT/orig/$f"
    done
    docker rm "$tmp" >/dev/null 2>&1
fi

rm -rf "$OUT/vllm"
mkdir -p "$OUT"
cp -a "$OUT/orig/vllm" "$OUT/vllm"

info "Applying the tool-marker patches..."
docker run --rm -v "$OUT:/w" -v "$SCRIPT_DIR/patches:/p:ro" \
    --entrypoint bash "$IMAGE" -c '
set -euo pipefail
cd /w
for f in qwen-tool-preamble qwen-tool-marker-guard; do
    patch -p1 --batch --forward --fuzz=0 < "/p/$f.patch"
done
chmod 644 $(find vllm -type f)'

for f in "${FILES[@]}"; do
    [[ -f "$OUT/$f" ]] || err "missing after patch: $f"
    python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$OUT/$f" \
        || err "patched file does not parse: $f"
done
info "ok: $OUT/vllm (4 files)"
