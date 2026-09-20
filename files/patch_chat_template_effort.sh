#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# patch_chat_template_effort.sh — accept the reasoning_effort values clients
# actually send.
#
# The checkpoint's template takes xhigh (its default), medium and low, and
# raises on anything else. "high" is what an OpenAI-compatible client sends, so
# it comes back as HTTP 400:
#
#   Unexpected reasoning effort high. Supported types are xhigh (default),
#   medium, and low.
#
# Rewrites one line to map high/max -> xhigh and minimal -> low, leaving every
# value the template already accepted untouched. Writes the copy beside the
# recipe; the checkpoint is never modified. Idea and the exact substitution from
# blazux/qwen3.8-Flash-DGX (scripts/serve.sh, EFFORT_ALIAS, Apache-2.0).
#
# Skips itself, loudly, if the template is not the one it expects -- better to
# serve a 400 on an unusual effort than to silently ship a template nobody
# checked.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAPSHOT="${1:?usage: patch_chat_template_effort.sh <snapshot-dir>}"
SRC="$SNAPSHOT/chat_template.jinja"
OUT="$SCRIPT_DIR/chat_template_effort.jinja"

OLD="{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}"
NEW="{%- set resolved_reasoning_effort = {'high': 'xhigh', 'max': 'xhigh', 'minimal': 'low'}.get(reasoning_effort|default('xhigh'), reasoning_effort|default('xhigh')) %}"

warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }

# A copy from an earlier run must not survive a decision not to rewrite: start.sh
# mounts whatever is here, and a stale template outliving the checkpoint that
# produced it is exactly the kind of thing nobody looks at again.
rm -f "$OUT"

[[ -f "$SRC" ]] || { warn "chat template not found at $SRC; serving stock"; exit 0; }

if ! grep -qF "Supported types are xhigh (default), medium, and low." "$SRC"; then
    warn "chat template does not carry the known effort check; serving stock"
    exit 0
fi
# Exactly once, or we do not know what we are editing.
if [[ "$(grep -cF "$OLD" "$SRC")" != "1" ]]; then
    warn "effort line is not the expected one (found $(grep -cF "$OLD" "$SRC")); serving stock"
    exit 0
fi

python3 - "$SRC" "$OUT" "$OLD" "$NEW" <<'PY'
import sys
src, out, old, new = sys.argv[1:5]
text = open(src, encoding="utf-8").read()
assert text.count(old) == 1
head = ("{#- files/patch_chat_template_effort.sh: effort line rewritten, "
        "high/max -> xhigh, minimal -> low. The rest is the checkpoint's. -#}")
open(out, "w", encoding="utf-8").write(head + text.replace(old, new))
PY
echo "ok $OUT"
