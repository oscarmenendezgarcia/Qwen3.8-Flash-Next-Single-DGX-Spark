#!/usr/bin/env bash
# Install and enable flashnext-vllm.service.
#
#     sudo ./deploy/install-unit.sh
#
# The tracked unit carries placeholders instead of absolute paths, because
# systemd needs absolute paths and they cannot be right for two people at once.
# This script fills them in from wherever the repo actually lives and from the
# user who owns it, so the unit is portable and the installed copy is not.
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "Run with sudo."; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(dirname "$HERE")"
RUN_USER="$(stat -c %U "$RECIPE_DIR")"
HOME_DIR="$(getent passwd "$RUN_USER" | cut -d: -f6)"
DST=/etc/systemd/system/flashnext-vllm.service

[[ -x "$RECIPE_DIR/start.sh" ]] || { echo "no start.sh in $RECIPE_DIR"; exit 1; }
echo "  recipe : $RECIPE_DIR"
echo "  user   : $RUN_USER  (home $HOME_DIR)"

sed -e "s|@RECIPE_DIR@|$RECIPE_DIR|g" \
    -e "s|@HOME_DIR@|$HOME_DIR|g" \
    -e "s|@RUN_USER@|$RUN_USER|g" \
    "$HERE/flashnext-vllm.service" > "$DST"
chmod 0644 "$DST"
grep -q "@" "$DST" && { echo "placeholders left unfilled in $DST"; exit 1; }
echo "[ OK ] written -> $DST"

systemctl daemon-reload && echo "[ OK ] daemon-reload"
systemctl enable flashnext-vllm.service && echo "[ OK ] enabled"

# start.sh returns early when the container is already up, so this syncs
# systemd's view rather than restarting a live server.
systemctl start flashnext-vllm.service && echo "[ OK ] started"

echo
systemctl is-active  flashnext-vllm.service | sed 's/^/  active:  /'
systemctl is-enabled flashnext-vllm.service | sed 's/^/  enabled: /'
PORT="$(grep -oP '^PORT=\K[0-9]+' "$RECIPE_DIR/.env" 2>/dev/null || echo 8888)"
printf '  health:  %s (port %s)\n' "$(curl -s -m 15 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" "$PORT"
