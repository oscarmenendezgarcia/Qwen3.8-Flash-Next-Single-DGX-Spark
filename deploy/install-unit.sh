#!/usr/bin/env bash
# Install and enable flashnext-vllm.service. Run with sudo:
#     sudo ~/flashnext-recipe/deploy/install-unit.sh
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "Run with sudo."; exit 1; }

SRC=/home/oscar/flashnext-recipe/deploy/flashnext-vllm.service
DST=/etc/systemd/system/flashnext-vllm.service

install -m 0644 "$SRC" "$DST"
echo "[ OK ] copied -> $DST"
systemctl daemon-reload
echo "[ OK ] daemon-reload"
systemctl enable flashnext-vllm.service
echo "[ OK ] enabled (will start on boot)"

# start.sh returns early when the container is already up, so this only syncs
# systemd's view of the world; it does not restart the running server.
systemctl start flashnext-vllm.service
echo "[ OK ] started"

echo
systemctl is-active  flashnext-vllm.service | sed 's/^/  active:  /'
systemctl is-enabled flashnext-vllm.service | sed 's/^/  enabled: /'
printf '  health:  %s\n' "$(curl -s -m 15 -o /dev/null -w '%{http_code}' http://127.0.0.1:8890/health)"
