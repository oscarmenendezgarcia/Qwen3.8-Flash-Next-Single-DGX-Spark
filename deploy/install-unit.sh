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
# The launcher's own state cannot track the container (see the unit's header), so
# a second unit does, and both are installed together or the truthful one is
# missing exactly when it matters.
DST_ALIVE=/etc/systemd/system/flashnext-vllm-alive.service
DST_ALERT=/etc/systemd/system/flashnext-alert@.service

[[ -x "$RECIPE_DIR/start.sh" ]] || { echo "no start.sh in $RECIPE_DIR"; exit 1; }
echo "  recipe : $RECIPE_DIR"
echo "  user   : $RUN_USER  (home $HOME_DIR)"

for pair in "flashnext-vllm.service:$DST" "flashnext-vllm-alive.service:$DST_ALIVE"; do
    src="${pair%%:*}"; dst="${pair##*:}"
    sed -e "s|@RECIPE_DIR@|$RECIPE_DIR|g" \
        -e "s|@HOME_DIR@|$HOME_DIR|g" \
        -e "s|@RUN_USER@|$RUN_USER|g" \
        "$HERE/$src" > "$dst"
    chmod 0644 "$dst"
    # @[A-Z_]+@, not a bare @: systemd's own template syntax uses @ (the alive
    # unit's OnFailure=flashnext-alert@%n.service), and a bare-@ check aborted
    # here AFTER overwriting the launcher on disk, blaming a substitution bug
    # that did not exist.
    grep -qE '@[A-Z_]+@' "$dst" && { echo "placeholders left unfilled in $dst"; exit 1; }
    echo "[ OK ] written -> $dst"
done

# The alive unit's OnFailure= names a SYSTEM template, and install-probes.sh
# installs flashnext-alert@.service only into the user manager: the two scopes do
# not share a namespace, so without this the one unit whose whole job is to alert
# fails silently. Derived from the same file rather than duplicated, with the two
# directives a system unit needs and a user unit must not have.
sed -e "s|@RECIPE_DIR@|$RECIPE_DIR|g" \
    -e "/^\[Service\]/a User=$RUN_USER\nEnvironment=HOME=$HOME_DIR" \
    "$HERE/flashnext-alert@.service" > "$DST_ALERT"
chmod 0644 "$DST_ALERT"
grep -qE '@[A-Z_]+@' "$DST_ALERT" && { echo "placeholders left unfilled in $DST_ALERT"; exit 1; }
echo "[ OK ] written -> $DST_ALERT"

systemctl daemon-reload; echo "[ OK ] daemon-reload"
systemctl enable flashnext-vllm.service; echo "[ OK ] enabled"

# start.sh returns early when the container is already up, so this syncs
# systemd's view rather than restarting a live server.
systemctl start flashnext-vllm.service; echo "[ OK ] started"
# Started after the launcher, never enabled on its own: PartOf ties its lifetime
# to the launcher's, and starting it alone would only wait on a container that
# nothing had been asked to create.
systemctl start flashnext-vllm-alive.service; echo "[ OK ] liveness tracker started"

echo
# The launcher's "active" only means it ran; the tracker's means the container is
# up. Both are printed so the difference is visible from the first install.
systemctl is-active  flashnext-vllm.service       | sed 's/^/  launcher ran:    /'
systemctl is-active  flashnext-vllm-alive.service | sed 's/^/  container alive: /'
systemctl is-enabled flashnext-vllm.service | sed 's/^/  enabled: /'
PORT="$(grep -oP '^PORT=\K[0-9]+' "$RECIPE_DIR/.env" 2>/dev/null || echo 8888)"
printf '  health:  %s (port %s)\n' "$(curl -s -m 15 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" "$PORT"
