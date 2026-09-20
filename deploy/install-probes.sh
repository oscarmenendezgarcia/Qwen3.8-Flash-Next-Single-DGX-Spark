#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# install-probes.sh — install the stateless probes as systemd USER units.
#
# This is the half of systemd/ that costs nothing to adopt. The 24/7 supervisor
# there calls start.sh and stop.sh, and so does deploy/flashnext-vllm.service;
# neither knows about the other, so running both leaves two things starting and
# stopping one container. That is a choice between them. The probes are not:
# health-probe.sh and smoke-test.sh only read and generate, so they sit beside
# whatever owns the server.
#
# No root: user units, in ~/.config/systemd/user. `loginctl enable-linger` is
# what makes them run without a login, and the script checks rather than assumes.
#
# memwatch is already wired -- start.sh launches it and stop.sh kills it -- so
# it is not here.
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"
UNITS=(flashnext-health-probe.service flashnext-health-probe.timer
       flashnext-smoke.service flashnext-smoke.timer "flashnext-alert@.service")

mkdir -p "$UNIT_DIR"
for u in "${UNITS[@]}"; do
    sed -e "s|@RECIPE_DIR@|$RECIPE_DIR|g" "$RECIPE_DIR/deploy/$u" > "$UNIT_DIR/$u"
    echo "[ OK ] $UNIT_DIR/$u"
done

systemctl --user daemon-reload && echo "[ OK ] daemon-reload"
systemctl --user enable --now flashnext-health-probe.timer
systemctl --user enable --now flashnext-smoke.timer
echo "[ OK ] timers enabled"

if [[ "$(loginctl show-user "$USER" --property=Linger --value 2>/dev/null)" != "yes" ]]; then
    echo "[WARN]  Linger is off: these timers stop when you log out and do not"
    echo "[WARN]  come back at boot. Turn it on with:  loginctl enable-linger $USER"
fi

echo
systemctl --user list-timers --no-pager 'flashnext-*' || true
