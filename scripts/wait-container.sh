#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Block while the serving container runs, so a systemd unit's own state can BE
# the container's liveness instead of claiming it.
#
# Why this exists: deploy/flashnext-vllm.service is a launcher --
# oneshot + RemainAfterExit, deliberately, so `systemctl start` blocks until
# /health answers and `systemctl stop` runs stop.sh. The cost is that its state
# never changes afterwards: when the watchdog stops the container (as it did on
# 2026-09-28 at MemFree 1.2 GiB), `systemctl is-active flashnext-vllm` still
# answers "active" with nothing serving. A human or a script that checks the
# unit instead of the endpoint is told the wrong thing.
#
# `docker wait` returns only when the container stops, so a Type=simple unit
# running this is active exactly as long as the container is.
#
# Exit codes are deliberate: reaching the end of this script means the container
# stopped WITHOUT anyone stopping the unit (a deliberate `systemctl stop`
# signals this process instead, and these lines are never reached), so it is a
# death and the unit should fail and alert.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 3

# The name, resolved the way start.sh resolves it (start.sh:250), without
# sourcing .env: it is a config file, not a script, and this needs one value.
C="${TP1_CONTAINER_NAME:-}"
if [[ -z "$C" && -f .env ]]; then
    C="$(sed -n 's/^[[:space:]]*TP1_CONTAINER_NAME=\([^[:space:]#]*\).*/\1/p' .env | tail -1)"
fi
C="${C:-vllm-fn-tp1}"

# Wait for it rather than giving up. The operator's own path is ./stop.sh then
# ./start.sh (README, supervise.sh, maintenance-relaunch.sh), which never touches
# systemd, so without this the unit latched `failed` after every relaunch and
# needed a root `systemctl start` to clear -- five times on 2026-09-30 alone.
# With Restart=on-failure on the unit, a stop now costs one alert (correct: the
# container did stop) and the tracker re-attaches by itself when it comes back.
WAIT_S="${WAIT_FOR_CONTAINER_S:-1800}"
waited=0
while ! docker inspect "$C" >/dev/null 2>&1; do
    if (( waited >= WAIT_S )); then
        echo "container $C has not appeared in ${WAIT_S}s" >&2
        exit 2
    fi
    (( waited == 0 )) && echo "container $C is not up; waiting up to ${WAIT_S}s for it"
    sleep 5; waited=$(( waited + 5 ))
done
(( waited > 0 )) && echo "container $C appeared after ${waited}s"
echo "tracking $C; this unit stays active while it runs"
code="$(docker wait "$C" 2>/dev/null)" || { echo "docker wait failed for $C" >&2; exit 2; }
echo "container $C stopped on its own (exit $code) -- nothing is serving" >&2
exit 1
