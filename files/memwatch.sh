#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
# memwatch.sh <container> [min_avail_gib] [consecutive_samples]
#
# Host-memory watchdog for the single-Spark deployment. On unified memory an
# exhausted pool hangs the kernel instead of raising an OOM, so this stops the
# container when the host runs out of margin. It is a second line of defence
# behind start.sh's HOST_RESERVE_GIB budget; a userspace poller cannot catch a
# GiB/s collapse alone.
#
# Two independent triggers, each debounced over CONSEC consecutive samples
# (a lone excursion is logged and resets the counter):
#   * MemAvailable < min_avail_gib      (arg 2, default 6)  -- page cache gone,
#     PLE lookups about to hit NVMe.
#   * MemFree < MEMWATCH_MIN_FREE_GIB   (env, default 2)    -- the NVIDIA driver
#     starts refusing allocations (NV_ERR_NO_MEMORY in `journalctl -k`) with
#     MemFree around 3 GiB while MemAvailable still reads 6+ GiB, so the
#     MemAvailable floor alone reacts late. Counted ONLY while MemAvailable is
#     also under MEMWATCH_FREE_GATE_GIB (default 10): with the stock kernel
#     watermarks MemFree legitimately falls to the ~170 MB low watermark
#     whenever the page cache is full of reclaimable data. Measured here
#     2026-09-05 00:49 during weight loading: MemFree 0.9 GiB, MemAvailable
#     32 GiB, zero NVRM errors. Free pages backed by reclaimable cache are not
#     what the driver runs out of; free pages with no cache left to reclaim are.
#
# Every 10 s it also counts NV_ERR_NO_MEMORY lines the kernel log gained since
# the previous check; that is the earliest signal this box gives and is logged
# whenever it is non-zero. (A fixed 12 s window every 10 s double-counted lines
# in the 2 s overlap: 6 counted for 5 logged on 2026-09-05 09:00.)
#
# Timeline every 5 s (every sample once within 1 GiB of either floor) with the
# /proc/meminfo fields needed to tell page cache from anon from driver memory:
#   driver = MemTotal - MemFree - Buffers - Cached - AnonPages - Slab
#            - PageTables - KernelStack
# i.e. memory that is neither free, page cache, anon, kernel slab nor page
# tables: taken through the NVIDIA driver (GPU allocations, pinned host
# buffers). A permanent step up in `driver` with `free` flat is a request
# growing the CUDA caching allocator; that memory does not come back.
#
# Before stopping the container it archives `docker logs --tail 3000` and a
# copy of its own log to logs/archive/, then `docker stop -t $MEMWATCH_GRACE`
# (SIGTERM; a SIGKILL leaks the container's POSIX shm onto the host's
# /dev/shm until reboot because of --ipc host), falling back to docker kill.
#
# Env: MEMWATCH_MIN_FREE_GIB (2), MEMWATCH_FREE_GATE_GIB (10), MEMWATCH_GRACE (30), MEMWATCH_LOG (this
# script's own log, for archiving; default logs/memwatch-<container>.log),
# MEMWATCH_ARCHIVE_DIR (logs/archive).
CONTAINER="${1:?container}"; MIN_GIB="${2:-6}"; CONSEC="${3:-5}"
MIN_FREE_GIB="${MEMWATCH_MIN_FREE_GIB:-2}"
FREE_GATE_GIB="${MEMWATCH_FREE_GATE_GIB:-10}"
# Co-tenant awareness. scripts/supervise.sh already steps aside while ComfyUI is
# up (its "port thief" check); this watchdog did not, and on 2026-09-28 that cost
# a healthy server: a deliberate Qwen-Image generation peaked at 111.6 of
# 121.63 GiB -- 10 GiB spare -- and MemFree dipped to 1.2 GiB because the PLE
# table's page cache holds the rest and cannot be evicted while mapped
# (posix_fadvise(DONTNEED) on it returned 0.3 of 10.8 GiB, measured). The
# MemFree floor fired with MemAvailable at 8.2 GiB and the generation finished
# fine. So while a named co-tenant is up, the MemFree floor is suspended and
# says so once; the MemAvailable floor -- the one that means memory is actually
# gone -- keeps acting. Empty MEMWATCH_COTENANT restores the old behaviour.
COTENANT="${MEMWATCH_COTENANT-comfyui}"
COTENANT_UNITS="${MEMWATCH_COTENANT_UNITS-comfy-h3.service}"
# Seconds to wait for MemAvailable to climb back after shedding the co-tenant,
# before falling through to stopping the protected container anyway.
COTENANT_RECOVER_S="${MEMWATCH_COTENANT_RECOVER_S:-60}"

cotenant_up() {   # container name substring, then the units the supervisor checks
    [[ -z "$COTENANT" && -z "$COTENANT_UNITS" ]] && return 1
    if [[ -n "$COTENANT" ]] && \
       [[ -n "$(docker ps --filter "name=$COTENANT" --filter status=running -q 2>/dev/null)" ]]; then
        return 0
    fi
    local u
    for u in $COTENANT_UNITS; do
        systemctl is-active "$u" >/dev/null 2>&1 && return 0
        systemctl --user is-active "$u" >/dev/null 2>&1 && return 0
    done
    return 1
}
GRACE="${MEMWATCH_GRACE:-30}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
OWN_LOG="${MEMWATCH_LOG:-$REPO_DIR/logs/memwatch-${CONTAINER}.log}"
ARCHIVE_DIR="${MEMWATCH_ARCHIVE_DIR:-$REPO_DIR/logs/archive}"

MIN_KB=$(( MIN_GIB * 1048576 ))
MIN_FREE_KB=$(( MIN_FREE_GIB * 1048576 ))
FREE_GATE_KB=$(( FREE_GATE_GIB * 1048576 ))
NEAR_KB=$(( MIN_KB + 1048576 ))            # verbose band: avail floor + 1 GiB
NEAR_FREE_KB=$(( MIN_FREE_KB + 1048576 ))  # verbose band: free floor + 1 GiB

echo "$(date '+%F %T') watchdog start: container=$CONTAINER" \
     "floors: MemAvailable<${MIN_GIB}GiB, MemFree<${MIN_FREE_GIB}GiB (while MemAvailable<${FREE_GATE_GIB}GiB);" \
     "trigger=${CONSEC} consecutive samples; grace=${GRACE}s; archive=$ARCHIVE_DIR;" \
     "co-tenant suspends the MemFree floor: container ~'${COTENANT:-none}', units '${COTENANT_UNITS:-none}'"
cotenant_noted=0

archive_logs() {  # <timestamp>
    mkdir -p "$ARCHIVE_DIR"
    docker logs --tail 3000 "$CONTAINER" > "$ARCHIVE_DIR/${CONTAINER}-$1-container.log" 2>&1 || true
    [[ -f "$OWN_LOG" ]] && cp -f "$OWN_LOG" "$ARCHIVE_DIR/${CONTAINER}-$1-memwatch.log"
    echo "$(date '+%F %T') archived container log + watchdog log to $ARCHIVE_DIR/${CONTAINER}-$1-*.log"
}

shed_cotenant() {  # <reason>; 0 = shed and memory recovered, 1 = nothing to shed or it did not help
    local names
    names=$(docker ps --filter "name=$COTENANT" --filter status=running --format '{{.Names}}' 2>/dev/null)
    [[ -z "$COTENANT" || -z "$names" ]] && return 1
    echo "$(date '+%F %T') shedding co-tenant(s) [$names] before touching $CONTAINER: $1"
    if [[ -x "$REPO_DIR/scripts/alert.sh" ]]; then
        "$REPO_DIR/scripts/alert.sh" "memwatch shed co-tenant [$names] to protect $CONTAINER: $1" || true
    fi
    docker stop -t 10 $names >/dev/null 2>&1 || docker kill $names >/dev/null 2>&1
    local waited=0 a
    while (( waited < COTENANT_RECOVER_S )); do
        sleep 3; waited=$(( waited + 3 ))
        a=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        if (( a > MIN_KB + 1048576 )); then
            echo "$(date '+%F %T') co-tenant shed worked: MemAvailable=$((a/1024)) MiB after ${waited}s. $CONTAINER left running."
            return 0
        fi
    done
    echo "$(date '+%F %T') co-tenant shed did not recover memory in ${COTENANT_RECOVER_S}s (MemAvailable=$((a/1024)) MiB); stopping $CONTAINER as well."
    return 1
}

stop_container() {  # <reason>
    # The co-tenant is the guest; this container is what the box is for. Shedding
    # the guest first turns a burst of image generations -- eight back to back on
    # 2026-09-30 took MemAvailable under the floor and cost the server 50 minutes
    # -- into an aborted generation instead of a stopped model.
    if shed_cotenant "$1"; then
        below_avail=0
        below_free=0
        return 0
    fi
    local ts; ts=$(date '+%Y%m%dT%H%M%S')
    echo "$(date '+%F %T') $1 -> stopping $CONTAINER"
    archive_logs "$ts"
    docker stop -t "$GRACE" "$CONTAINER" >/dev/null 2>&1 \
        || docker kill "$CONTAINER" >/dev/null 2>&1
    echo "$(date '+%F %T') stopped (NV_ERR_NO_MEMORY seen since watchdog start: $nvrm_total)"
    # The marker line is how the supervisor tells an emergency stop from a
    # clean stop.sh: only the emergency path emits it (review §4.1). It goes
    # into the live log BEFORE the final archive copy so both carry it.
    echo "WATCHDOG EMERGENCY STOP $1" >> "$OWN_LOG"
    # Alert after the stop so the reason carries the final state; a failure
    # here must not change control flow (review §4.6).
    if [[ -x "$REPO_DIR/scripts/alert.sh" ]]; then
        "$REPO_DIR/scripts/alert.sh" "memwatch emergency stop: $1" || true
    fi
    [[ -f "$OWN_LOG" ]] && cp -f "$OWN_LOG" "$ARCHIVE_DIR/${CONTAINER}-$ts-memwatch.log"
    exit 2
}

tick=0
below_avail=0
below_free=0
nvrm_total=0
nvrm_since=$(date '+%Y-%m-%d %H:%M:%S')
cg_path=""
# LEAK TREND (review §4.3): baseline driver figure over a post-load window,
# then flag growth >= TREND_GIB (4) once per day. The baseline window opens
# only after TREND_WARMUP_S — the weight-load ramp (multi-tens-of-GiB) grows
# `driver` immediately, and a min-including-that-ramp would make every cold
# start look like a 4 GiB leak.
baseline_driver=""
trend_logged_day=""
TREND_GIB="${MEMWATCH_TREND_GIB:-4}"
TREND_WARMUP_S="${MEMWATCH_TREND_WARMUP_S:-900}"
TREND_BASELINE_S="${MEMWATCH_TREND_BASELINE_S:-600}"
while docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}\$"; do
    eval "$(awk '/^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapFree|AnonPages|Shmem|Mapped|Slab|SUnreclaim|PageTables|KernelStack):/ {
                     sub(":", "", $1); print "m_" $1 "=" $2 }' /proc/meminfo)"
    avail=$m_MemAvailable; free=$m_MemFree
    driver=$(( m_MemTotal - m_MemFree - m_Buffers - m_Cached - m_AnonPages - m_Slab - m_PageTables - m_KernelStack ))
    if (( driver > 0 )); then
        # Baseline window: [TREND_WARMUP_S, TREND_WARMUP_S + TREND_BASELINE_S)
        # after memwatch start. Take the minimum driver in that post-load
        # window, then freeze it.
        if (( tick >= TREND_WARMUP_S && tick < TREND_WARMUP_S + TREND_BASELINE_S )); then
            if [[ -z "$baseline_driver" || "$driver" -lt "$baseline_driver" ]]; then
                baseline_driver="$driver"
            fi
        elif (( tick == TREND_WARMUP_S + TREND_BASELINE_S )) && [[ -z "$baseline_driver" ]]; then
            baseline_driver="$driver"
        fi
    fi
    if [[ -n "$baseline_driver" ]] && (( driver - baseline_driver >= TREND_GIB * 1048576 )); then
        _today=$(date '+%F')
        if [[ "$trend_logged_day" != "$_today" ]]; then
            trend_logged_day="$_today"
            echo "$(date '+%F %T') LEAK TREND: driver ${TREND_GIB} GiB above the baseline $((baseline_driver/1048576)) MiB (now $((driver/1048576)) MiB). The 2-3 GiB per-request growth is accumulating; scheduled relaunch is the response."
        fi
    fi
    if [[ -z "$cg_path" || ! -f "$cg_path" ]]; then
        cg_path="/sys/fs/cgroup/system.slice/docker-$(docker inspect -f '{{.Id}}' "$CONTAINER" 2>/dev/null).scope/memory.current"
    fi
    cg=$(cat "$cg_path" 2>/dev/null || echo 0)

    if (( avail < MIN_KB )); then
        below_avail=$(( below_avail + 1 ))
        echo "$(date '+%F %T') below MemAvailable floor ${below_avail}/${CONSEC}: MemAvailable=$((avail/1024)) MiB MemFree=$((free/1024)) MiB"
        (( below_avail >= CONSEC )) && stop_container "MemAvailable under ${MIN_GIB} GiB for ${CONSEC} samples"
    else
        (( below_avail > 0 )) && echo "$(date '+%T') recovered after ${below_avail} sub-floor MemAvailable sample(s): MemAvailable=$((avail/1024)) MiB"
        below_avail=0
    fi
    if (( free < MIN_FREE_KB && avail < FREE_GATE_KB )) && cotenant_up; then
        # Suspended, not ignored: this is the operator's own co-tenant, and the
        # MemAvailable floor above still covers real exhaustion.
        if [[ "$cotenant_noted" != "1" ]]; then
            cotenant_noted=1
            echo "$(date '+%F %T') MemFree floor SUSPENDED while a co-tenant is up (MemFree=$((free/1024)) MiB, MemAvailable=$((avail/1024)) MiB). MemAvailable<${MIN_GIB}GiB still stops the container."
        fi
        below_free=0
    elif (( free < MIN_FREE_KB && avail < FREE_GATE_KB )); then
        below_free=$(( below_free + 1 ))
        echo "$(date '+%F %T') below MemFree floor ${below_free}/${CONSEC}: MemFree=$((free/1024)) MiB MemAvailable=$((avail/1024)) MiB"
        (( below_free >= CONSEC )) && stop_container "MemFree under ${MIN_FREE_GIB} GiB for ${CONSEC} samples"
    else
        (( below_free > 0 )) && echo "$(date '+%T') recovered after ${below_free} sub-floor MemFree sample(s): MemFree=$((free/1024)) MiB"
        below_free=0
        if [[ "$cotenant_noted" == "1" ]] && ! cotenant_up; then
            cotenant_noted=0
            echo "$(date '+%F %T') MemFree floor back in force: the co-tenant is gone."
        fi
    fi

    if (( tick % 10 == 0 )); then
        # journalctl --since is inclusive at second granularity, so exclude the
        # boundary second on the next pass by advancing it past this check.
        nvrm_now=$(date '+%Y-%m-%d %H:%M:%S')
        nvrm=$(journalctl -k --since "$nvrm_since" --until "$nvrm_now" -q 2>/dev/null | grep -c NV_ERR_NO_MEMORY || true)
        nvrm_since="$nvrm_now"
        if (( nvrm > 0 )); then
            nvrm_total=$(( nvrm_total + nvrm ))
            echo "$(date '+%F %T') NVRM: ${nvrm} NV_ERR_NO_MEMORY since last check (total ${nvrm_total}); MemFree=$((free/1024)) MiB MemAvailable=$((avail/1024)) MiB"
        fi
    fi

    if (( tick % 5 == 0 || avail < NEAR_KB || (free < NEAR_FREE_KB && avail < FREE_GATE_KB) )); then
        echo "$(date '+%T') avail=$((avail/1024))MiB free=$((free/1024))MiB swapfree=$((m_SwapFree/1024))MiB container=$((cg/1048576))MiB" \
             "cached=$((m_Cached/1024))MiB anon=$((m_AnonPages/1024))MiB shmem=$((m_Shmem/1024))MiB mapped=$((m_Mapped/1024))MiB" \
             "sunreclaim=$((m_SUnreclaim/1024))MiB driver=$((driver/1024))MiB"
    fi
    tick=$((tick+1))
    sleep 1
done
echo "$(date '+%F %T') container gone; watchdog exit (NV_ERR_NO_MEMORY seen since watchdog start: $nvrm_total)"
