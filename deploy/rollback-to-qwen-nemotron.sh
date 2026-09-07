#!/usr/bin/env bash
# Roll back from Qwen3.8-Flash-Next to the pair this box served before it:
# qwen3.8-27b (SGLang + DFlash2, :8004) and nemotron-3.5-lightning (vLLM, :8005).
#
#   ./rollback-to-qwen-nemotron.sh            graceful
#   ./rollback-to-qwen-nemotron.sh --force    skip the graceful stop (wedged box)
#   ./rollback-to-qwen-nemotron.sh --check    report what it sees, change nothing
#
# Run --check now and then. A rescue script you have never executed is a guess,
# and this one is only ever reached on a bad day.
#
# Why this is not just "systemctl start qwen38-dflash nemotron-vllm":
#
#   * The GB10 has UNIFIED memory. Flash-Next holds ~95 GiB and the pair needs
#     ~92 GiB. Starting the pair while the driver has not yet released the
#     Flash-Next pages does not fail cleanly -- it wedges the box into a
#     hard-power-off. So this waits for the GPU to be observably empty first,
#     and that wait is the whole point of the script.
#   * Boot ORDER decides who fits: qwen takes its 0.56 fraction of whatever is
#     free at launch, so it must come up before nemotron, not alongside it.
#     nemotron-vllm.service encodes this as After=qwen38-dflash.service, but the
#     script path has to enforce it too.
#   * The units may not be installed, or may be out of sync with reality (the
#     containers here have been started by hand before, leaving both units
#     `inactive` while both servers answered). So systemd is preferred and the
#     launch scripts are the fallback.
set -uo pipefail

FN_CONTAINER="vllm-fn-tp1"
QWEN_CONTAINER="qwen3.8-27b-sglang"
NEM_CONTAINER="nemotron35-vllm"
QWEN_RECIPE="/home/oscar/qwen38-recipe"
FN_RECIPE="/home/oscar/flashnext-recipe"
GPU_DRAIN_TIMEOUT="${GPU_DRAIN_TIMEOUT:-300}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"
FORCE=false
CHECK=false
case "${1:-}" in
    --force) FORCE=true ;;
    --check) CHECK=true ;;
    "")      ;;
    *)       echo "unknown option: ${1} (try --check)" >&2; exit 1 ;;
esac

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

unit_installed() { systemctl list-unit-files "$1" --no-legend 2>/dev/null | grep -q .; }
running()        { docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$1"; }
mem_avail_gib()  { awk '/^MemAvailable:/{printf "%.1f", $2/1048576}' /proc/meminfo; }
gpu_tenants()    { nvidia-smi --query-compute-apps=pid,process_name,used_memory \
                     --format=csv,noheader 2>/dev/null | sed '/^$/d'; }

# PIDs belonging to the pair's containers. Matching on process NAME does not
# work: qwen's main process reports to nvidia-smi as plain "python3", so a
# name-based filter reads the server we are restoring as a foreign tenant and
# waits forever. docker top lists every PID in the container's namespace and
# covers all of them.
pair_pids() {
    local c
    for c in "$QWEN_CONTAINER" "$NEM_CONTAINER"; do
        running "$c" && docker top "$c" -o pid 2>/dev/null | tail -n +2 | tr -d ' '
    done
}

# Tenants that are NOT part of the pair -- i.e. what we are actually waiting on.
foreign_tenants() {
    local allowed; allowed="$(pair_pids)"
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        local pid="${line%%,*}"; pid="${pid// /}"
        grep -qx "$pid" <<<"$allowed" || echo "$line"
    done < <(gpu_tenants)
}

wait_health() {  # <url> <label> <timeout>
    local url="$1" label="$2" timeout="$3" waited=0 code
    while (( waited < timeout )); do
        code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || echo 000)
        [[ "$code" == "200" ]] && { ok "$label answers 200 after ${waited}s"; return 0; }
        sleep 10; waited=$((waited + 10))
    done
    return 1
}

# ---------------------------------------------------------------------------
# 0. --check: dry run
# ---------------------------------------------------------------------------
if $CHECK; then
    info "=== dry run: nothing below is executed ==="
    printf '  %-26s %s\n' "flashnext container" "$(running "$FN_CONTAINER" && echo RUNNING || echo "not running")"
    printf '  %-26s %s\n' "qwen container"      "$(running "$QWEN_CONTAINER" && echo RUNNING || echo "not running")"
    printf '  %-26s %s\n' "nemotron container"  "$(running "$NEM_CONTAINER" && echo RUNNING || echo "not running")"
    for u in flashnext-vllm qwen38-dflash nemotron-vllm; do
        printf '  %-26s %s\n' "${u}.service" \
            "$(unit_installed "${u}.service" && echo "installed, $(systemctl is-active "${u}.service"), $(systemctl is-enabled "${u}.service" 2>/dev/null)" || echo "NOT installed")"
    done
    for f in "$FN_RECIPE/stop.sh" "$QWEN_RECIPE/start-dflash.sh" "$QWEN_RECIPE/start-nemotron-vllm.sh"; do
        printf '  %-26s %s\n' "$(basename "$f")" "$([[ -x "$f" ]] && echo "executable" || echo "*** MISSING/NOT EXECUTABLE: $f ***")"
    done
    printf '  %-26s %s GiB\n' "MemAvailable" "$(mem_avail_gib)"
    echo "  GPU tenants:";        gpu_tenants   | sed 's/^/    /'
    echo "  of those, foreign:";  { foreign_tenants | sed 's/^/    /'; } ; [[ -z "$(foreign_tenants)" ]] && echo "    (none -- step 2 would not block)"
    echo
    info "would: stop Flash-Next -> wait for foreign tenants to clear -> start qwen (:8004) -> wait 200 -> start nemotron (:8005) -> wait 200"
    exit 0
fi

# ---------------------------------------------------------------------------
# 1. Stop Flash-Next
# ---------------------------------------------------------------------------
info "=== Step 1: stop Flash-Next ==="
if unit_installed flashnext-vllm.service && [[ "$(systemctl is-active flashnext-vllm.service)" == "active" ]]; then
    info "stopping flashnext-vllm.service"
    sudo systemctl stop flashnext-vllm.service || warn "systemctl stop returned non-zero; continuing to the container check"
fi
if running "$FN_CONTAINER"; then
    if $FORCE; then
        warn "--force: docker kill $FN_CONTAINER (no log archive, no graceful /dev/shm unlink)"
        docker kill "$FN_CONTAINER" >/dev/null 2>&1 || true
        docker rm -f "$FN_CONTAINER" >/dev/null 2>&1 || true
    else
        info "running $FN_RECIPE/stop.sh (archives logs, stops the memwatch watchdog)"
        "$FN_RECIPE/stop.sh" || warn "stop.sh returned non-zero; checking the container anyway"
    fi
fi
# The watchdog outlives a docker-level kill; stop.sh handles it, --force does not.
pkill -f "memwatch.sh $FN_CONTAINER" 2>/dev/null && info "memwatch watchdog stopped"
running "$FN_CONTAINER" && err "$FN_CONTAINER is still running. Not starting anything on top of it."
ok "Flash-Next is not running."

# ---------------------------------------------------------------------------
# 2. Wait for the driver to actually release the memory
# ---------------------------------------------------------------------------
info "=== Step 2: wait for the GPU to drain (unified memory: this is the safety step) ==="
waited=0
while (( waited < GPU_DRAIN_TIMEOUT )); do
    foreign="$(foreign_tenants)"
    if [[ -z "$foreign" ]]; then
        ok "no foreign GPU tenants; MemAvailable $(mem_avail_gib) GiB"
        break
    fi
    (( waited % 30 == 0 )) && info "still held (${waited}s): $(tr '\n' ';' <<<"$foreign")"
    sleep 5; waited=$((waited + 5))
done
if [[ -n "$(foreign_tenants)" ]]; then
    echo "--- tenants not belonging to qwen/nemotron ---"; foreign_tenants
    err "GPU still held after ${GPU_DRAIN_TIMEOUT}s. Starting the pair now risks wedging the box. Investigate before retrying."
fi

# ---------------------------------------------------------------------------
# 3. qwen FIRST -- it sizes its 0.56 fraction from whatever is free at launch
# ---------------------------------------------------------------------------
info "=== Step 3: start qwen3.8-27b (:8004) ==="
info "MemAvailable before launch: $(mem_avail_gib) GiB"
if running "$QWEN_CONTAINER"; then
    ok "$QWEN_CONTAINER already running"
elif unit_installed qwen38-dflash.service; then
    sudo systemctl start qwen38-dflash.service || warn "systemctl start failed; falling back to the script"
    running "$QWEN_CONTAINER" || "$QWEN_RECIPE/start-dflash.sh"
else
    warn "qwen38-dflash.service not installed; using the script directly"
    "$QWEN_RECIPE/start-dflash.sh"
fi
wait_health "http://127.0.0.1:8004/health" "qwen3.8-27b :8004" "$HEALTH_TIMEOUT" \
    || err "qwen did not answer within ${HEALTH_TIMEOUT}s. NOT starting nemotron on top of a half-up qwen."

# ---------------------------------------------------------------------------
# 4. nemotron second
# ---------------------------------------------------------------------------
info "=== Step 4: start nemotron-3.5-lightning (:8005) ==="
info "MemAvailable before launch: $(mem_avail_gib) GiB"
if running "$NEM_CONTAINER"; then
    ok "$NEM_CONTAINER already running"
elif unit_installed nemotron-vllm.service; then
    sudo systemctl start nemotron-vllm.service || warn "systemctl start failed; falling back to the script"
    running "$NEM_CONTAINER" || "$QWEN_RECIPE/start-nemotron-vllm.sh"
else
    warn "nemotron-vllm.service not installed; using the script directly"
    "$QWEN_RECIPE/start-nemotron-vllm.sh"
fi
wait_health "http://127.0.0.1:8005/health" "nemotron :8005" "$HEALTH_TIMEOUT" \
    || err "nemotron did not answer within ${HEALTH_TIMEOUT}s. qwen is up; investigate nemotron alone."

# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------
echo
info "=== Restored ==="
printf '  %-24s %s\n' "qwen3.8-27b :8004" "$(curl -s -m 10 -o /dev/null -w '%{http_code}' http://127.0.0.1:8004/health)"
printf '  %-24s %s\n' "nemotron :8005"    "$(curl -s -m 10 -o /dev/null -w '%{http_code}' http://127.0.0.1:8005/health)"
printf '  %-24s %s GiB\n' "MemAvailable" "$(mem_avail_gib)"
gpu_tenants | sed 's/^/  tenant: /'
echo
warn "The units' enabled state is NOT changed by this script. If flashnext-vllm.service"
warn "was enabled, it will come back on the next boot and Conflicts= will stop this pair."
warn "Disable it explicitly:  sudo systemctl disable flashnext-vllm.service"
