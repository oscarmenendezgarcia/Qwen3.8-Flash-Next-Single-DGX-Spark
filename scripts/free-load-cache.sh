#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Return the page cache the weight load leaves behind. Once, after the server is
# up; it is not maintenance.
#
# Loading reads ~71 GiB of shards through the page cache, and nothing reads them
# again — the weights live in the driver from then on. Those pages stay resident
# anyway, because the stock kernel keeps only a 44 MB free-page floor and has no
# reason to evict anything. Measured on this host on 2026-09-30: dropping them
# took MemFree from 2.75 to 17.25 GiB instantly, with the server answering
# throughout, and a second pass minutes later freed exactly nothing — they do not
# come back.
#
# Why it matters beyond tidiness: the NVIDIA driver allocates from the same
# unified pool and cannot wait, so a co-tenant (ComfyUI) could not even create a
# CUDA context — `cudaMemGetInfo: out of memory` at 21.8 GiB of MemAvailable but
# 2.4 GiB of MemFree. After this it saw 11.9 GiB free and started.
#
# The PLE table is deliberately NOT touched: vLLM keeps it mapped and reads it at
# random every token, so its pages are live (a DONTNEED on it frees 0.3 of
# 10.8 GiB, measured) and evicting the rest would only cause re-reads from SSD.
#
#   scripts/free-load-cache.sh <snapshot-dir>
set -uo pipefail
DIR="${1:?snapshot directory}"
[[ -d "$DIR" ]] || { echo "free-load-cache: no such directory: $DIR" >&2; exit 1; }

before=$(awk '/^MemFree:/ {print $2}' /proc/meminfo)
python3 - "$DIR" <<'PY'
import glob, os, sys
freed = 0
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.safetensors"))):
    base = os.path.basename(path)
    if "ple" in base.lower():        # mapped and live; see the header
        continue
    try:
        fd = os.open(os.path.realpath(path), os.O_RDONLY)
    except OSError:
        continue
    try:
        os.posix_fadvise(fd, 0, os.fstat(fd).st_size, os.POSIX_FADV_DONTNEED)
        freed += 1
    except OSError:
        pass
    finally:
        os.close(fd)
print(f"free-load-cache: released the load-time page cache of {freed} shard(s)")
PY
sleep 2
after=$(awk '/^MemFree:/ {print $2}' /proc/meminfo)
awk -v b="$before" -v a="$after" 'BEGIN {printf "free-load-cache: MemFree %.2f -> %.2f GiB (%+.2f)\n", b/1048576, a/1048576, (a-b)/1048576}'
