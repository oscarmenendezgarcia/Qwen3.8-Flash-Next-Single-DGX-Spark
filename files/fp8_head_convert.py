#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quantize the output head to per-tensor fp8, in a variant snapshot.

The head is 22.8% of the bytes a decode step reads on this host: lm_head.weight
is 248,320 x 2,560 in bf16, 1.27 GB, against 2.97 GB of fp8 dense layers and
1.33 GB of the 10-of-512 active experts. Decode is bandwidth-bound, so halving
it is worth about 11% of bytes per step. bilikaz/qwen38-flash-next-recipe took
the same tensor to NVFP4 (1.27 -> 0.33 GB) and measured 50 -> 60 tok/s.

PER-TENSOR fp8, not the blockwise kind this recipe uses elsewhere, and that is
forced rather than chosen:

  * vLLM's own blockwise Fp8Config returns None for a ParallelLMHead -- it does
    not quantize the head at all. Only ModelOpt's methods accept
    (LinearBase, ParallelLMHead).
  * ModelOptFp8LinearMethod.create_weights registers `weight`, `weight_scale`
    (PerTensorScaleParameter) and `input_scale`. So the head needs one scalar
    scale named weight_scale.
  * That naming is also what keeps this clear of the hybrid shim, which only
    claims layers carrying `.weight_scale_inv`. The head falls through to the
    ModelOpt path, which is the one that can serve it. No shim change.

Writes a snapshot of relative symlinks beside the hybrid, with only the shard
holding lm_head rewritten (5.6 GB of the 71), the index extended with the new
scale tensor, and lm_head declared FP8 in quantized_layers and dropped from the
exclusion list.

    python3 files/fp8_head_convert.py <hybrid-snapshot> [<out>]

Run it inside the serving image: the host has no torch or safetensors.
"""
import json
import os
import re
import sys

import torch
from safetensors.torch import save_file

FP8_MAX = 448.0


def main() -> None:
    if not 2 <= len(sys.argv) <= 3:
        sys.exit(__doc__)
    src = sys.argv[1].rstrip("/")
    dst = sys.argv[2] if len(sys.argv) == 3 else src + "-head"
    idx_name = "model.safetensors.index.json"

    index = json.load(open(os.path.join(src, idx_name), encoding="utf-8"))
    wmap = index["weight_map"]
    if "lm_head.weight" not in wmap:
        sys.exit("no lm_head.weight in the index")
    if "lm_head.weight_scale" in wmap:
        sys.exit("lm_head.weight_scale already present: already converted?")
    shard = wmap["lm_head.weight"]

    os.makedirs(dst, exist_ok=True)
    rewritten = {shard, idx_name, "config.json", "hf_quant_config.json"}
    for name in os.listdir(src):
        if name in rewritten:
            continue
        link = os.path.join(dst, name)
        if os.path.lexists(link):
            continue
        entry = os.path.join(src, name)
        # Relative, always: the container mounts the cache at another absolute
        # path, and an absolute link dangles there (it shows up as an unrelated
        # "Can't load image processor" much later).
        target = os.readlink(entry) if os.path.islink(entry) else os.path.join(
            "..", os.path.basename(src), name
        )
        os.symlink(target, link)

    # Chunked, because the obvious version dies: load_file brings in the whole
    # 5.6 GB shard, and three float32 copies of a 1.27 GB tensor for the error
    # stats took the peak past a 14 GiB container (exit 137, no output -- the
    # kill lands before the first print flushes).
    from safetensors import safe_open

    with safe_open(os.path.join(src, shard), framework="pt") as fh:
        names = list(fh.keys())
        amax = torch.zeros((), dtype=torch.float32)
        rows = None
        for name in names:
            if name != "lm_head.weight":
                continue
            w = fh.get_slice(name)
            shp = w.get_shape()
            rows = shp[0]
            CH = 16384
            for i in range(0, rows, CH):
                amax = torch.maximum(amax, w[i : i + CH].abs().max().to(torch.float32))
        scale = (amax / FP8_MAX).clamp(min=torch.finfo(torch.float32).tiny)
        print(f"  lm_head.weight {tuple(shp)}, {shp[0]*shp[1]*2/1e9:.2f} GB bf16")
        print(f"  escala {scale.item():.6g}")

        tensors = {}
        for name in names:
            if name == "lm_head.weight":
                continue
            tensors[name] = fh.get_tensor(name)

        w = fh.get_slice("lm_head.weight")
        q = torch.empty(shp, dtype=torch.float8_e4m3fn)
        CH = 16384
        err_rel, muestras = [], 0
        for i in range(0, rows, CH):
            blk = w[i : i + CH].to(torch.float32)
            qb = (blk / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            q[i : i + CH] = qb
            if muestras < 3:  # error sobre una muestra, no sobre 635M valores
                back = qb.to(torch.float32) * scale
                d = (back - blk).abs() / blk.abs().clamp(min=1e-6)
                err_rel.append(d.median().item())
                muestras += 1
            del blk, qb
        print(f"  error mediano relativo {sum(err_rel)/len(err_rel)*100:.2f}%  (muestra de {muestras} bloques)")

    tensors["lm_head.weight"] = q
    tensors["lm_head.weight_scale"] = scale.reshape(1)
    save_file(tensors, os.path.join(dst, shard), metadata={"format": "pt"})
    del tensors, q
    print(f"  {shard}: {os.path.getsize(os.path.join(src, shard))/1e9:.2f} -> "
          f"{os.path.getsize(os.path.join(dst, shard))/1e9:.2f} GB")

    wmap["lm_head.weight_scale"] = shard
    index["weight_map"] = wmap
    if "metadata" in index and "total_size" in index["metadata"]:
        index["metadata"]["total_size"] = sum(
            os.path.getsize(os.path.realpath(os.path.join(dst, s)))
            for s in sorted(set(wmap.values()))
        )
    json.dump(index, open(os.path.join(dst, idx_name), "w", encoding="utf-8"), indent=2)

    for name in ("config.json", "hf_quant_config.json"):
        doc = json.load(open(os.path.join(src, name), encoding="utf-8"))
        node = doc.get("quantization_config") or doc.get("quantization")
        node["quantized_layers"]["lm_head"] = {"quant_algo": "FP8"}
        key = "ignore" if "ignore" in node else "exclude_modules"
        before = len(node[key])
        node[key] = [p for p in node[key] if p != "lm_head" and not re.fullmatch(r"lm_head\*?", p)]
        json.dump(doc, open(os.path.join(dst, name), "w", encoding="utf-8"), indent=2)
        print(f"  {name}: lm_head declared FP8, {key} {before} -> {len(node[key])}")
    print(f"ok {dst}")


if __name__ == "__main__":
    main()
