#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Declare the fp8 hybrid's dense layers so stock vLLM 0.30 quantizes them.

make_fp8_hybrid.sh converts the layers that are read in full every token --
linear_attn in/out projections, self_attn q/k/v/o, mlp.shared_expert -- to
blockwise fp8, and the pinned image picks them up through a runtime shim that
SCANS the tensors (files/fp8_hybrid_modelopt.py.in). Stock vLLM 0.30 does not
scan: ModelOptMixedPrecisionConfig reads a per-layer algo out of config.json and
leaves anything undeclared unquantized, which would try to load fp8 weights as
BF16.

0.30 supports this checkpoint shape natively -- its own docstring says "FP8 for
dense layers and NVFP4 for MoE experts" -- so no shim is needed there, only
honest metadata. Two things have to change, and only the second is obvious:

1. `quantized_layers` gains one entry per converted projection, algo
   **FP8_PB_WO**. Not FP8_BLOCK_SCALES: that one is in _BLOCK_FP8_MOE_ALGOS and
   is resolved for MoE layers only, while LINEAR_ALGOS carries FP8_PB_WO with
   the comment "FP8 128x128 per-block weight". Declaring the wrong one resolves
   the algo and still returns UnquantizedLinearMethod.

2. The exclusion list -- `ignore` in config.json, `exclude_modules` in
   hf_quant_config.json -- drops the patterns that cover those layers:
   `layers.N.linear_attn*`, `layers.N.self_attn*`, `layers.N.mlp.shared_expert*`.
   NVIDIA excludes them because NVIDIA leaves them in BF16, and exclusion is
   checked before quantized_layers, so without this step the entries above are
   dead. Everything else stays excluded: mlp.gate, shared_expert_gate, the
   hyper-connection modules, lm_head, embed_tokens and model.visual*.

Verified against vllm/vllm-openai:v0.30.0 by resolving two layers through the
engine's own code: both returned ModelOptLinearMethod, where before the change
they returned UnquantizedLinearMethod.

Weights are untouched. The variant is a snapshot directory of symlinks beside
the hybrid, so it costs a few hundred KiB.

    python3 files/make_v030_metadata.py <hybrid-snapshot-dir> [<out-dir>]
"""
import json
import os
import re
import sys

DENSE_ALGO = {"quant_algo": "FP8_PB_WO", "group_size": 128}
# Only these three families cover what make_fp8_hybrid.sh converts.
DROP_EXCLUDE = re.compile(r"\.(linear_attn|self_attn|mlp\.shared_expert)\*$")


def converted_layers(snapshot: str) -> list[str]:
    """Tensor prefixes carrying a blockwise fp8 scale, minus what the checkpoint
    already declares: the NVFP4 experts, the PLE table and the MTP experts."""
    index = os.path.join(snapshot, "model.safetensors.index.json")
    weight_map = json.load(open(index, encoding="utf-8"))["weight_map"]
    suffix = ".weight_scale_inv"
    return sorted(
        {
            name[: -len(suffix)]
            for name in weight_map
            if name.endswith(suffix)
            and not name.startswith("mtp.")
            and ".mlp.experts." not in name
            and ".ple." not in name
        }
    )


def main() -> None:
    if not 2 <= len(sys.argv) <= 3:
        sys.exit(__doc__)
    src = sys.argv[1].rstrip("/")
    dst = sys.argv[2] if len(sys.argv) == 3 else src + "-v030"
    dense = converted_layers(src)
    if not dense:
        sys.exit(f"no blockwise-fp8 dense layers in {src}: is this the hybrid?")

    os.makedirs(dst, exist_ok=True)
    rewritten = {"config.json", "hf_quant_config.json"}
    for name in os.listdir(src):
        if name in rewritten:
            continue
        link = os.path.join(dst, name)
        if not os.path.lexists(link):
            os.symlink(os.path.realpath(os.path.join(src, name)), link)

    for name in sorted(rewritten):
        doc = json.load(open(os.path.join(src, name), encoding="utf-8"))
        node = doc.get("quantization_config") or doc.get("quantization")
        if node is None:
            sys.exit(f"{name}: no quantization config to edit")
        for layer in dense:
            node["quantized_layers"][layer] = dict(DENSE_ALGO)
        key = "ignore" if "ignore" in node else "exclude_modules"
        before = len(node[key])
        node[key] = [p for p in node[key] if not DROP_EXCLUDE.search(p)]
        with open(os.path.join(dst, name), "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
        print(
            f"{name}: declared {len(dense)} dense layers as FP8_PB_WO, "
            f"{key} {before} -> {len(node[key])}"
        )
    print(f"ok {dst}")


if __name__ == "__main__":
    main()
