#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Let the output head be quantized: pass quant_config to ParallelLMHead.

The head is 22.8% of the bytes a decode step reads on this host -- lm_head.weight
is 248,320 x 2,560 bf16, 1.27 GB, against 2.97 GB of fp8 dense layers and 1.33 GB
of the 10-of-512 active experts -- so halving it is worth ~11% of bytes per step,
and decode here is bandwidth-bound.

Everything else was already in place and verified before writing this:

  * The pinned image's ModelOptMixedPrecisionConfig.get_quant_method dispatches
    `quant_algo == "FP8"` for (LinearBase, ParallelLMHead) to
    ModelOptFp8LinearMethod.
  * files/fp8_head_convert.py declares lm_head FP8 in quantized_layers and drops
    it from the exclusion list; the config resolves `lm_head` (and every prefix
    variant) to FP8, not excluded.
  * The hybrid shim does not claim it: its scan needs `.weight_scale_inv`, and a
    per-tensor head carries `weight_scale`.

But model.py builds the head with no quant_config:

    self.lm_head = ParallelLMHead(
        config.vocab_size,
        config.hidden_size,
        prefix=maybe_prefix(prefix, "lm_head"),
    )

so nothing ever asks the config about that layer. The head loads unquantized and
the extra scale tensor has nowhere to go:

    ValueError: There is no module or parameter named 'lm_head.weight_scale'
    in Qwen3_8FlashNextForCausalLM

This adds the argument, and it does work: with it mounted the engine logs
"Selected FlashInferFP8ScaledMMLinearKernel for ModelOptFp8LinearMethod", so the
head is quantized.

**BLOCKED, cause unknown.** With the head quantized, loading then fails on a
different layer, at 0% of the shards:

    File "vllm/model_executor/layers/linear.py", line 732, in weight_loader
      param_data = param.data
    AttributeError: 'MergedColumnParallelLinear' object has no attribute 'data'

The same snapshot WITHOUT this patch loads every shard and fails only at the end
on the orphan scale tensor, so this patch is what introduces it. The signature
matches what a scale-name convention mismatch produced on the vLLM 0.30 lane --
the loader holding a module where a Parameter belongs -- but the causal chain
here is not established.

One hypothesis was checked and is FALSE: ModelOptFp8LinearMethod does not mutate
the shared fp8_config in __init__ or create_weights, so this is not the
"linear-only change leaks into the MoE method that shares the sub-config" bug
that vLLM 0.30 later guarded against.

Where to look next: this is worth retrying on the 0.30 lane, which already
serves the hybrid (see the 2026-09-25 entry) and carries later fixes in exactly
this area. That is also where bilikaz quantizes the head.

No-ops loudly if the call is not the one it expects.
"""
import os
import re
import sys

TARGET = "models/qwen3_8_flash_next/nvidia/model.py"
OLD = """        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )"""
NEW = """        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            # files/patch_lm_head_quant.py: without this the head is never
            # offered to the quantization config, so a checkpoint carrying a
            # quantized lm_head fails to load on its scale tensor.
            # without_modelopt_fp4 is this file's own idiom for "quantize this
            # only when the checkpoint's scheme covers it": it returns None for
            # modelopt_fp4, so a plain NVFP4 checkpoint keeps an unquantized head
            # exactly as today, while a MIXED_PRECISION one that declares lm_head
            # gets it quantized.
            quant_config=without_modelopt_fp4(vllm_config.quant_config),
            prefix=maybe_prefix(prefix, "lm_head"),
        )"""


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "lm_head_quant", "orig", "model.py")
    out = os.path.join(here, "lm_head_quant", "model.py")
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        print(TARGET)
        return
    if not os.path.exists(src):
        sys.exit(f"missing original: {src}")
    text = open(src, encoding="utf-8").read()
    if "patch_lm_head_quant.py" in text:
        print("already patched")
        return
    if text.count(OLD) != 1:
        sys.exit(
            f"model.py: the ParallelLMHead call is not the expected one "
            f"({text.count(OLD)} matches); serving stock"
        )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text.replace(OLD, NEW))
    print(f"patched {out}")


if __name__ == "__main__":
    main()
