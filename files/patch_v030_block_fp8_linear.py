#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adapt this recipe's scale naming to ModelOpt's on the vLLM 0.30 lane.

NOT an upstream bug, and not a candidate for a PR -- that was the first reading
and it was wrong. vLLM routes FP8_PB_WO linear layers to ModelOptLinearMethod on
purpose, and `tests/quantization/test_modelopt.py` guards it:

    @pytest.mark.parametrize("algo", list(LINEAR_ALGOS))
    def test_modelopt_mixed_precision_dispatches_every_linear_algo(algo):
        ...
        assert isinstance(method, ModelOptLinearMethod)

That path does support 128x128 block fp8 -- `resolve()` returns
QuantSpec(weight=kFp8Static128BlockSym, ...) and names CompressedTensors block
FP8 as its reference. The mismatch is the SCALE TENSOR NAME. ModelOpt's linear
path reads `weight_scale` (`process_weights_after_loading` touches
`layer.weight_scale`); files/fp8_convert.py writes `weight_scale_inv`, the
DeepSeek convention, which is what the pinned image's shim and vLLM's own
Fp8LinearMethod expect.

So this patch is an adapter for OUR convention, not a fix for theirs: it sends
block-scaled fp8 linear layers to the Fp8 method that reads the names we wrote,
the same object the RoutedExperts branch already uses:

    if quant_algo in _BLOCK_FP8_MOE_ALGOS:
        return Fp8MoEMethod(self.fp8_block_config, layer)

For the linear half it does not. LINEAR_ALGOS maps FP8_PB_WO onto
ModelOptLinearMethod, which reads the ModelOpt per-tensor convention
(weight_scale). A 128x128 block-scaled checkpoint carries weight_scale_inv, and
the loader then hands a module where a Parameter is expected:

    File "vllm/model_executor/layers/linear.py", line 762, in weight_loader
      param_data = param.data
    AttributeError: 'MergedColumnParallelLinear' object has no attribute 'data'

This adds the same branch the MoE path already has, before the LINEAR_ALGOS
lookup: block-scaled fp8 goes to self.fp8_block_config, which is the Fp8Config
that __init__ already builds with weight_block_size=[group_size, group_size].

With it, the fp8 hybrid this recipe builds loads on stock vLLM 0.30 and serves.
Measured on one GB10 against the pinned-image lane: KV pool 801,076 tokens
against 486,172 (3.06x against 1.74x at 262k), prefill +23% to +33%, decode -3%
to -9%, lexical audit 16.3 malformations per 10k with 0/30 drift, and the repo
smoke test at 7 passed / 0 failed / 1 known warning.

Needs the layers declared: see files/make_v030_metadata.py.

The alternative, which would need no patch at all, is to emit `weight_scale`
instead of `weight_scale_inv`. That means a second converted checkpoint, because
the pinned-image lane reads the other name -- another ~71 GiB against 100 GiB
free. Worth doing only if the 0.30 lane is adopted.
"""
import os
import sys

TARGET = "vllm/model_executor/layers/quantization/modelopt.py"
NEEDLE = """            if quant_algo is None or quant_algo not in LINEAR_ALGOS:
                # Layer not in quantized_layers — leave unquantized
                return UnquantizedLinearMethod()"""
REPLACEMENT = """            # files/patch_v030_block_fp8_linear.py: block-scaled fp8 linear layers
            # go to vLLM's own Fp8 method, the way the RoutedExperts branch below
            # already routes them to Fp8MoEMethod(self.fp8_block_config).
            if quant_algo in _BLOCK_FP8_MOE_ALGOS:
                return self.fp8_block_config.get_quant_method(layer, prefix)
""" + NEEDLE


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "v030_blockfp8", "orig", "modelopt.py")
    out = os.path.join(here, "v030_blockfp8", "modelopt.py")
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        print(TARGET)
        return
    text = open(src, encoding="utf-8").read()
    if "patch_v030_block_fp8_linear.py" in text:
        print("already patched")
        return
    if text.count(NEEDLE) != 1:
        sys.exit(f"modelopt.py does not carry the expected branch ({text.count(NEEDLE)} matches)")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text.replace(NEEDLE, REPLACEMENT))
    print(f"patched {out}")


if __name__ == "__main__":
    main()
