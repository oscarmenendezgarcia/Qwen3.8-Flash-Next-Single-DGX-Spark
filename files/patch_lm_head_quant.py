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

With the argument in place the head does load and serve: the engine logs
"Selected FlashInferFP8ScaledMMLinearKernel for ModelOptFp8LinearMethod" and the
drafter's own head (mtp.py, patched below) loads the same tensors. Two failures
came first and both are recorded because each was mine, not vLLM's:

  * `AttributeError: 'MergedColumnParallelLinear' object has no attribute
    'data'`, blamed on this patch, was EXTRA_DOCKER_ARGS passed by env. That
    variable is in start.sh's _ENV_SNAPSHOT_VARS, so passing it REPLACES the
    .env value, which silently dropped VLLM_FP8_HYBRID=1: the blockwise scales
    of our dense layers then had no method to load them.
  * `NotImplementedError: "addmm_cuda" not implemented for 'Float8_e4m3fn'` in
    mtp.py get_top_tokens(): the reduced draft vocabulary slices rows out of
    lm_head.weight and feeds them to a plain linear. Fixed in
    files/patch_mtp_draft_vocab.py, which now dequantizes the slice to bf16.

Then the head served, and the output was noise -- the drafter's acceptance fell
from 76.6% to 3.3% and generations were cross-script gibberish. That has a
single, sufficient cause, and it is why this patch cannot simply hand the head
to ModelOpt:

    scale[:] = torch.finfo(torch.float32).min      # -3.4e38
    layer.register_parameter("input_scale", scale)
    ...
    layer.input_scale = Parameter(layer.input_scale.max())

ModelOptFp8LinearMethod is static-only ("Future support might be added for
dynamic scales") and overwrites that sentinel only from a checkpoint
`input_scale`. A ModelOpt fp8 checkpoint ships one, calibrated; ours is
converted by files/fp8_head_convert.py from a head that was never calibrated,
so it carries a weight scale and no activation scale, and the head quantizes
hidden states by -3.4e38.

So the head gets its activation scale computed at runtime instead, per token,
via _DynamicActFp8Head below. This needs no calibration pass and is strictly
more accurate than one static per-tensor scale; the weight side is untouched
(static per-tensor, the scalar the converter wrote).

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
            quant_config=fp8_head_quant_config(
                    without_modelopt_fp4(vllm_config.quant_config)
                ),
            prefix=maybe_prefix(prefix, "lm_head"),
        )"""



HEAD_METHOD_ANCHOR = """    if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
        return None
    return quant_config"""

HEAD_METHOD_BLOCK = HEAD_METHOD_ANCHOR + '''


class _DynamicActFp8Head(ModelOptFp8LinearMethod):
    """files/patch_lm_head_quant.py: ModelOpt's fp8 linear with the activation
    scale computed at runtime.

    ModelOptFp8LinearMethod is static-only. It initialises input_scale to
    finfo(float32).min and overwrites it only from a checkpoint `input_scale`,
    which a calibrated ModelOpt export carries and our converted head does not.
    Left alone it quantizes hidden states by -3.4e38 and the logits come out as
    noise (measured: drafter acceptance 76.6% -> 3.3%, cross-script gibberish).

    Per-token dynamic activation scales need no calibration pass and are
    strictly more accurate than one static per-tensor scale. The weight side is
    untouched: static per-tensor, the scalar files/fp8_head_convert.py wrote.
    """

    # Per-token first; per-tensor is the fallback if no kernel takes per-token
    # for this weight shape on this device.
    _ACT_KEYS = (kFp8DynamicTokenSym, kFp8DynamicTensorSym)

    def create_weights(self, layer, *args, **kwargs):
        super().create_weights(layer, *args, **kwargs)
        # Nothing will ever load it, and a dynamic scheme must not read it.
        layer._parameters.pop("input_scale", None)
        failure = None
        for act_key in self._ACT_KEYS:
            try:
                self.fp8_linear = init_fp8_linear_kernel(
                    activation_quant_key=act_key,
                    weight_quant_key=kFp8StaticTensorSym,
                    weight_shape=layer.weight.shape,
                    input_dtype=self.input_dtype,
                    out_dtype=self.out_dtype,
                    module_name=type(self).__name__,
                )
            except Exception as exc:  # no kernel for this pairing
                failure = exc
                continue
            _head_logger.info(
                "fp8 head: dynamic activation scale (%s)", act_key.scale
            )
            return
        raise failure

    def process_weights_after_loading(self, layer) -> None:
        # ModelOpt's own, minus the `layer.input_scale.max()` it would read.
        # Our head is one shard, so its per-tensor scale needs no requantizing.
        layer.weight = torch.nn.Parameter(layer.weight.data.t(), requires_grad=False)
        layer.weight.input_dim = 0
        layer.weight.output_dim = 1
        layer.weight_scale = torch.nn.Parameter(
            layer.weight_scale.data.max(), requires_grad=False
        )
        layer.input_scale = None
        self.fp8_linear.process_weights_after_loading(layer)


class _Fp8HeadQuantConfig:
    """Delegates to the real config, substituting the head's fp8 method."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def get_quant_method(self, layer, prefix):
        method = self._inner.get_quant_method(layer, prefix)
        if type(method) is ModelOptFp8LinearMethod:
            return _DynamicActFp8Head(method.quant_config)
        return method


def fp8_head_quant_config(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    """Wrap a config so a per-tensor fp8 head gets dynamic activation scales."""

    if quant_config is None:
        return None
    return _Fp8HeadQuantConfig(quant_config)'''

HEAD_IMPORT_OLD = """def without_modelopt_fp4("""
HEAD_IMPORT_NEW = """from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8LinearMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
)

_head_logger = init_logger(__name__)


def without_modelopt_fp4("""

MTP_IMPORT_OLD = """from .model import (
    _HC_WEIGHTS_MAPPER,"""
MTP_IMPORT_NEW = """from .model import (
    _HC_WEIGHTS_MAPPER,
    fp8_head_quant_config,
    without_modelopt_fp4,"""
MTP_HEAD_OLD = """                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )"""
MTP_HEAD_NEW = """                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    # files/patch_lm_head_quant.py: the drafter shares the output
                    # head, so it is handed lm_head.weight from the same
                    # checkpoint -- and therefore its scale too. Without this the
                    # main model accepts the quantized head and the drafter
                    # rejects its scale:
                    #   ValueError: no module or parameter named
                    #   'lm_head.weight_scale' in Qwen3_8FlashNextMTP
                    quant_config=fp8_head_quant_config(
                        without_modelopt_fp4(vllm_config.quant_config)
                    ),
                    prefix=maybe_prefix(prefix, "lm_head"),
                )"""


def _patch_mtp(here: str) -> None:
    """Patch the GENERATED mtp_patched.py, in place.

    It has to be the generated file and not an orig: start.sh regenerates it from
    the image on every launch through patch_mtp_draft_vocab.py, so this runs after
    that one and edits its output. Idempotent.
    """
    path = os.path.join(here, "mtp_patched.py")
    if not os.path.exists(path):
        print("mtp_patched.py not there yet; skipping the drafter head")
        return
    text = open(path, encoding="utf-8").read()
    if "patch_lm_head_quant.py" in text:
        print("mtp_patched.py: already patched")
        return
    for old, what in ((MTP_IMPORT_OLD, "the .model import"), (MTP_HEAD_OLD, "the ParallelLMHead call")):
        if text.count(old) != 1:
            print(f"mtp_patched.py: {what} is not the expected one ({text.count(old)}); drafter head left alone")
            return
    text = text.replace(MTP_IMPORT_OLD, MTP_IMPORT_NEW).replace(MTP_HEAD_OLD, MTP_HEAD_NEW)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"patched {path} (drafter head)")


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
    if "patch_lm_head_quant.py" not in text:
        if text.count(OLD) != 1:
            sys.exit(
                f"model.py: the ParallelLMHead call is not the expected one "
                f"({text.count(OLD)} matches); serving stock"
            )
        for needle, what in (
            (HEAD_IMPORT_OLD, "the without_modelopt_fp4 definition"),
            (HEAD_METHOD_ANCHOR, "the without_modelopt_fp4 body"),
        ):
            if text.count(needle) != 1:
                sys.exit(
                    f"model.py: {what} is not the expected one "
                    f"({text.count(needle)} matches); serving stock"
                )
        text = (
            text.replace(OLD, NEW)
            .replace(HEAD_IMPORT_OLD, HEAD_IMPORT_NEW)
            .replace(HEAD_METHOD_ANCHOR, HEAD_METHOD_BLOCK)
        )
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"patched {out}")
    else:
        print("already patched")
    _patch_mtp(here)


if __name__ == "__main__":
    main()
