#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""vLLM 0.30: stop ReasoningConfig from raising on a placeholder tokenizer.

On the 0.30 lane the engine never starts:

    ReasoningConfig: failed to tokenize reasoning strings:
    reasoning_start_str='', reasoning_end_str=''

Captured from the failing launch by instrumenting the function, the cause is not
the strings and not this checkpoint:

    tok_type= CachedQwen3_5Tokenizer  vocab= 1
    locales: start='<think>' end='</think>'
    start_ids= [] end_ids= [] natural_ids= []

The parser hands over the right strings. The tokenizer it is asked to encode
them with reports a vocabulary of ONE, so every encode returns []. Loaded
directly from the same snapshot, Qwen3_5Tokenizer reports 248,044 and encodes
'<think>' to [248068], and cached_tokenizer_from_config outside the engine
returns a working CachedQwen2Tokenizer at tokenizer_mode auto, hf and slow
alike. So a placeholder instance reaches ReasoningConfig, and no CLI knob
changes which one.

This turns the raise into a warning and leaves reasoning token IDs
uninitialised, which is what the function already does when the strings are
absent -- one branch above, `return` on an empty string, no exception. The
text-level reasoning parser is unaffected: it matches '<think>' in the output
stream and does not need these IDs.

Reproduced with the plain nvidia snapshot too, so it is not about the fp8
hybrid metadata. Drop this patch when upstream stops validating reasoning
strings against a placeholder tokenizer.
"""
import os
import sys

TARGET = "vllm/config/reasoning.py"
NEEDLE = """            raise ValueError(
                f"ReasoningConfig: failed to tokenize reasoning strings: \""""
REPLACEMENT = '''            # files/patch_v030_reasoning.py: a placeholder tokenizer (vocab 1)
            # reaches this point and encodes everything to []. Warn and leave
            # the IDs uninitialised, exactly as the empty-string branch above
            # does, instead of refusing to build the config.
            import logging

            logging.getLogger(__name__).warning(
                "ReasoningConfig: tokenizer produced no IDs for %r/%r "
                "(vocab_size=%s); reasoning token IDs stay uninitialised.",
                reasoning_start_str,
                reasoning_end_str,
                getattr(tokenizer, "vocab_size", None),
            )
            return
            raise ValueError(
                f"ReasoningConfig: failed to tokenize reasoning strings: "'''


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "v030_reasoning", "orig", "reasoning.py")
    out = os.path.join(here, "v030_reasoning", "reasoning.py")
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        print(TARGET)
        return
    text = open(src, encoding="utf-8").read()
    if "patch_v030_reasoning.py" in text:
        print("already patched")
        return
    if text.count(NEEDLE) != 1:
        sys.exit(f"reasoning.py does not carry the expected raise ({text.count(NEEDLE)} matches)")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text.replace(NEEDLE, REPLACEMENT))
    print(f"patched {out}")


if __name__ == "__main__":
    main()
