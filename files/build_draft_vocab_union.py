#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build a reduced draft vocabulary as corpus frequency UNION language probes.

files/draft_vocab_es_en_code_65k.txt was built the other way round: the shipped
47k English+code file whole as an untouchable floor, plus Spanish Wikipedia on
top. That spends 47,172 of its 65,536 ids on a distribution this host does not
serve, and measures worse for it -- on held-out text (80/20 split of the session
corpora) it covers 98.62% of Spanish occurrences and 97.79% of mixed, while a
31,870-id slice chosen purely by frequency over the same corpora covers 99.53%
and 99.72%. Half the ids, better coverage.

Frequency alone is not enough, and the reason is recorded in
docs/spanish-drafting-and-performance-2026-09-13.md: a locally rebuilt
vocabulary once degraded output badly enough to force a rollback, because the
corpus never exercised tokens the traffic needs. So this unions two things:

  1. build_draft_vocab.py's output over the corpora (which already pins special,
     added and byte-level tokens unconditionally), and
  2. every id the language probes in check_draft_vocab.py tokenize to, for each
     word in both bare and leading-space form.

Step 2 cost 22 ids and took the probe guards from code 96.7% / es 98.7% (FAIL)
to 100% / 100%. That is the whole point: 22 ids buy the guard that a rollback
once paid for.

    python3 files/build_draft_vocab_union.py CORPUS... --model SNAPSHOT \\
        --out files/draft_vocab_es_en_code_32k.txt --size 32768

Run it inside the serving image: the host has no transformers. The corpora are
generated artifacts and live in ~/.cache/vllm/draft_vocab/, not in this repo.
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile


def probe_ids(tokenizer) -> set:
    """Every id the per-language probe words tokenize to, both surface forms."""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "check_draft_vocab", os.path.join(here, "check_draft_vocab.py")
    )
    module = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, ["check_draft_vocab.py", "placeholder"]
    try:
        spec.loader.exec_module(module)   # main() runs only under __main__
    finally:
        sys.argv = saved
    ids = set()
    for words in module.PROBES.values():
        for word in words:
            for form in (word, " " + word):
                ids.update(tokenizer(form, add_special_tokens=False)["input_ids"])
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="+", help="corpus files, optionally path:N weighted")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=32768)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    # Before the corpus pass, not after: probe_ids() imports this file, and
    # spec_from_file_location does not stat it, so a missing one used to surface as
    # a bare FileNotFoundError after minutes of tokenizing, with no --out written.
    checker = os.path.join(here, "check_draft_vocab.py")
    if not os.path.exists(checker):
        sys.exit(f"missing {checker}: the per-language probes live there, and the "
                 f"union needs them (frequency alone fails the code and es guards)")
    with tempfile.NamedTemporaryFile("r", suffix=".txt", delete=False) as tmp:
        stage = tmp.name
    try:
        subprocess.run(
            [sys.executable, os.path.join(here, "build_draft_vocab.py"), *args.corpus,
             "--model", args.model, "--out", stage, "--size", str(args.size)],
            check=True,
        )
        base = {int(line) for line in open(stage) if line.strip()}
    finally:
        if os.path.exists(stage):
            os.unlink(stage)
    if not base:
        sys.exit("build_draft_vocab.py wrote no ids; refusing to ship a vocabulary "
                 "that is only the probe floor")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    probes = probe_ids(tokenizer)

    union = base | probes
    with open(args.out, "w") as handle:
        handle.write("\n".join(str(i) for i in sorted(union)) + "\n")
    print(f"corpus {len(base):,} + probes {len(probes):,} "
          f"({len(probes - base):,} new) -> {len(union):,} ids in {args.out}")


if __name__ == "__main__":
    main()
