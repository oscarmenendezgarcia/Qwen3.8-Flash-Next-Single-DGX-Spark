# Spanish drafting and performance on one GB10, 2026-09-13

Two findings from running this recipe on a Spanish-language box, written down
because neither is visible from upstream's own measurements.

1. A locally built draft vocabulary made the model answer in **Asturian**. The
   cause is identified and reproducible on paper; the fix is to use the
   vocabulary this repo ships.
2. The gap between the tok/s published here and the tok/s a user measures is
   **not a deployment problem**. It is two different benchmarks.

---

## 1. The draft vocabulary that broke Spanish

### What happened

On 2026-09-08 this host served Flash-Next with a locally built reduced draft
vocabulary. The model began answering in Asturian — a language closely related
to Castilian Spanish, sharing most of its morphology. The deployment was
reverted the same week.

At the time the reduced-vocabulary mechanism was believed to be incapable of
changing output at all, on the strength of this argument: draft sampling is
greedy, and the rejection sampler's greedy branch is
`accepted = target_argmax == draft_sampled`, storing `draft_sampled if accepted
else target_argmax`. A draft is kept only when it equals the target model's own
choice. What poor coverage costs is speed, never correctness.

**That argument was quoted without its boundary.** The CHANGELOG's quality
evidence is MGSM in English and Chinese, and it states plainly that the result
"does not establish universal quality preservation". Nobody had tested Spanish.

### The actual defect

The vocabulary in use, `qwen38fn_es_en_code_44k_v2.txt`, was built by passing
`es_dict.txt` — a Spanish **dictionary**, a flat word list — to
`files/build_draft_vocab.py` as if it were a corpus.

The builder counts token frequencies over text. In a dictionary every inflected
form appears exactly once, so `ababillándoos` carried the same weight as `que`.
The frequency signal the method depends on was destroyed.

Measured against the 65k vocabulary it replaced:

| | value |
|---|---|
| ids in the v2 | 44,138 |
| shared with the previous 65k | 29,376 |
| new ids, from the dictionary | 14,762 |
| **ids discarded** | **36,160** |
| of the 2,000 most frequent ids, dropped | 130 |

And the decisive number — ids below 400, the byte-fallback range:

| vocabulary | byte-fallback ids retained |
|---|---|
| `files/draft_vocab_en_code_47k.txt` (shipped) | **376** |
| `qwen38fn_es_en_code_44k_v2.txt` (local, broken) | **286** |

Those 90 missing ids are the pieces BPE uses to assemble `ñ`, `á`, `é` and every
multi-byte UTF-8 sequence. Decoding the dropped tokens returns `'�'`
repeatedly; the tokens accepted in exchange are ordinary Spanish words
(`' lentamente'`, `' delito'`, `' considerada'`).

A drafter that cannot propose the byte layer of accented Spanish, on a
multilingual model, drifts to the nearest thing it can propose. Asturian is
that thing.

**The exact mechanism connecting a reduced draft vocabulary to changed output
has not been verified here.** The rejection-sampler argument says it should not
happen. It did happen. The honest position is that the argument holds for the
cases upstream tested and that this configuration was not one of them.

### What to do instead

Use `files/draft_vocab_en_code_47k.txt`, shipped in this repo since d038090.
47,149 rows, English and code, no Spanish. **Three independent projects
converged on this same file**: MiaAI-Lab, `styles01/sparkrun-recipes` and
`sojufx/sojufx-Qwen3.8-Flash-Next`. That convergence is itself evidence.

Out-of-vocabulary traffic is break-even, not slower: the CHANGELOG's Chinese
case has 50.6% coverage, identical accuracy, and no measurable throughput loss.
Spanish needs no special vocabulary to be correct — it needs one that keeps the
byte layer intact.

If a Spanish-aware vocabulary is ever built, three rules follow from this:

- **A dictionary is not a corpus.** Use Spanish text with natural frequencies
  (Spanish Wikipedia, the way `english.txt` uses wikitext-103), weighted
  alongside the other sources.
- **Pin the byte-fallback range unconditionally**, the way special and added
  tokens are already pinned. One line, and it makes this class of failure
  impossible.
- **Gate it on Spanish output** before serving it to anyone.

### The gate that now exists

`bench/audit-spanish.py`. Eight long generations — essays, narrative, a formal
report, Python with Spanish docstrings, JSON — plus a five-turn conversation.
Analysis is **per paragraph**, because a reply can begin in Castilian and drift
halfway through; looking only at the whole response hides that.

It flags Unicode replacement characters and markers for Asturian, Galician,
Catalan and Portuguese separately.

Result with the shipped 47k vocabulary, 2026-09-13 — roughly 15,000 tokens of
output and 24,025 characters of accumulated thread context:

    replacement characters:  0
    Asturian markers:        0
    Galician / Catalan / Portuguese: 0

Accents and `ñ` render correctly throughout.

**Run it with thinking off.** With thinking on, long prompts spend the entire
token budget reasoning and `content` returns empty on a perfectly healthy
server. The README's sanity test warns about this at 200 tokens; it also happens
at 2,000 when the prompt asks for a thousand words. Five of eight tests failed
that way on the first pass and the failures were entirely an artefact of the
harness.

---

## 2. The performance gap that was not a gap

A day was spent here on 2026-09-07 asking why this host measured **37.15 tok/s**
against the 48.7 published in the README. The engine step was decomposed, MTP
acceptance was measured per position, clocks were sampled under load. Nothing
explained it.

`sojufx/sojufx-Qwen3.8-Flash-Next` publishes two benchmark families side by
side, and the answer is in the difference between them:

| benchmark | C1 |
|---|---|
| Decode Bench (synthetic counting stream) | **66.17 tok/s** |
| Agent / tool JSON | 37.1 |
| Code edit | 42.8 |
| Generic coding | 46.9 |
| Long-context review | 37.8 |

Our 37.15 matches their "Agent / tool JSON" (37.1) and "Long-context review"
(37.8) to the decimal. **The published peaks come from a counting stream;
realistic work lands at 37-47.** Nothing was ever slow here.

As they put it: a counting stream is a clean decode measurement; realistic
tools, code and long context reveal the cost of actual work. Both belong in an
evaluation. Comparing one against the other does not.

### Measured here, their protocol

`bench/bench-sojufx-protocol.py`. 400 tokens, temperature 0, top_p 1, thinking
off, warm server. Shipped 47k vocabulary, `MAX_NUM_SEQS=4`, KV pool 1,057,362
tokens, native 262,144 context.

| | this host | sojufx |
|---|---|---|
| Decode Bench C1, per stream | 61.41 | 66.17 |
| Decode Bench C2, aggregate | 104.13 | 109.70 |
| Decode Bench C4, aggregate | **182.54** | 179.31 |
| Agent / tool JSON C1 | 46.6 | 37.1 |
| Code edit C1 | 44.0 | 42.8 |
| Generic coding C1 | 54.1 | 46.9 |
| Long-context review C1 | 34.7 | 37.8 |

5-7% below at C1 and C2, slightly above at C4 — while running
`MAX_NUM_SEQS=4` against their 8.

**Caveat.** Their prompt *text* is not published, so only the Decode Bench row
compares cleanly; a counting stream is well defined. The production-suite
prompts here are equivalent in spirit, not identical, and MTP acceptance is
prompt-dependent. Read those four rows as indicative.

### Why MAX_NUM_SEQS stays at 4

`.env.sample` ships 4 and the README's own decode table is measured at 8. The
reason for the difference is stated in `docs/overnight-2026-09-05.md`: eight
concurrent long requests will contend and preempt, and no long-context run has
been done. This host's traffic is long sessions, which is exactly the case that
was not shipped for. `sojufx` runs 8 without publishing a long-context test
either.

---

## Two measurement traps worth carrying forward

**Container age.** On a sibling deployment on this box, the same binary measured
71.3 tok/s on a container with 29 hours of real serving and 50.2 on a fresh one
— about 40% on a code-shaped probe, reproduced across two different images. An
*idle* soak does not reproduce it: 6 minutes and 71 minutes of idle agreed to
0.04%. The likely cause is radix-cache occupancy, which fills with traffic and
not with uptime; it is unverified. Any A/B where one side ran on a long-lived
production container and the other on a fresh boot is suspect.

**Reasoning budget.** Through an OpenAI-compatible proxy, `enable_thinking`
defaults on and every reply spends 150-320 tokens reasoning before writing. A
one-line greeting cost 168 tokens of which 158 were reasoning. With a tight
`max_tokens` the client receives an empty `content` and no error. Thinking off
answers the same question in 40 tokens.
