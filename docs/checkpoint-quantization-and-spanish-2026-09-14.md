# The PLE table, at 4 bits, is what breaks Spanish

A day of measurements on one DGX Spark (GB10, 121.63 GiB unified). It starts
from a user report — *the model writes words that do not exist* — and ends with
a checkpoint swap that removes the failure. Everything here was measured on this
box; the harnesses are `bench/audit-lexical.py` and `bench/probe-logprobs.py`.

## The short version

| | Mia mirror | NVIDIA hybrid + bf16 KV |
|---|---|---|
| malformations per 10k words | **123.3** | **30.0** |
| generations drifting to another language | 2/30 | **0/30** |
| decode, single stream (code) | 64.5 tok/s | **61.0 tok/s** |
| decode, single stream (English prose) | 52.8 | 48.1 |
| KV pool | 500,951 tok (1.91x @262k) | 474,503 tok (1.81x) |

Four times fewer malformations and no drift, for **2-9% of decode** — about 5%
on the workloads this box actually serves.

> **Correction, 2026-09-15.** The first revision of this document reported the
> cost as 18% of decode, from a 34.4 vs 41.9 tok/s comparison. Both numbers came
> from a local harness running Spanish literary prose, which turns out to be the
> worst point of the whole range: it is the one workload where the draft
> vocabulary covers badly and acceptance collapses. Measured with sparkDash
> across four prompt types, on both checkpoints, same host reserve and same KV
> dtype, the real spread is 2-9%. Section 6 carries the table.

---

## 1. What the failure is

Two symptoms that turned out to be one mechanism:

- **Malformed words.** `arancaba` (*arrancaba*), `visivilidad`, `nocha`,
  `cangosta`, `vehicolo`, `comenzana` (*comenzaban*), `bloqua` (*bloquea*).
- **Sustained drift.** A whole generation in Asturian, or — reported from a
  Hermes session — in Italian.

A subset is spelled with letters that do not belong to Spanish orthography:
`generaziones`, `parezian`, `plastiko`, `neblika`, `esperansa`, and in the
Italian case `revizione`. Those are subword pieces from a neighbouring
language, not random letter noise. One piece breaks a word; a sustained run of
them takes the whole text.

**It is specific to Spanish.** Same sampling, same day, ten matched prompts per
language:

| language | words | malformations | per 10k |
|---|---|---|---|
| Spanish | 16,792 | 29 | **17.3** |
| English | 22,236 | 1 (arguable) | **0.4** |

English is clean once contractions and dictionary gaps are removed. That alone
rules out generic quantization damage to the output head, which would not
respect language.

## 2. What it is not

Each of these was a live hypothesis, and each was killed by a measurement:

- **The reduced draft vocabulary.** Every piece of every correct form is inside
  the 65k set; there is no coverage hole to fall into.
- **Speculative decoding.** Three arms, `MTP_NUM_SPECULATIVE_TOKENS` 3 / 0 / 3,
  restarting between each: **123.3 / 145.6 / 170.3** per 10k. Two *identical*
  configurations differ by 47 points, so the run-to-run spread is wider than
  anything switching MTP off produces.
- **`presence_penalty`.** 62.2 vs 66.2 per 10k between 1.5 and 0.0 at the same
  temperature. One number. (Production sends none: LiteLLM sent only
  `temperature`, and the model's `generation_config.json` supplies the rest.)
- **The nucleus being skipped in the speculative path.** The only caller passing
  `skip_top_k_top_p=True` is the non-speculative `sample()`, which applies the
  filter itself afterwards; `_verify()` takes the default. Behaviourally: at
  `temperature 2.0`, `top_k=1` and `top_p=0.01` both return coherent prose,
  while `top_k=0` with `top_p=1.0` returns multilingual token salad.
- **`use_local_argmax_reduction`.** Only selects what the drafter proposes, and
  at TP=1 it is a no-op.

## 3. Sampling moves it, but only halfway

The malformed continuation is not a remote tail event. Forcing the exact prefix
that preceded each one and reading the top-20:

| produced | correct | rank/p correct | rank/p malformed |
|---|---|---|---|
| `arancaba` | *arrancaba* | 1 / 0.48 | 8 / **0.016** |
| `nocha` | *noche* | 1 / 0.62 | 19 / 0.0017 |
| `chocoate` | *chocolate* | 1 / 0.24 | 8 / **0.016** |

`top_k=20` and `top_p=0.95` both keep 1-2% of mass, and `temperature 1.0`
samples it at that rate. A temperature sweep on the drift-triggering prompt,
20 generations per arm, `top_p` fixed at 0.80:

| temperature | malformations per 10k | generations with drift |
|---|---|---|
| 0.3 | 49.7 | 2/20 |
| 0.5 | 71.6 | 2/20 |
| 0.7 | 61.0 | 3/20 |
| 1.0 | **291.2** | 3/20 |

Two different answers. Malformations collapse below 0.7 and the curve is flat
under it — 0.7 is the right place to sit, and going lower buys nothing.
**Sustained drift does not respond to temperature at all**, and still happens at
0.3, where the tail is essentially shut. That is not a sampling artefact: once
the model is in a rural register, the Asturian continuation is its high-probability
choice.

Production was moved to the instruct preset on the strength of the first row
(`temperature: 0.7`, `top_p: 0.80`, in LiteLLM and in both Hermes configs;
measured end to end through the proxy: 123.3 -> 31.6 per 10k). Note the client
wins over the proxy: setting it only in LiteLLM does nothing for a client that
sends `temperature` itself, which Hermes did.

## 4. The cause: the PLE n-gram table

The checkpoint this box had been serving, `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`,
says in its own README that it is **a mirror, not a quantization** — the weights
come from `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`, whose model card
documents nothing: no calibration data, no layer policy, no caveats.

Reading the safetensors headers of both checkpoints:

| | Mia mirror | NVIDIA |
|---|---|---|
| routed experts (U8 = NVFP4, 4-bit) | 56.41 GiB | 56.25 GiB |
| dense side layers, F8_E4M3 | 10.06 GiB | 7.03 GiB |
| dense side layers, BF16 | 3.69 GiB | 9.99 GiB |
| **PLE n-gram table** | **26.88 GiB (4-bit)** | **47.75 GiB (FP8)** |
| `lm_head` / `embed_tokens` | BF16 | BF16 |

47.68 / 23.84 = exactly 2.0. The experts are the same; the output head and token
embeddings are BF16 in both, as convention requires. **The two checkpoints differ
in precisely two independent places**, and each is better in one.

The PLE is not the token embedding. It is a per-layer table of
320,001,536 n-grams x 160, a lookup feature added on every layer. That fits the
symptom better than "embeddings are degraded" would: an n-gram is where a
language's *spelling* regularities live, so blurring it damages letter-level
prediction specifically. Every malformation we collected is a spelling failure;
none is a semantic one.

Of the six NVFP4 quantizations published for this model, the mirror we were
serving is the most aggressive by a wide margin (98.7 GiB against 123-174).

**Independent corroboration.** Discussion #3 on NVIDIA's repo is titled *"Any
person have a typo problem running qwen3.8 flash next on vllm or sglang?"*; the
attached screenshot is Thai, and the text is visibly corrupted. A reply notes
the same language broke with another quantization, and that French is fine.

## 5. The measurement

Same battery, same sampling (`temperature 1.0` / `top_p 0.95`, thinking off,
1,400 tokens x 30 generations), restarting between arms:

| arm | malformations per 10k | drift |
|---|---|---|
| Mia mirror | 123.3 | 2/30 |
| Mia mirror, second run | 170.3 | 1/30 |
| **NVIDIA** | **26.1** | **0/30** |
| **NVIDIA + fp8 side layers** | **24.9** | **0/30** |
| **NVIDIA + fp8 side layers + bf16 KV** | **30.0** | **0/30** |

On the drift-triggering prompt alone, 20 generations per arm: NVIDIA 10.3 and
12.0 per 10k against Mia's 291.2, and **0 drift in 43 generations** against
Mia's 12-29%.

What survives the filter on the NVIDIA arms is mostly real Spanish the wordlist
lacks — `engobe`, `chamote`, `piolet`, `meseteño`, `culé` — not corruption. The
true rate is below the number.

## 6. Paying for it: the speed

NVIDIA's checkpoint costs 31% of decode, and the reason is in the table above:
it leaves 9.99 GiB of dense side layers in BF16 where the mirror has 3.69. Those
layers are read **in full on every decoded token**, so they dominate decode
bandwidth; the experts are sparse (10 of 512 active) and already 4-bit.
The mirror is, in effect, already the "hybrid" layout that
[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX) builds by
hand.

Two steps recovered most of it, neither costing quality:

Measured with the local harness (Spanish prose, 600 tokens, C=1):

| | tok/s | KV pool | concurrency @262k |
|---|---|---|---|
| NVIDIA as published | 27.2 | 660k tok | 2.5x |
| + dense side layers to blockwise fp8 | 30.5 (+12%) | 829k | 3.2x |
| + bf16 KV instead of fp8 | **34.4** (+13%) | 481k | **1.84x** |

### What it actually costs (sparkDash, both checkpoints, same day)

The numbers above measure the steps, not the deployment. Run against both
checkpoints with the same tool, the same `HOST_RESERVE_GIB=28` and the same
bf16 KV, at concurrency 1, 600 tokens:

| prompt type | Mia mirror | NVIDIA hybrid | delta |
|---|---|---|---|
| structured | 64.77 | 62.64 | **-3.3%** |
| code | 64.51 | 61.01 | **-5.4%** |
| json | 58.01 | 56.75 | **-2.2%** |
| English prose | 52.82 | 48.07 | **-9.0%** |

Code ladder, aggregate: 64.2 / 110.7 / 195.0 for the mirror against
61.0 / **112.9** / 184.7 — the hybrid is ahead at two streams and 5% behind at
one and four. KV pools land within 6% of each other (500,951 against 474,503
tokens).

**The workload dominates far more than the checkpoint does.** Spanish literary
prose runs at 34.4 tok/s and structured output at 62.6 on the same server, a
1.8x spread; the checkpoints differ by 5%. Any single-workload benchmark of this
model says more about the prompt than about the build.

Prefill on the hybrid, unique prefix per size so the cache cannot inflate it:
1,532 tok/s at 1k, 2,380 at 16k, 2,338 at 64k, 2,222 at 128k, 2,004 at 260k —
and 260k completes without error, which is the closest thing to a long-context
check this build has had. TTFT is 59 s at 128k and 130 s at 260k on a cold
prefix; prefix caching takes a repeated prefix back down to ~0.5 s.

The conversion is `tools/fp8_convert.py` from blazux (Apache-2.0), 300 tensors,
5.42 GiB of BF16 -> 2.71 GiB of fp8, worst per-tensor max relative error 3.5%.
It needs a dispatch shim so vLLM routes those layers to its blockwise-fp8 GEMM
instead of the excluded bf16 path; ours is appended to `modelopt_patched.py` by
`files/patch_modelopt_mxfp8.py` and is a no-op unless `VLLM_FP8_HYBRID=1`.

The bf16 KV comes from blazux's options table, which lists fp8 KV as -10% decode
and one tournament scenario lost. Here it was +13%. Its cost is the pool: 481k
tokens instead of 829k, and concurrency at a full 262k request from 3.2x to
1.84x. A full-context request still fits (it needs 7.20 GiB).

**What this section got wrong the first time.** It closed by warning the gap was
"nearer 25% than 18%", reasoning that the mirror's 41.9 tok/s baseline had been
measured with fp8 KV and would rise with bf16. The direction was right and the
size was wrong: measured properly the next day, the gap is 5%. The error was not
the KV dtype, it was benchmarking one workload — the slowest one — and
generalising from it.

## 7. What is not measured

- **Tool-loop and code *correctness*.** Code and structured output are now
  benchmarked for speed (section 6) and a day of real coding traffic passed
  without a complaint, but neither is a correctness check. The fp8 conversion
  puts 3.5% per-tensor error into dense layers; blazux validated theirs with a
  17-scenario agentic tournament (45/51 before and after), we have no equivalent.
  This is still the largest gap, and it is the workload this box actually serves.
- **Long context retrieval.** A 260k prefill completes without error (section 6),
  which shows the build does not break at full context — but it is not a
  needle-in-a-haystack. One open report claims a community NVFP4 of this model
  finds 300 of 2,048 needles at 400k where NVIDIA's is perfect; blazux measures
  both at parity, 6/6 to 413k. Neither tested a hand-converted snapshot.
- **Stability over hours.** This configuration has run for well under a day, and
  a container-age effect worth ~40% on a code probe has bitten this box before.
- **The draft vocabulary** was built against the mirror; acceptance on this
  checkpoint may differ.

## 8. The harness gap

`bench/audit-spanish.py` answers *"is this Castilian?"* — dialect markers,
accents, replacement characters. **It passed every malformation in this
document.** `comenzana`, `bloqua` and `revizione` are impeccable in every
dimension it measures and still misspelled; each was found by a person reading
the output, not by the harness. Language identification is not spelling
verification, which is why `bench/audit-lexical.py` exists.

Its own limits are worth stating: it reports *candidates*, not errors. On a
clean checkpoint most of what it flags is real Spanish missing from the
wordlist, and the list needs reading. It also cannot see a malformation that
happens to be another valid word.
