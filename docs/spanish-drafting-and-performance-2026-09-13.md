# Serving this recipe in Spanish: drafting, sampling and benchmarks

Notes from running `MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark` on a
Spanish-language DGX Spark. Three findings, none of them visible from the
measurements already in this repository, and a harness for each.

| | |
|---|---|
| **1** | A draft vocabulary built from a dictionary destroys Spanish. Use the one this repo ships. |
| **2** | Sampling parameters differ by mode, and mixing them causes language mixing. |
| **3** | The published tok/s and the tok/s you measure are different benchmarks, not a deployment problem. |

---

## 1. Build the draft vocabulary from text, never from a dictionary

`files/draft_vocab_en_code_47k.txt` ships with this repo and is English and
code. It needs no Spanish to serve Spanish correctly. **Three independent
projects converged on this exact file** — MiaAI-Lab, `styles01/sparkrun-recipes`
and `sojufx/sojufx-Qwen3.8-Flash-Next` — which is itself evidence.

A locally built replacement, intended to add Spanish coverage, degraded output
badly enough to force a rollback. The cause is in how it was built: a Spanish
**dictionary** — a flat word list — was passed to `files/build_draft_vocab.py`
as if it were a corpus.

The builder counts token frequencies over text. In a dictionary every inflected
form appears exactly once, so `ababillándoos` carries the weight of `que`. The
frequency signal the method depends on is gone.

Measured against the 65k vocabulary it replaced:

| | |
|---|---|
| ids in the dictionary-built vocabulary | 44,138 |
| shared with its predecessor | 29,376 |
| **ids discarded** | **36,160** |
| of the 2,000 most frequent ids, dropped | 130 |

And the number that matters, ids below 400 — the byte-fallback range:

| vocabulary | byte-fallback ids retained |
|---|---|
| `draft_vocab_en_code_47k.txt` (shipped) | **376** |
| dictionary-built (local) | **286** |

Those 90 missing ids are what BPE uses to assemble `ñ`, `á`, `é` and every
multi-byte UTF-8 sequence. Decoding them returns `'�'`; the tokens accepted
in exchange are ordinary words (`' lentamente'`, `' delito'`, `' considerada'`).

### If a Spanish-aware vocabulary is ever wanted

- **Use text with natural frequencies** — Spanish Wikipedia, the way
  `english.txt` uses wikitext-103 — weighted alongside the other sources.
- **Pin the byte-fallback range unconditionally**, the way special and added
  tokens already are. One line, and this failure becomes impossible.
- **Gate it on Spanish output** before serving it to anyone.

### On the correctness argument

The CHANGELOG argues that a reduced vocabulary cannot change output: drafting is
greedy, and the rejection sampler's greedy branch is
`accepted = target_argmax == draft_sampled`, so a draft survives only when it
equals the target's own choice. Poor coverage costs speed, never correctness.

Two boundaries on that, both worth stating:

- **The evidence behind it is MGSM in English and Chinese**, and the CHANGELOG
  says plainly that the result "does not establish universal quality
  preservation". Spanish was never tested.
- **It is the greedy branch.** Above temperature 0 the verification is
  probabilistic, and there the draft distribution can influence what is
  accepted. That is a plausible route for a reduced vocabulary to affect output
  and it is **unverified** — settling it needs an A/B with and without the
  reduced vocabulary at production temperature.

---

## 2. Sampling parameters depend on the mode

Qwen publishes **different** settings for the two modes
([model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)):

| | thinking | instruct (thinking off) |
|---|---|---|
| temperature | 1.0 | **0.7** |
| top_p | 0.95 | **0.80** |
| top_k | 20 | 20 |
| presence_penalty | 0.0 | **1.5** |
| repetition_penalty | 1.0 | 1.0 |

The checkpoint's `generation_config.json` carries the **thinking** values and
vLLM applies them automatically; the boot log says so:

```
Default vLLM sampling parameters have been overridden by the model's
generation_config.json: {'temperature': 1.0, 'top_k': 20, 'top_p': 0.95}
```

So a server left alone is correct for thinking traffic. **A client that turns
thinking off without also sending the instruct values runs the combination the
model card warns about:** *"using a higher value may occasionally result in
language mixing and a slight decrease in model performance."*

Observed here, not hypothetical: an audit pass at thinking-off with
thinking-mode sampling answered a Spanish prompt in English, on an unrelated
subject. It did not reproduce in three retries once sampling matched the mode.

Practical consequence for anyone with a proxy in front of this model: pinning
`temperature: 1.0` unconditionally is right for thinking traffic and wrong for
everything else.

---

## 3. The performance gap that is not a gap

A day was spent here asking why this host measured **37.15 tok/s** against the
48.7 published in the README — decomposing the engine step, measuring MTP
acceptance per position, sampling clocks under load. Nothing explained it.

`sojufx` publishes two benchmark families side by side, and the answer is the
difference between them:

| benchmark | C1 |
|---|---|
| Decode Bench (synthetic counting stream) | **66.17 tok/s** |
| Agent / tool JSON | 37.1 |
| Code edit | 42.8 |
| Generic coding | 46.9 |
| Long-context review | 37.8 |

Our 37.15 matched their "Agent / tool JSON" (37.1) and "Long-context review"
(37.8) to the decimal. **The published peaks come from a counting stream;
realistic work lands at 37-47.** Nothing was ever slow.

Both belong in an evaluation — a counting stream is a clean decode measurement,
realistic work reveals the cost of actual work. Comparing one against the other
does not.

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

**Caveat**: their prompt *text* is not published, so only the Decode Bench row
compares cleanly. The production-suite prompts here are equivalent in spirit,
not identical, and MTP acceptance is prompt-dependent.

### Why MAX_NUM_SEQS stays at 4

`.env.sample` ships 4 while the README's decode table is measured at 8, and
`docs/overnight-2026-09-05.md` gives the reason: eight concurrent long requests
will contend and preempt, and no long-context run has been done. This host's
traffic is long sessions, which is exactly the untested case. `sojufx` runs 8
without publishing a long-context test either.

---

## The harnesses

`bench/audit-spanish.py` — the Spanish quality gate that did not exist. Eight
long generations (essays, narrative, a formal report, Python with Spanish
docstrings, JSON) plus a five-turn conversation. Analysis is **per paragraph**,
because a reply can begin in Castilian and drift halfway through; looking only
at the whole response hides that. Flags Unicode replacement characters and
markers for Asturian, Galician, Catalan and Portuguese separately. Selects
sampling from the mode.

Result with the shipped 47k vocabulary: across four runs, two vocabularies, two
temperatures and both sampling regimes — roughly **60,000 tokens of Spanish
output** — **zero replacement characters and zero dialect-drift markers**.
Accents and `ñ` render correctly throughout.

It also carries a b/v orthography check, and an honest note about it: **that
check has never caught a real error.** Across four audits it produced only false
positives from its own regexes — `la` and `el` from a capturing group inside a
lookahead, then the correct `anduvo`, then the correct `hacia`. The one
confirmed defect, *reescriví* for *reescribí*, was found by a user in
production. That output is impeccable Castilian, correctly accented, with no
replacement characters and no dialect markers: **it passes every language check
here.** Language identification is not spelling verification. Treat the check as
an unvalidated safety net.

`bench/bench-sojufx-protocol.py` — replicates the published protocol so numbers
taken here can sit beside sojufx's and Mia's.

`bench/verify-smoke.py` — the short version, for a quick check after a boot.

**Run the audit with thinking off.** With thinking on, long prompts spend the
entire budget reasoning and `content` returns empty on a perfectly healthy
server. The README's sanity test warns about this at 200 tokens; it also happens
at 2,000 when the prompt asks for a thousand words.

---

## Two measurement traps

**Container age.** On a sibling deployment on this box, the same binary measured
71.3 tok/s on a container with 29 hours of real serving and 50.2 on a fresh one
— about 40% on a code-shaped probe, reproduced across two different images. An
*idle* soak does not reproduce it: 6 minutes and 71 minutes of idle agreed to
0.04%. The likely cause is radix-cache occupancy, which fills with traffic and
not with uptime; unverified. Any A/B where one side ran on a long-lived
production container and the other on a fresh boot is suspect.

**Reasoning budget.** Through an OpenAI-compatible proxy, `enable_thinking`
defaults on and every reply spends 150-320 tokens reasoning before writing. A
one-line greeting cost 168 tokens of which 158 were reasoning. With a tight
`max_tokens` the client receives an empty `content` and no error.
