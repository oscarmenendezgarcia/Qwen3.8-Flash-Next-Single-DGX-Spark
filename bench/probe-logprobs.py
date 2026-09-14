#!/usr/bin/env python3
"""Ask the model what probability it gave the malformed word it just wrote.

Takes generations produced by bench/audit-lexical.py, reconstructs the exact
context that preceded a malformation, and reads the top-20 at that position.
Answers whether the malformed continuation was a remote tail event (which
top_p/top_k should have clipped) or ordinary probability mass the nucleus keeps.

Two traps, both of which invalidate the measurement silently:

  1. TRAILING SPACE. BPE does not encode " arr" as " " + "arr". A prefix ending
     in a space puts the model out of distribution and the top-20 comes back as
     mid-word fragments. rstrip() the prefix and match tokens against " "+word.

  2. MISSING CHAT CONTEXT. The generated text alone is a bare prose fragment,
     not what the model saw. Wrap it in the chat template with the original
     prompt, or you are measuring a different distribution.

    python3 bench/probe-logprobs.py --dir lexical-audit \\
        --pair arancaba=arrancaba --pair nocha=noche
"""
import argparse, glob, json, math, os, re, urllib.request

PORT  = os.environ.get("PORT", "8890")
MODEL = os.environ.get("SERVED_MODEL_NAME", "qwen3.8-flash-next")
URL   = f"http://127.0.0.1:{PORT}/v1/completions"
# Qwen3.8-Flash-Next, thinking off. Check with the tokenizer if it ever changes:
#   tok.apply_chat_template([...], add_generation_prompt=True, enable_thinking=False)
TEMPLATE = ("<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n{generated}")


def top_logprobs(prefix, n=20):
    body = {"model": MODEL, "prompt": prefix, "max_tokens": 1,
            "logprobs": n, "temperature": 0.0}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    data = json.loads(urllib.request.urlopen(req, timeout=180).read())
    return data["choices"][0]["logprobs"]["top_logprobs"][0]


def rank_of(ranked, target):
    """Longest top-20 token that starts `target`, with its 1-based rank.

    Longest wins: when the correct and the malformed form share a first token
    the divergence is further along and this row says nothing — which is worth
    seeing rather than hiding."""
    best = None
    for i, (tok, lp) in enumerate(ranked, 1):
        if len(tok) >= 2 and target.startswith(tok):
            if best is None or len(tok) > len(best[2]):
                best = (i, lp, tok)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directory of generations")
    ap.add_argument("--pair", action="append", required=True,
                    metavar="MALFORMED=CORRECT")
    ap.add_argument("--prompts", help="JSON list of the prompts used, indexed by "
                                      "file number mod len; defaults to audit-lexical's")
    a = ap.parse_args()

    if a.prompts:
        prompts = json.load(open(a.prompts, encoding="utf-8"))
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "al", os.path.join(os.path.dirname(__file__), "audit-lexical.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        prompts = mod.PROMPTS

    print(f"  {'produced':<14} {'correct':<14} {'r/p correct':>14} {'r/p malformed':>15}")
    for pair in a.pair:
        bad, good = pair.split("=", 1)
        for path in sorted(glob.glob(os.path.join(a.dir, "*.txt"))):
            text = open(path, encoding="utf-8").read()
            m = re.search(r"\b" + re.escape(bad) + r"\b", text)
            if not m:
                continue
            idx = int(re.sub(r"\D", "", os.path.basename(path)) or 0) % len(prompts)
            prefix = TEMPLATE.format(prompt=prompts[idx],
                                     generated=text[:m.start()].rstrip())
            ranked = sorted(top_logprobs(prefix).items(), key=lambda kv: -kv[1])
            hit_bad, hit_good = rank_of(ranked, " " + bad), rank_of(ranked, " " + good)
            fmt = lambda h: f"{h[0]:>3} / {math.exp(h[1]):.4f}" if h else "    >20 / -"
            print(f"  {bad:<14} {good:<14} {fmt(hit_good):>14} {fmt(hit_bad):>15}")
            break
        else:
            print(f"  {bad:<14} not found in {a.dir}")


if __name__ == "__main__":
    main()
