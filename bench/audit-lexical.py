#!/usr/bin/env python3
"""Lexical audit: catch malformed words, not the wrong language.

bench/audit-spanish.py answers "is this Castilian?" — dialect markers, accents,
replacement characters. It passes text that is impeccable Castilian and still
misspelled: `comenzana`, `bloqua`, `cabizajo` all clear every one of its checks.
This harness answers the other question.

Signature it looks for: a word absent from a large Spanish wordlist that sits
ONE edit (insert / delete / substitute) from a word in it. That is what a
corrupted subword produces, and it rejects loanwords, which sit far from
everything Spanish.

Noise this removes, in order of how much it matters:
  - proper nouns          (capitalised in the source)
  - English               (a second wordlist)
  - enclitics             (destrabarlos = destrabar + los)
  - dialectal endings     (-ao/-ío/-u: register, not error, and very common
                           once the model adopts a rural voice)

What is left needs reading. It is small — tens of items per 15k words — and a
real defect is obvious on sight.

    PORT=8890 python3 bench/audit-lexical.py --generate 30
    python3 bench/audit-lexical.py --analyse out/

Wordlists: ES_DICT (one per line or whitespace-separated, both accepted) plus
/usr/share/dict/spanish, and /usr/share/dict/{american,british}-english.
"""
import argparse, glob, json, os, re, sys, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

ES_DICT = os.environ.get("ES_DICT", os.path.expanduser(
    "~/.cache/vllm/draft_vocab/es_dict.txt"))
PORT    = os.environ.get("PORT", "8890")
MODEL   = os.environ.get("SERVED_MODEL_NAME", "qwen3.8-flash-next")

# Half technical, half literary on purpose. Literary Spanish carries no
# anglicisms, so the filter runs clean there and real defects stand out; it is
# also where sustained drift into a neighbouring language shows up.
PROMPTS = [
    "Cuenta una historia larga sobre una expedición a la montaña, sin tecnicismos ni anglicismos.",
    "Escribe un relato extenso, en castellano llano, sobre una familia que se muda de pueblo.",
    "Redacta un ensayo largo sobre la historia de la imprenta, en castellano culto.",
    "Describe con mucho detalle un mercado de abastos un sábado por la mañana.",
    "Escribe una crónica deportiva extensa de un partido de fútbol, en castellano.",
    "Redacta una carta formal larga de reclamación a una compañía eléctrica.",
    "Escribe un informe largo sobre una incidencia en producción: causa raíz y medidas.",
    "Narra en varios párrafos un viaje en tren por la meseta castellana en invierno.",
    "Escribe una reseña larga de una novela imaginaria del siglo XIX.",
    "Describe extensamente el oficio de un alfarero y su taller.",
]

# Sustained drift into Asturian/Leonese. Word-boundary anchored: without \b,
# "onde" matches inside "donde" and every file looks affected.
DRIFT = re.compile(
    r"\b(mesmu|infartu|casta[ñn]u|pellizcu|pinchazu|propuestu|cabizbaxo|sal[ií]u|"
    r"dixo|reciu|onde|apret[áa]us|torc[ií]u|veranu|disparu|humu|negru|pol|"
    r"p[áa]jarus|amarillus|chill[ií]u|chirri[ií]u|tej[áa]u|peque[ñn]u|ladru|"
    r"olv[ií]u|agarr[áa]u|reg[áa]u|guaje|yera|ne[ñn]os)\b")

DIALECT_SUFFIX = re.compile(r"([uú]s?|a[uú]s?|[íi]u|[áa]u|[ao]o|[ií]o)$")
ENCLITICS = ("los", "las", "les", "nos", "me", "te", "se", "lo", "la", "le")
ALPHABET = "abcdefghijklmnopqrstuvwxyzáéíóúüñ"


def load(path):
    """Both wordlist shapes occur in the wild: one per line, and whitespace
    separated. .split() covers both; readlines() silently yields nothing
    usable on the second."""
    with open(path, encoding="utf-8", errors="ignore") as fh:
        return {w.lower() for w in fh.read().split() if w}


def spanish(word, es):
    if word in es:
        return True
    for e in ENCLITICS:
        if word.endswith(e) and len(word) > len(e) + 2:
            stem = word[: -len(e)]
            if stem in es or stem + "r" in es:
                return True
    return False


def one_edit_from(word, es):
    for i in range(len(word)):
        if word[:i] + word[i + 1:] in es:
            return True
    for i in range(len(word) + 1):
        for c in ALPHABET:
            if word[:i] + c + word[i:] in es:
                return True
    for i in range(len(word)):
        for c in ALPHABET:
            cand = word[:i] + c + word[i + 1:]
            if cand != word and cand in es:
                return True
    return False


def generate(outdir, n, temperature, top_p, thinking):
    os.makedirs(outdir, exist_ok=True)
    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"

    def one(job):
        i, prompt = job
        body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature, "top_p": top_p, "top_k": 20,
                "max_tokens": 1400,
                "chat_template_kwargs": {"enable_thinking": thinking}}
        try:
            req = urllib.request.Request(url, json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            data = json.loads(urllib.request.urlopen(req, timeout=900).read())
            text = data["choices"][0]["message"].get("content") or ""
        except Exception as exc:                      # noqa: BLE001
            print(f"  request {i} failed: {exc}", file=sys.stderr)
            text = ""
        with open(f"{outdir}/{i:03d}.txt", "w", encoding="utf-8") as fh:
            fh.write(text)
        return len(text)

    jobs = [(i, PROMPTS[i % len(PROMPTS)]) for i in range(n)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        sizes = list(pool.map(one, jobs))
    empty = sum(1 for s in sizes if not s)
    print(f"  {n} generations, {sum(sizes):,} chars"
          + (f", {empty} EMPTY (thinking ate the budget?)" if empty else ""))


def analyse(outdir):
    es = load(ES_DICT) | load("/usr/share/dict/spanish")
    en = {w for w in (load("/usr/share/dict/american-english")
                      | load("/usr/share/dict/british-english")) if len(w) >= 3}

    files = sorted(glob.glob(os.path.join(outdir, "*.txt")))
    if not files:
        sys.exit(f"no *.txt under {outdir}")

    words = 0
    suspects, examples, drifted = Counter(), {}, []
    for path in files:
        raw = open(path, encoding="utf-8").read()
        if DRIFT.findall(raw.lower()):
            drifted.append(os.path.basename(path))
        text = re.sub(r"```.*?```", " ", raw, flags=re.S)   # code blocks
        text = re.sub(r"`[^`]*`", " ", text)                # inline code
        text = re.sub(r"https?://\S+", " ", text)
        for m in re.finditer(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{4,}", text):
            surface = m.group(0)
            word = surface.lower()
            words += 1
            if spanish(word, es) or word in en:
                continue
            if surface[0].isupper():                        # proper noun
                continue
            if DIALECT_SUFFIX.search(word):                 # register
                continue
            if not one_edit_from(word, es):                 # loanword
                continue
            suspects[word] += 1
            examples.setdefault(word, (os.path.basename(path),
                                       text[max(0, m.start() - 70): m.end() + 70]
                                       .replace("\n", " ")))

    total = sum(suspects.values())
    rate = 1e4 * total / words if words else 0.0
    print(f"  {len(files)} files, {words:,} words")
    print(f"  malformation candidates: {total} ({len(suspects)} distinct)"
          f"  ->  {rate:.1f} per 10k")
    print(f"  sustained drift: {len(drifted)}/{len(files)} "
          f"{' '.join(drifted) if drifted else ''}")
    print()
    for word, n in suspects.most_common(60):
        where, ctx = examples[word]
        print(f"  {word!r} x{n}  [{where}]")
        print(f"      ...{ctx}...")
    return rate


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generate", type=int, metavar="N",
                    help="produce N generations into --out, then analyse")
    ap.add_argument("--analyse", metavar="DIR", help="analyse an existing directory")
    ap.add_argument("--out", default="lexical-audit")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.80)
    ap.add_argument("--thinking", action="store_true")
    a = ap.parse_args()
    if not a.generate and not a.analyse:
        ap.error("one of --generate or --analyse is required")
    if a.generate:
        generate(a.out, a.generate, a.temperature, a.top_p, a.thinking)
    analyse(a.analyse or a.out)


if __name__ == "__main__":
    main()
