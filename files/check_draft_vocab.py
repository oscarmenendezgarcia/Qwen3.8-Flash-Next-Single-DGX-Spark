#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check a reduced draft vocabulary for per-language holes.

files/build_draft_vocab.py reports one global coverage number over its corpus.
That number hides language skew: a 65,536-id slice built from an English-heavy
corpus can report >99% coverage and still be missing most Spanish content
words, because Spanish tokens sit at high ids and English ones at low ids.

  ES content words live at ids like 79422 (cambios), 178793 (anteriores),
  201538 (muestran); EN ones at 279 (the), 25547 (changes).

This checks coverage per language and fails when one falls below a floor, so a
rebuilt vocabulary cannot silently regress a language.

  python3 files/check_draft_vocab.py draft_vocab.txt
  python3 files/check_draft_vocab.py draft_vocab.txt --min-coverage 98
  python3 files/check_draft_vocab.py draft_vocab.txt --text es=sample_es.txt

Two modes, both reported:

  words  each probe word is *representable* only if every token it splits into
         is in the slice. This is the number that predicts wrong output: a word
         whose tokens are all outside can only be reached through the target
         model, and if verification ever prefers an in-slice near-miss, that
         near-miss is what gets written.
  text   occurrence-weighted coverage over a real sample, per language. Same
         metric build_draft_vocab.py prints, split by language.

Exit status is 1 when any language is below --min-coverage, so this is usable
as a gate after rebuilding the vocabulary.

Requires `transformers`, so it normally runs inside the serving container:

  docker cp files/check_draft_vocab.py vllm-fn-tp1:/tmp/
  docker exec vllm-fn-tp1 python3 /tmp/check_draft_vocab.py /path/to/vocab.txt

files/check_draft_vocab.sh does that for you.
"""
import argparse
import os
import sys


# High-frequency words per language. Function words alone will not find the
# hole -- they are short, shared, and land at low ids in every language -- so
# each list is deliberately weighted towards inflected content words, which is
# where a frequency-ranked slice starts dropping things.
PROBES = {
    "es": """
        que de no a la el es y en lo un por una te los se con para mi está si
        bien pero yo eso las su tu aquí del al como más este ya cuando todo
        nada ahora siempre después mientras aunque porque cambios muestran
        anteriores vuelvo vencidas capturas pantalla idioma orden regenero
        revisar archivo carpeta sistema versión actualizar guardar borrar
        buscar crear eliminar modificar ejecutar comprobar validar desplegar
        configurar instalar reiniciar detener arrancar leer escribir abrir
        cerrar enviar recibir cargar descargar consulta usuario contraseña
        servidor cliente respuesta petición error fallo aviso registro prueba
        pruebas tarea tareas tablero rama commit fusión conflicto dependencia
        biblioteca módulo función variable constante clase método parámetro
        argumento devuelve resultado cadena número lista diccionario fichero
        ruta directorio permiso entorno despliegue construcción compilación
        ejecución memoria disco proceso hilo cola mensaje evento estado
        pendiente completado corriendo detenido fallido correcto incorrecto
        siguiente anterior primero último nuevo viejo mismo distinto
        """.split(),
    "en": """
        that of not to the it is and in what a for me one you with my if well
        but this already when everything nothing now always after while
        although because changes show previous return expired screenshots
        screen language order regenerate review file folder system version
        update save delete search create remove modify run check validate
        deploy configure install restart stop start read write open close
        send receive load download query user password server client response
        request error failure warning log test tests task tasks board branch
        commit merge conflict dependency library module function variable
        constant class method parameter argument returns result string number
        list dictionary path directory permission environment deployment
        build compilation execution memory disk process thread queue message
        event state pending completed running stopped failed correct wrong
        next previous first last new old same different
        """.split(),
    "code": """
        def class return import from async await self None True False lambda
        yield raise except finally assert global nonlocal elif while for in
        not and or is if else try with as pass break continue del print len
        range list dict set tuple str int float bool bytes open read write
        append extend insert remove pop sort reverse join split strip format
        const let var function export default extends implements interface
        typeof instanceof null undefined new this super static public private
        protected throw catch switch case void enum struct impl trait match
        """.split(),
}


def load_vocab(path: str) -> set:
    ids = set()
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                ids.add(int(line))
    if not ids:
        print(f"ERROR: {path} contained no ids", file=sys.stderr)
        sys.exit(2)
    return ids


def word_report(tok, vocab, words, leading_space=True):
    """A word counts as representable only when every one of its tokens is in
    the slice. Words are probed with a leading space because that is how they
    occur mid-sentence, which is the position that matters."""
    ok, missing = 0, []
    for word in words:
        ids = tok.encode((" " if leading_space else "") + word,
                         add_special_tokens=False)
        outside = [i for i in ids if i not in vocab]
        if outside:
            missing.append((word, ids, outside))
        else:
            ok += 1
    return ok, missing


def text_report(tok, vocab, path):
    with open(path, errors="replace") as handle:
        text = handle.read()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if not ids:
        return 0, 0
    covered = sum(1 for i in ids if i in vocab)
    return covered, len(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vocab", help="draft vocabulary file, one token id per line")
    ap.add_argument("--model", default=os.environ.get(
        "DRAFT_VOCAB_MODEL", "Mia-AiLab/Qwen3.8-Flash-Next-NVFP4"))
    ap.add_argument("--min-coverage", type=float, default=99.0,
                    help="fail if any language is below this %% of words (default 99)")
    ap.add_argument("--words", action="append", default=[], metavar="LANG=FILE",
                    help="extra probe list, replaces the built-in list for that language")
    ap.add_argument("--text", action="append", default=[], metavar="LANG=FILE",
                    help="sample text for occurrence-weighted coverage")
    ap.add_argument("--show", type=int, default=12,
                    help="how many missing words to list per language (default 12)")
    args = ap.parse_args()

    vocab = load_vocab(args.vocab)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    probes = dict(PROBES)
    for spec in args.words:
        lang, _, path = spec.partition("=")
        if not path:
            print(f"ERROR: --words wants LANG=FILE, got {spec!r}", file=sys.stderr)
            sys.exit(2)
        with open(path) as handle:
            probes[lang] = handle.read().split()

    total_ids = len(tok)
    print(f"vocabulary:  {len(vocab):,} of {total_ids:,} ids "
          f"({100.0 * len(vocab) / total_ids:.1f}%)")

    # Where the slice sits matters as much as its size: a slice packed into low
    # ids is a slice that speaks English.
    bands = [(0, 65536), (65536, 131072), (131072, 196608), (196608, total_ids)]
    print("id distribution:")
    for lo, hi in bands:
        count = sum(1 for i in vocab if lo <= i < hi)
        print(f"  {lo:>7,}-{hi:<7,}  {count:>7,}  ({100.0 * count / len(vocab):4.1f}%)")

    failures = []
    print("\nword coverage (a word passes only if every token is in the slice):")
    for lang in sorted(probes):
        words = probes[lang]
        ok, missing = word_report(tok, vocab, words)
        pct = 100.0 * ok / len(words) if words else 100.0
        verdict = "ok" if pct >= args.min_coverage else "FAIL"
        if verdict == "FAIL":
            failures.append((lang, pct))
        print(f"  {lang:<5} {ok:>4}/{len(words):<4} {pct:6.1f}%  {verdict}")
        for word, ids, outside in missing[:args.show]:
            print(f"        missing {word:<16} ids={ids} outside={outside}")
        if len(missing) > args.show:
            print(f"        ... and {len(missing) - args.show} more")

    if args.text:
        print("\noccurrence coverage on sample text:")
        for spec in args.text:
            lang, _, path = spec.partition("=")
            covered, total = text_report(tok, vocab, path)
            if not total:
                print(f"  {lang:<5} (empty)")
                continue
            pct = 100.0 * covered / total
            verdict = "ok" if pct >= args.min_coverage else "FAIL"
            if verdict == "FAIL":
                failures.append((lang + ":text", pct))
            print(f"  {lang:<5} {covered:,}/{total:,} {pct:6.2f}%  {verdict}")

    if failures:
        print("\nFAIL: below the %.1f%% floor: %s" % (
            args.min_coverage,
            ", ".join(f"{lang} {pct:.1f}%" for lang, pct in failures)))
        print("A word whose tokens are all outside the slice cannot be drafted. "
              "Rebuild with that language represented in the corpus "
              "(build_draft_vocab.py accepts corpus:N repeat weights), or serve "
              "with MTP_DRAFT_VOCAB unset.")
        sys.exit(1)

    print(f"\nPASS: every language at or above {args.min_coverage:.1f}%")


if __name__ == "__main__":
    main()
