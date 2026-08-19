#!/usr/bin/env python3
"""Build a descriptive author-style profile from a Russian text corpus.

This script does not diagnose personality and does not decide what is "good"
Russian. It extracts observable frequencies for the humanizer+ru+user layer.
Semantic interpretation is left to the editor/model and must be supported by
corpus examples.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+(?:-[A-Za-zА-Яа-яЁё]+)?")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

DISCOURSE_MARKERS = [
    "ну", "вот", "то есть", "короче", "в общем", "кстати", "просто",
    "вообще", "значит", "слушай", "смотри", "в принципе", "при этом",
    "так что", "с другой стороны", "мне кажется", "похоже", "скорее всего",
    "возможно", "видимо", "наверное", "ладно", "хотя", "впрочем",
]

SELF_REPAIR_MARKERS = [
    "то есть", "точнее", "вернее", "нет,", "хотя нет", "или нет",
    "в смысле", "я имею в виду", "если точнее",
]

STANCE_HEDGES = [
    "мне кажется", "похоже", "скорее всего", "возможно", "видимо",
    "наверное", "предположительно", "я думаю", "я так понимаю",
]

CERTAINTY_MARKERS = [
    "точно", "очевидно", "безусловно", "однозначно", "реально", "точно не",
]

# Very rough proxy for a finite/infinitive verb. It is intentionally labelled
# as a proxy in output; do not treat it as morphology.
VERB_PROXY = re.compile(
    r"\b[а-яё]{3,}(?:ть|ться|ет|ёт|ют|ут|ит|ат|ят|ешь|ишь|ем|им|ете|ите|"
    r"ал|ала|али|ял|яла|яли|ил|ила|или|лся|лась|лись|ем|ен|ена|ены)\b",
    re.I,
)

FIRST_PERSON = re.compile(r"\b(?:я|мы|мне|нам|меня|нас|мой|моя|моё|мои|наш|наша|наше|наши)\b", re.I)


def load_paths(paths: list[str]) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.suffix.lower() in {".txt", ".md"} and child.is_file():
                    docs.append((str(child), child.read_text(encoding="utf-8")))
        elif p.is_file():
            docs.append((str(p), p.read_text(encoding="utf-8")))
        else:
            raise FileNotFoundError(raw)
    return docs


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def sentences(text: str) -> list[str]:
    flat = re.sub(r"\s*\n+\s*", " ", text).strip()
    return [s.strip() for s in SENT_SPLIT.split(flat) if s.strip()]


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0
    v = sorted(values)
    pos = (len(v) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    frac = pos - lo
    return round(v[lo] * (1 - frac) + v[hi] * frac, 2)


def count_phrases(text_low: str, phrases: list[str]) -> dict[str, int]:
    return {p: len(re.findall(r"(?<!\w)" + re.escape(p) + r"(?!\w)", text_low)) for p in phrases}


def top_ngrams(tokens: list[str], n: int, limit: int = 20) -> list[dict]:
    if len(tokens) < n:
        return []
    grams = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return [
        {"text": " ".join(g), "count": c}
        for g, c in grams.most_common(limit)
        if c >= 2
    ]


def analyse(docs: list[tuple[str, str]]) -> dict:
    all_text = "\n\n".join(text for _, text in docs)
    low = all_text.lower()
    toks = words(all_text)
    sents = sentences(all_text)
    sent_lengths = [len(words(s)) for s in sents]
    total_words = len(toks)

    markers = count_phrases(low, DISCOURSE_MARKERS)
    repairs = count_phrases(low, SELF_REPAIR_MARKERS)
    hedges = count_phrases(low, STANCE_HEDGES)
    certainty = count_phrases(low, CERTAINTY_MARKERS)

    no_verb_proxy = [s for s in sents if len(words(s)) >= 1 and not VERB_PROXY.search(s)]
    first_person_sents = [s for s in sents if FIRST_PERSON.search(s)]

    latin_tokens = [t for t in toks if re.fullmatch(r"[a-z]+(?:-[a-z]+)?", t)]

    punctuation = {
        "em_dash": all_text.count("—"),
        "en_dash": all_text.count("–"),
        "colon": all_text.count(":"),
        "semicolon": all_text.count(";"),
        "question": all_text.count("?"),
        "exclamation": all_text.count("!"),
        "ellipsis_unicode": all_text.count("…"),
        "ellipsis_three_dots": all_text.count("..."),
        "parentheses_open": all_text.count("("),
    }

    def per_10k(count: int) -> float:
        return round(count * 10000 / total_words, 2) if total_words else 0

    return {
        "version": 1,
        "corpus": {
            "documents": len(docs),
            "words": total_words,
            "sentences": len(sents),
            "files": [name for name, _ in docs],
        },
        "sentence_length": {
            "p25": percentile(sent_lengths, 0.25),
            "median": percentile(sent_lengths, 0.50),
            "p75": percentile(sent_lengths, 0.75),
            "p90": percentile(sent_lengths, 0.90),
            "short_le_4_rate": round(sum(x <= 4 for x in sent_lengths) / len(sents), 4) if sents else 0,
        },
        "discourse_markers": [
            {"text": k, "count": v, "per_10k_words": per_10k(v)}
            for k, v in sorted(markers.items(), key=lambda x: (-x[1], x[0])) if v
        ],
        "self_repair_markers": [
            {"text": k, "count": v, "per_10k_words": per_10k(v)}
            for k, v in sorted(repairs.items(), key=lambda x: (-x[1], x[0])) if v
        ],
        "stance": {
            "hedges": {k: v for k, v in hedges.items() if v},
            "certainty": {k: v for k, v in certainty.items() if v},
        },
        "syntax_proxies": {
            "sentences_without_verb_proxy_rate": round(len(no_verb_proxy) / len(sents), 4) if sents else 0,
            "first_person_sentence_rate": round(len(first_person_sents) / len(sents), 4) if sents else 0,
            "warning": "These are regex proxies, not morphological analysis.",
        },
        "punctuation": {
            key: {"count": value, "per_10k_words": per_10k(value)}
            for key, value in punctuation.items()
        },
        "code_switching": {
            "latin_token_count": len(latin_tokens),
            "latin_token_rate": round(len(latin_tokens) / total_words, 4) if total_words else 0,
            "top_latin_tokens": Counter(latin_tokens).most_common(30),
        },
        "ngrams": {
            "bigrams": top_ngrams(toks, 2),
            "trigrams": top_ngrams(toks, 3),
        },
        "settings": {
            "imitate_errors": False,
            "correct_norm_errors": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="UTF-8 .txt/.md files or directories")
    parser.add_argument("-o", "--output", help="write JSON to file; stdout otherwise")
    args = parser.parse_args()

    profile = analyse(load_paths(args.paths))
    rendered = json.dumps(profile, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
