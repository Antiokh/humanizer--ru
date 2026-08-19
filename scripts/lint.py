#!/usr/bin/env python3
"""Deterministic surface linter for humanizer+ru.

The linter is deliberately conservative. It does NOT decide whether Russian
word order, ellipsis, a contrast construction, a rhetorical question, a dash,
or a particle is correct. Those require context.

It reports four kinds of surface findings:

ARTIFACT      technical traces of chatbot/citation copy-paste; reliable gate
AI_PATTERN    repeated formulae or calque-like surface patterns; review only
STYLE_WARNING rhythm/format patterns that may be intentional; review only
METRIC        descriptive measurements; never a language norm

Exit status is non-zero only for ARTIFACT findings.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# Technical traces are the only automatic gate. Keep these specific.
ARTIFACT_PATTERNS = [
    ("openai citation marker", re.compile(r"\boaicite\b", re.I)),
    ("tool turn marker", re.compile(r"\bturn\d+(?:search|news|fetch|view|file|image|product|business)\d+\b", re.I)),
    ("bracket citation placeholder", re.compile(r"\[(?:cite|citation)\s*:\s*\d+[^\]]*\]", re.I)),
    ("chatgpt/openai utm", re.compile(r"utm_source=(?:chatgpt(?:\.com)?|openai)", re.I)),
]

# Single hits are not verdicts. Families are useful only when they cluster.
AI_PHRASE_FAMILIES = {
    "assistant-wrapper": [
        "надеюсь, это поможет",
        "надеюсь, было полезно",
        "дайте знать, если",
        "буду рад помочь",
        "вот краткий обзор",
    ],
    "importance-announcement": [
        "важно отметить",
        "следует подчеркнуть",
        "стоит обратить внимание",
        "нельзя не упомянуть",
        "необходимо учитывать",
    ],
    "pseudo-depth": [
        "если копнуть глубже",
        "глубинная проблема",
        "настоящий вопрос в том",
        "в конечном счёте",
        "вот в чём штука",
    ],
    "video-script": [
        "давайте разберёмся",
        "погрузимся в",
        "вот что нужно знать",
        "перейдём к главному",
        "без лишних слов",
    ],
    "generic-conclusion": [
        "подводя итог",
        "в заключение",
        "резюмируя",
        "будущее выглядит ярким",
        "впереди захватывающие времена",
    ],
    "stack-connector": [
        "кроме того",
        "более того",
        "также стоит",
        "ещё один аспект",
        "ещё одним аспектом",
    ],
}

# These are examples of common calques, not a complete dictionary.
CALQUE_PATTERNS = [
    ("literal possessives", re.compile(
        r"\b(?:свою\s+руку\s+в\s+свой\s+карман|мой\s+ответ|мою\s+встречу|свою\s+руку)\b",
        re.I,
    )),
    ("address a problem", re.compile(r"\bадрес(?:овать|ует|уем|уют|ация)\s+(?:проблем|вопрос)", re.I)),
    ("deliver value", re.compile(r"\bдостав(?:лять|ить|ляет|ляем|ляют)\s+ценност", re.I)),
    ("have influence", re.compile(r"\bиме(?:ть|ет|ют|ем)\s+влияни", re.I)),
    ("be ready by", re.compile(r"\bмогу\s+быть\s+готов(?:ым|ой|ы)?\s+к\b", re.I)),
]

# American/English-language copywriting rhetoric. One occurrence is allowed;
# repeated use in one text is the signal.
SLOGAN_PATTERNS = [
    re.compile(r"\bхорошая новость\?", re.I),
    re.compile(r"\bглавное\?", re.I),
    re.compile(r"\bпочему это важно\?", re.I),
    re.compile(r"\bвот почему это важно\b", re.I),
    re.compile(r"\bодин вопрос\.?\s+один ответ\b", re.I),
    re.compile(r"\bне теория\.?\s+практика\b", re.I),
]

CONTRAST_PATTERNS = [
    re.compile(r"\bне\s+просто\b", re.I),
    re.compile(r"\bне\s+только\b", re.I),
    re.compile(r"\bэто\s+не\b[^.!?\n]{0,100}?\bа\b", re.I),
]

PARCELLATED_ENUM = re.compile(
    r"\b(?:две|три|четыре|пять)\s+[а-яё-]{2,}\s*[.!]\s*(?:либо|или)\b",
    re.I,
)

ASCII_DASH_IN_PROSE = re.compile(r"(?<=[А-Яа-яЁё0-9»)])\s-\s(?=[А-Яа-яЁё0-9«(])")
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")
BOLD_SPAN = re.compile(r"\*\*[^*\n]+\*\*")
URL_OR_CODE = re.compile(r"```.*?```|`[^`\n]+`|https?://\S+", re.S)


def strip_frontmatter(lines: list[str]) -> list[str]:
    if lines and lines[0].strip() == "---":
        for i in range(1, min(len(lines), 50)):
            if lines[i].strip() == "---":
                return [""] * (i + 1) + lines[i + 1 :]
    return lines


def prose_text(text: str) -> str:
    """Remove code/URLs and markdown-only lines, preserve prose punctuation."""
    clean = URL_OR_CODE.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    lines = strip_frontmatter(clean.splitlines())
    kept = []
    for line in lines:
        if not line.strip():
            kept.append("")
            continue
        if re.match(r"^\s*(#|\||[-*+]\s|\d+\.\s|>)", line):
            continue
        kept.append(re.sub(r"\*\*|«|»", "", line))
    return "\n".join(kept)


def sentences(text: str) -> list[str]:
    prose = re.sub(r"\s*\n+\s*", " ", prose_text(text)).strip()
    if not prose:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", prose) if s.strip()]


def word_count(s: str) -> int:
    return len(re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", s))


def first_words(s: str, n: int = 2) -> tuple[str, ...]:
    words = re.findall(r"[A-Za-zА-Яа-яЁё]+", s.lower())
    return tuple(words[:n])


def add(findings: list[dict], kind: str, rule: str, excerpt: str, line: int = 0, note: str = "") -> None:
    findings.append({
        "kind": kind,
        "line": line,
        "rule": rule,
        "excerpt": excerpt[:160],
        "note": note,
    })


def lint(text: str) -> tuple[list[dict], dict]:
    findings: list[dict] = []

    # Artifacts must scan raw text, including URLs.
    for rule, rx in ARTIFACT_PATTERNS:
        for m in rx.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            add(findings, "ARTIFACT", rule, m.group(0), line,
                "technical trace; remove before publication")

    prose = prose_text(text)
    sents = sentences(text)
    lengths = [word_count(s) for s in sents]

    # Phrase families: report hits, but never classify a text by one phrase.
    low = prose.lower()
    for family, phrases in AI_PHRASE_FAMILIES.items():
        hits = [p for p in phrases if p in low]
        if hits:
            add(findings, "AI_PATTERN", family, "; ".join(hits), 0,
                "soft signal; judge by function and clustering")

    # Literal calque candidates.
    for rule, rx in CALQUE_PATTERNS:
        for m in rx.finditer(prose):
            add(findings, "AI_PATTERN", f"calque: {rule}", m.group(0), 0,
                "candidate only; verify idiom, audience and context")

    # Contrast is normal Russian. Warn only on repeated formula use.
    contrast_hits = sum(len(rx.findall(prose)) for rx in CONTRAST_PATTERNS)
    if contrast_hits >= 3:
        add(findings, "STYLE_WARNING", "repeated contrast formula",
            f"{contrast_hits} contrast formulae",
            note="`не X, а Y` is normative; review only repetitive rhetorical use")

    # Slogan rhetoric is also cluster-based.
    slogan_hits = sum(len(rx.findall(prose)) for rx in SLOGAN_PATTERNS)
    if slogan_hits >= 2:
        add(findings, "AI_PATTERN", "slogan question/answer cluster",
            f"{slogan_hits} slogan-like constructions",
            note="one emphatic construction may be intentional")

    # A concrete Russian punctuation/discourse smell: generalizer + full stop + Либо/Или.
    for m in PARCELLATED_ENUM.finditer(prose):
        add(findings, "STYLE_WARNING", "parcellated enumeration", m.group(0), 0,
            "check whether a colon and one syntactic enumeration are more natural")

    # Three or more consecutive micro-sentences: review, not error.
    run: list[str] = []
    for s in sents + ["SENTINEL LONG ENOUGH TO FLUSH"]:
        if word_count(s) <= 4:
            run.append(s)
        else:
            if len(run) >= 3:
                add(findings, "STYLE_WARNING", "short-fragment cluster",
                    " | ".join(run[:5]), 0,
                    "parcellation may be intentional; verify that it adds an accent")
            run = []

    # Repeated sentence starts can reveal SVO-lock or template structure.
    starts = [first_words(s, 2) for s in sents]
    for i in range(len(starts) - 2):
        tri = starts[i : i + 3]
        if tri[0] and tri[0] == tri[1] == tri[2]:
            add(findings, "STYLE_WARNING", "repeated sentence start",
                " / ".join(" ".join(x) for x in tri), 0,
                "candidate for SVO-lock or mechanical parallelism; do not vary words blindly")
            break

    # A weaker SVO-lock proxy: same explicit first token in 3 consecutive normal-length sentences.
    first_tokens = [first_words(s, 1) for s in sents]
    for i in range(len(first_tokens) - 2):
        tri = first_tokens[i : i + 3]
        if tri[0] and tri[0] == tri[1] == tri[2] and all(lengths[j] >= 5 for j in range(i, i + 3)):
            add(findings, "STYLE_WARNING", "repeated explicit subject candidate",
                tri[0][0], 0,
                "check whether Russian context allows a pronoun, zero subject, ellipsis or different information structure")
            break

    # Typography is not an AI verdict. Hyphen surrounded by spaces is worth checking in Russian prose.
    if ASCII_DASH_IN_PROSE.search(prose):
        add(findings, "STYLE_WARNING", "ascii hyphen used as dash", " - ", 0,
            "check typography; do not replace normative em dash with a hyphen for anti-detection")

    # Formatting metrics.
    emoji_count = len(EMOJI.findall(prose))
    bold_count = len(BOLD_SPAN.findall(text))
    dash_count = len(re.findall(r"[—–]", prose))
    colon_count = prose.count(":")
    question_count = prose.count("?")
    words_total = sum(lengths)

    metrics = {
        "sentences": len(sents),
        "words": words_total,
        "sentence_length_median": sorted(lengths)[len(lengths) // 2] if lengths else 0,
        "short_sentences_le_4": sum(1 for x in lengths if x <= 4),
        "dashes": dash_count,
        "colons": colon_count,
        "questions": question_count,
        "emoji": emoji_count,
        "bold_spans": bold_count,
    }

    # Density warning only when extreme, explicitly labelled heuristic.
    if len(sents) >= 6 and dash_count >= 5 and dash_count > len(sents) / 2:
        add(findings, "STYLE_WARNING", "high dash density",
            f"{dash_count} dashes / {len(sents)} sentences", 0,
            "heuristic only: inspect whether the same dash construction repeats")

    return findings, metrics


def self_test() -> None:
    normative = "Это не ошибка в расчёте, а ошибка в исходных данных. Первый вариант дорогой. Второй — быстрее."
    f, _ = lint(normative)
    assert not [x for x in f if x["kind"] == "ARTIFACT"], f
    assert not [x for x in f if x["rule"] == "repeated contrast formula"], f

    enum = "С такими курсами обычно две беды. Либо чистая теория. Либо пересказ пересказа."
    f, _ = lint(enum)
    assert any(x["rule"] == "parcellated enumeration" for x in f), f
    assert any(x["rule"] == "short-fragment cluster" for x in f), f

    contrasts = "Это не просто курс, а опыт. Это не просто опыт, а путь. Это не просто путь, а философия."
    f, _ = lint(contrasts)
    assert any(x["rule"] == "repeated contrast formula" for x in f), f

    artifact = "Текст с oaicite и ?utm_source=chatgpt.com"
    f, _ = lint(artifact)
    assert len([x for x in f if x["kind"] == "ARTIFACT"]) >= 2, f

    calque = "Он положил свою руку в свой карман. Я дал ему мой ответ после того, как закончил мою встречу."
    f, _ = lint(calque)
    assert any(x["rule"].startswith("calque:") for x in f), f

    print("self-test: OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    findings, metrics = lint(text)

    if args.as_json:
        print(json.dumps({"findings": findings, "metrics": metrics}, ensure_ascii=False, indent=2))
    else:
        if findings:
            for f in findings:
                loc = f"line {f['line']}" if f["line"] else "text"
                print(f"{f['kind']:13} {loc:10} {f['rule']}: {f['excerpt']}")
                if f["note"]:
                    print(f"  {f['note']}")
        else:
            print("no deterministic surface findings")

        print("\nmetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        artifacts = [f for f in findings if f["kind"] == "ARTIFACT"]
        if artifacts:
            print("\ngate failed: technical chatbot artifacts remain")
        else:
            print("\ngate passed: no technical chatbot artifacts")
            print("soft findings still require contextual Russian-language review")

    if any(f["kind"] == "ARTIFACT" for f in findings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
