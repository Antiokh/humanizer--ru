#!/usr/bin/env python3
"""Conservative surface linter for humanizer+ru.

The linter does not decide what is grammatical or "human" by regex. It only
finds surface candidates for contextual review.

Kinds:
  ARTIFACT       technical chatbot/citation traces; the only automatic gate
  AI_PATTERN     repeated formulae or calque-like patterns
  NATIVE_WARNING formally possible but potentially synthetic/native-unfriendly
  STYLE_WARNING rhythm/format patterns that may be intentional

Descriptive metrics are returned separately. Exit status is non-zero only when
ARTIFACT findings remain.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ARTIFACT_PATTERNS = [
    ("openai citation marker", re.compile(r"\boaicite\b", re.I)),
    ("tool turn marker", re.compile(r"\bturn\d+(?:search|news|fetch|view|file|image|product|business)\d+\b", re.I)),
    ("bracket citation placeholder", re.compile(r"\[(?:cite|citation)\s*:\s*\d+[^\]]*\]", re.I)),
    ("chatgpt/openai utm", re.compile(r"utm_source=(?:chatgpt(?:\.com)?|openai)", re.I)),
]

AI_PHRASE_FAMILIES = {
    "assistant-wrapper": [
        "надеюсь, это поможет", "надеюсь, было полезно", "дайте знать, если",
        "буду рад помочь", "вот краткий обзор",
    ],
    "importance-announcement": [
        "важно отметить", "следует подчеркнуть", "стоит обратить внимание",
        "нельзя не упомянуть", "необходимо учитывать",
    ],
    "pseudo-depth": [
        "если копнуть глубже", "глубинная проблема", "настоящий вопрос в том",
        "в конечном счёте", "вот в чём штука",
    ],
    "video-script": [
        "давайте разберёмся", "погрузимся в", "вот что нужно знать",
        "перейдём к главному", "без лишних слов",
    ],
    "generic-conclusion": [
        "подводя итог", "в заключение", "резюмируя",
        "будущее выглядит ярким", "впереди захватывающие времена",
    ],
    "stack-connector": [
        "кроме того", "более того", "также стоит", "ещё один аспект",
        "ещё одним аспектом",
    ],
}

CALQUE_PATTERNS = [
    ("literal possessives", re.compile(
        r"\b(?:свою\s+руку\s+в\s+свой\s+карман|мой\s+ответ|мою\s+встречу|свою\s+руку)\b", re.I
    )),
    ("address a problem", re.compile(r"\bадрес(?:овать|ует|уем|уют|ация)\s+(?:проблем|вопрос)", re.I)),
    ("deliver value", re.compile(r"\bдостав(?:лять|ить|ляет|ляем|ляют)\s+ценност", re.I)),
    ("have influence", re.compile(r"\bиме(?:ть|ет|ют|ем)\s+влияни", re.I)),
    ("be ready by", re.compile(r"\bмогу\s+быть\s+готов(?:ым|ой|ы)?\s+к\b", re.I)),
]

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

# Candidate for factoring common material out of a contrast:
# "не ошибка в X, а ошибка в Y" -> "ошибка не в X, а в Y"
# "не меняем X, а меняем Y" -> "меняем не X, а Y"
REPEATED_CONTRAST_HEAD = re.compile(
    r"\bне\s+(?P<head>[а-яё]{4,})\b(?P<left>[^.!?\n]{0,90}?),\s*а\s+(?P=head)\b",
    re.I,
)
REPEATED_CONTRAST_STOP = {"только", "просто", "очень", "столько", "сколько"}

PARCELLATED_ENUM = re.compile(
    r"\b(?:две|три|четыре|пять)\s+[а-яё-]{2,}\s*[.!]\s*(?:либо|или)\b", re.I
)

ASCII_DASH_IN_PROSE = re.compile(r"(?<=[А-Яа-яЁё0-9»)])\s-\s(?=[А-Яа-яЁё0-9«(])")
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")
BOLD_SPAN = re.compile(r"\*\*[^*\n]+\*\*")
URL_OR_CODE = re.compile(r"```.*?```|`[^`\n]+`|https?://\S+", re.S)


def strip_frontmatter(lines: list[str]) -> list[str]:
    if lines and lines[0].strip() == "---":
        for i in range(1, min(len(lines), 50)):
            if lines[i].strip() == "---":
                return [""] * (i + 1) + lines[i + 1:]
    return lines


def prose_text(text: str) -> str:
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
        "excerpt": excerpt[:180],
        "note": note,
    })


def lint(text: str) -> tuple[list[dict], dict]:
    findings: list[dict] = []

    for rule, rx in ARTIFACT_PATTERNS:
        for m in rx.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            add(findings, "ARTIFACT", rule, m.group(0), line,
                "technical trace; remove before publication")

    prose = prose_text(text)
    sents = sentences(text)
    lengths = [word_count(s) for s in sents]
    low = prose.lower()

    for family, phrases in AI_PHRASE_FAMILIES.items():
        hits = [p for p in phrases if p in low]
        if hits:
            add(findings, "AI_PATTERN", family, "; ".join(hits),
                note="soft signal; judge by function and clustering")

    for rule, rx in CALQUE_PATTERNS:
        for m in rx.finditer(prose):
            add(findings, "AI_PATTERN", f"calque: {rule}", m.group(0),
                note="candidate only; verify idiom, audience and context")

    # Native Russian compression: repeated head on both sides of "а".
    for m in REPEATED_CONTRAST_HEAD.finditer(prose):
        head = m.group("head").lower()
        if head not in REPEATED_CONTRAST_STOP:
            add(findings, "NATIVE_WARNING", "repeated common element in contrast", m.group(0),
                note=(
                    f"common «{head}» may be factorable: compare a version where it is said once; "
                    "then re-check word order and information focus"
                ))

    contrast_hits = sum(len(rx.findall(prose)) for rx in CONTRAST_PATTERNS)
    if contrast_hits >= 3:
        add(findings, "STYLE_WARNING", "repeated contrast formula",
            f"{contrast_hits} contrast formulae",
            note="`не X, а Y` is normative; review only repetitive rhetorical use")

    slogan_hits = sum(len(rx.findall(prose)) for rx in SLOGAN_PATTERNS)
    if slogan_hits >= 2:
        add(findings, "AI_PATTERN", "slogan question/answer cluster",
            f"{slogan_hits} slogan-like constructions",
            note="one emphatic construction may be intentional")

    for m in PARCELLATED_ENUM.finditer(prose):
        add(findings, "NATIVE_WARNING", "parcellated enumeration", m.group(0),
            note="check whether a colon and one syntactic enumeration are more natural")

    run: list[str] = []
    for s in sents + ["SENTINEL LONG ENOUGH TO FLUSH"]:
        if word_count(s) <= 4:
            run.append(s)
        else:
            if len(run) >= 3:
                add(findings, "STYLE_WARNING", "short-fragment cluster",
                    " | ".join(run[:5]),
                    note="parcellation may be intentional; verify that it adds an accent")
            run = []

    starts = [first_words(s, 2) for s in sents]
    for i in range(len(starts) - 2):
        tri = starts[i:i + 3]
        if tri[0] and tri[0] == tri[1] == tri[2]:
            add(findings, "NATIVE_WARNING", "repeated sentence start",
                " / ".join(" ".join(x) for x in tri),
                note="candidate for repeated explicit context/SVO-lock; do not vary words blindly")
            break

    first_tokens = [first_words(s, 1) for s in sents]
    for i in range(len(first_tokens) - 2):
        tri = first_tokens[i:i + 3]
        if tri[0] and tri[0] == tri[1] == tri[2] and all(lengths[j] >= 5 for j in range(i, i + 3)):
            add(findings, "NATIVE_WARNING", "repeated explicit subject candidate",
                tri[0][0],
                note="check pronoun, zero subject, ellipsis or different information structure")
            break

    if ASCII_DASH_IN_PROSE.search(prose):
        add(findings, "STYLE_WARNING", "ascii hyphen used as dash", " - ",
            note="check typography; do not replace normative em dash for anti-detection")

    dash_count = len(re.findall(r"[—–]", prose))
    words_total = sum(lengths)
    metrics = {
        "sentences": len(sents),
        "words": words_total,
        "sentence_length_median": sorted(lengths)[len(lengths) // 2] if lengths else 0,
        "short_sentences_le_4": sum(1 for x in lengths if x <= 4),
        "dashes": dash_count,
        "colons": prose.count(":"),
        "questions": prose.count("?"),
        "emoji": len(EMOJI.findall(prose)),
        "bold_spans": len(BOLD_SPAN.findall(text)),
    }

    if len(sents) >= 6 and dash_count >= 5 and dash_count > len(sents) / 2:
        add(findings, "STYLE_WARNING", "high dash density",
            f"{dash_count} dashes / {len(sents)} sentences",
            note="heuristic only: inspect whether the same dash construction repeats")

    return findings, metrics


def self_test() -> None:
    # Grammatically valid but native layer should notice duplicated common head.
    synthetic = "Это не ошибка в расчёте, а ошибка в исходных данных."
    f, _ = lint(synthetic)
    assert any(x["kind"] == "NATIVE_WARNING" and "common element" in x["rule"] for x in f), f
    assert not [x for x in f if x["kind"] == "ARTIFACT"], f

    compressed = "Это ошибка не в расчёте, а в исходных данных. Первый вариант дорогой. Второй — быстрее."
    f, _ = lint(compressed)
    assert not [x for x in f if x["rule"] == "repeated common element in contrast"], f

    verb_repeat = "Мы не меняем цену, а меняем условия."
    f, _ = lint(verb_repeat)
    assert any(x["rule"] == "repeated common element in contrast" for x in f), f

    enum = "С такими курсами обычно две беды. Либо чистая теория. Либо пересказ пересказа."
    f, _ = lint(enum)
    assert any(x["rule"] == "parcellated enumeration" for x in f), f

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

    text = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
    findings, metrics = lint(text)

    if args.as_json:
        print(json.dumps({"findings": findings, "metrics": metrics}, ensure_ascii=False, indent=2))
    else:
        if findings:
            for f in findings:
                loc = f"line {f['line']}" if f["line"] else "text"
                print(f"{f['kind']:14} {loc:10} {f['rule']}: {f['excerpt']}")
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
            print("soft/native findings still require contextual Russian-language review")

    if any(f["kind"] == "ARTIFACT" for f in findings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
