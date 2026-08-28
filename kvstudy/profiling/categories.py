from __future__ import annotations

from dataclasses import dataclass


CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}
FUNCTION_POS = {"DET", "ADP", "AUX", "CCONJ", "SCONJ", "PART"}
QUESTION_WORDS = {"who", "what", "which", "where", "when", "why", "how", "whom", "whose"}
NEGATIONS = {"no", "not", "never", "neither", "nor", "n't"}
POS_NAMES = {
    "NOUN": "common_noun",
    "PROPN": "proper_noun",
    "VERB": "verb",
    "ADJ": "adjective",
    "ADV": "adverb",
    "NUM": "number",
    "DET": "determiner",
    "ADP": "adposition",
    "AUX": "auxiliary",
    "CCONJ": "coordinating_conjunction",
    "SCONJ": "subordinating_conjunction",
    "PART": "particle",
    "PRON": "pronoun",
    "PUNCT": "punctuation",
    "INTJ": "interjection",
    "SYM": "symbol",
}


@dataclass(frozen=True)
class TargetAnnotation:
    coarse_category: str
    fine_category: str
    pos: str
    dependency: str
    is_first_subtoken: bool
    subtoken_index: int


def lexical_categories(token) -> tuple[str, str]:
    """Return broad and interpretable lexical categories for a spaCy token."""
    lower = token.lower_
    if token.is_punct:
        return "special", "punctuation"
    if token.dep_ == "neg" or lower in NEGATIONS:
        return "special", "negation"
    if lower in QUESTION_WORDS:
        return "special", "question_word"
    if token.like_num:
        return "content", "number"
    if token.pos_ == "PRON":
        return "special", "pronoun"
    if token.pos_ in CONTENT_POS:
        return "content", POS_NAMES[token.pos_]
    if token.pos_ in FUNCTION_POS:
        return "function", POS_NAMES[token.pos_]
    return "other", POS_NAMES.get(token.pos_, "other")


def annotate_targets(
    text: str,
    offsets: list[tuple[int, int]],
    target_start: int,
    nlp,
) -> list[TargetAnnotation]:
    """Align target subwords with spaCy labels without parsing the full document."""
    char_start = offsets[target_start][0]
    while char_start > 0 and not text[char_start - 1].isspace():
        char_start -= 1
    char_end = offsets[-1][1]
    doc = nlp(text[char_start:char_end])
    annotations = [TargetAnnotation("other", "other", "X", "", True, 0)
                   for _ in range(len(offsets) - target_start)]

    for token in doc:
        absolute_start = char_start + token.idx
        absolute_end = absolute_start + len(token)
        members = [
            index for index in range(len(offsets))
            if offsets[index][1] > absolute_start and offsets[index][0] < absolute_end
        ]
        coarse, fine = lexical_categories(token)
        for subtoken_index, index in enumerate(members):
            if index < target_start:
                continue
            annotations[index - target_start] = TargetAnnotation(
                coarse_category=coarse,
                fine_category=fine,
                pos=token.pos_ or "X",
                dependency=token.dep_,
                is_first_subtoken=subtoken_index == 0,
                subtoken_index=subtoken_index,
            )
    return annotations
