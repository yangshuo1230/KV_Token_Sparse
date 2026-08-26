from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spacy
from transformers import PreTrainedTokenizerBase

CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "NUM"}
FUNCTION_POS = {"DET", "ADP", "AUX", "CCONJ", "SCONJ", "PART"}
SPECIAL = {"negation", "pronoun", "question", "punctuation", "number"}
CLAUSE_DEPS = {"advcl", "ccomp", "xcomp", "acl", "relcl"}


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    kind: str = "semantic"


@dataclass
class AnnotatedText:
    input_ids: list[int]
    offsets: list[tuple[int, int]]
    categories: list[str]
    pos_tags: list[str]
    idf: np.ndarray
    segments: list[Segment]
    evidence_blocks_by_size: dict[int, set[int]]


def load_nlp():
    try:
        return spacy.load("en_core_web_sm", disable=["ner"])
    except OSError as exc:
        raise RuntimeError("spaCy model missing; run: python -m spacy download en_core_web_sm") from exc


def _category(token) -> str:
    lower = token.lower_
    if token.is_punct:
        return "punctuation"
    if token.like_num:
        return "number"
    if token.dep_ == "neg" or lower in {"no", "not", "never", "neither", "nor"}:
        return "negation"
    if token.pos_ == "PRON":
        return "question" if lower in {"who", "what", "which", "where", "when", "why", "how"} else "pronoun"
    if token.pos_ in CONTENT_POS:
        return "content"
    if token.pos_ in FUNCTION_POS:
        return "function"
    return "other"


def _initial_char_spans(doc) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for sent in doc.sents:
        cuts = [sent.start_char]
        for token in sent:
            if token.dep_ in CLAUSE_DEPS and token.i > sent.start + 2:
                cuts.append(token.idx)
            elif token.text in {";", ":"}:
                cuts.append(token.idx + len(token))
        cuts.append(sent.end_char)
        cuts = sorted(set(cuts))
        spans.extend((a, b) for a, b in zip(cuts, cuts[1:]) if a < b)
    return spans


def annotate(text: str, tokenizer: PreTrainedTokenizerBase, nlp, idf_map: dict[str, float],
             min_tokens: int, max_tokens: int, max_length: int,
             block_sizes: list[int], evidence_spans: list[list[int]]) -> AnnotatedText:
    encoded = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length,
                        return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = [tuple(x) for x in encoded["offset_mapping"]]
    doc = nlp(text[: offsets[-1][1] if offsets else 0])
    categories = ["other"] * len(ids)
    pos_tags = ["OTHER"] * len(ids)
    weights = np.ones(len(ids), dtype=np.float32)
    for token in doc:
        cat = _category(token)
        value = idf_map.get(token.lower_, 1.0)
        for idx, (start, end) in enumerate(offsets):
            if end > token.idx and start < token.idx + len(token):
                categories[idx] = cat
                pos_tags[idx] = token.pos_
                weights[idx] = value

    char_spans = _initial_char_spans(doc)
    rough: list[Segment] = []
    for char_start, char_end in char_spans:
        members = [i for i, (a, b) in enumerate(offsets) if b > char_start and a < char_end]
        if not members:
            continue
        start, end = members[0], members[-1] + 1
        for left in range(start, end, max_tokens):
            rough.append(Segment(left, min(left + max_tokens, end)))
    merged: list[Segment] = []
    for seg in rough:
        if merged and seg.end - seg.start < min_tokens and seg.end - merged[-1].start <= max_tokens:
            merged[-1] = Segment(merged[-1].start, seg.end)
        else:
            merged.append(seg)
    segments = [s for s in merged if s.end - s.start >= min_tokens]

    evidence_by_size: dict[int, set[int]] = {}
    for size in block_sizes:
        blocks: set[int] = set()
        for idx, (start, end) in enumerate(offsets):
            if any(end > a and start < b for a, b in evidence_spans):
                blocks.add(idx // size)
        evidence_by_size[size] = blocks
    return AnnotatedText(ids, offsets, categories, pos_tags, weights, segments, evidence_by_size)
