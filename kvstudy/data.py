from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from .config import Config

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")


@dataclass
class SourceDocument:
    doc_id: str
    corpus: str
    text: str
    evidence_spans: list[tuple[int, int]]


def _pg19() -> Iterator[SourceDocument]:
    # Parquet mirror of deepmind/pg19. The original loader fetches one Google
    # Storage object per book, which is prohibitively slow on many clusters.
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    for index, row in enumerate(ds):
        text = row["text"]
        yield SourceDocument(str(row.get("short_book_title", row.get("url", index))), "pg19", text, [])


def _wikipedia() -> Iterator[SourceDocument]:
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    for row in ds:
        yield SourceDocument(str(row.get("id", row.get("title", "wiki"))), "wikipedia", row["text"], [])


def _hotpotqa() -> Iterator[SourceDocument]:
    ds = load_dataset("hotpot_qa", "distractor", split="validation", streaming=True)
    for row in ds:
        support = {(title, int(sent)) for title, sent in zip(
            row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]
        )}
        chunks: list[str] = []
        evidence: list[tuple[int, int]] = []
        cursor = 0
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
            heading = f"{title}. "
            chunks.append(heading)
            cursor += len(heading)
            for sent_id, sentence in enumerate(sentences):
                start = cursor
                chunks.append(sentence)
                cursor += len(sentence)
                if (title, sent_id) in support:
                    evidence.append((start, cursor))
                chunks.append(" ")
                cursor += 1
        question = f"\nQuestion: {row['question']}"
        chunks.append(question)
        yield SourceDocument(str(row["id"]), "hotpotqa", "".join(chunks), evidence)


LOADERS = {"pg19": _pg19, "wikipedia": _wikipedia, "hotpotqa": _hotpotqa}


def _clean(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"(?:\S+\s+){12,}\S+", lambda m: " " if "|" in m.group() else m.group(), text)
    return re.sub(r"\s+", " ", text).strip()


def prepare(cfg: Config) -> tuple[Path, Path]:
    data_dir = cfg.data_dir or cfg.output_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    contexts_path = data_dir / "contexts.jsonl"
    df: Counter[str] = Counter()
    written = 0
    with contexts_path.open("w", encoding="utf-8") as handle:
        for corpus, count in cfg.corpora.items():
            if corpus not in LOADERS:
                raise ValueError(f"unknown corpus: {corpus}")
            accepted = 0
            for doc in tqdm(LOADERS[corpus](), total=count, desc=f"sample {corpus}"):
                text = doc.text.strip() if doc.evidence_spans else _clean(doc.text)
                if not doc.evidence_spans:
                    # A generous character bound avoids caching entire books that
                    # the tokenizer will immediately truncate.
                    text = text[: cfg.max_length * 8]
                if len(text) < 300:
                    continue
                # Preserve evidence offsets only when cleaning did not alter QA text.
                evidence = doc.evidence_spans if text == doc.text.strip() else []
                words = {w.lower() for w in WORD_RE.findall(text)}
                df.update(words)
                record = {"id": f"{corpus}:{doc.doc_id}", "corpus": corpus, "text": text,
                          "evidence_spans": evidence}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                accepted += 1
                written += 1
                if accepted >= count:
                    break
            if accepted < count:
                raise RuntimeError(f"{corpus}: requested {count}, found {accepted}")
    idf = {word: float(__import__("math").log((written + 1) / (freq + 1)) + 1) for word, freq in df.items()}
    idf_path = data_dir / "idf.json"
    idf_path.write_text(json.dumps({"documents": written, "idf": idf}), encoding="utf-8")
    return contexts_path, idf_path


def read_contexts(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)
