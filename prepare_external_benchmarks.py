#!/usr/bin/env python3
"""Prepare fixed external language-model benchmarks for MyLocal_LLM.

The script intentionally stores plain text / JSONL snapshots under a user-chosen
benchmark root so later checkpoint comparisons use exactly the same examples.

Datasets prepared:
  * Penn Treebank test text
  * PG-19 test books (100 books, one JSON object per book)
  * C4 fixed validation prefix (default: first 5,000 documents)
  * LAMBADA OpenAI English test (5,153 examples)

Only this preparation script needs the optional ``datasets`` package. The
benchmark evaluators themselves do not depend on Hugging Face datasets.
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing optional dependency 'datasets'. Install it in the active "
            "venv with: python -m pip install datasets"
        ) from exc
    return load_dataset


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _report(path: Path, *, records=None):
    extra = "" if records is None else f", records={records:,}"
    print(
        f"Wrote {path} ({path.stat().st_size:,} bytes{extra})\n"
        f"  sha256={_sha256(path)}"
    )


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    _report(path, records=count)
    return count


def prepare_ptb(root: Path):
    print("Preparing Penn Treebank test text...")
    url = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt"
    path = root / "ptb" / "ptb.test.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())
    _report(path)


def prepare_pg19(root: Path):
    load_dataset = _require_datasets()
    print("Preparing PG-19 test books...")
    ds = load_dataset("emozilla/pg19-test", split="test")
    path = root / "pg19-test" / "pg19_test.jsonl"

    def rows():
        for index, row in enumerate(ds):
            yield {
                "id": index,
                "title": row.get("short_book_title", ""),
                "publication_date": row.get("publication_date"),
                "url": row.get("url", ""),
                "text": row["text"],
            }

    _write_jsonl(path, rows())


def prepare_c4(root: Path, documents: int):
    load_dataset = _require_datasets()
    print(f"Preparing fixed C4 validation prefix ({documents:,} documents)...")
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    path = root / "c4-validation" / f"c4_validation_first_{documents}.jsonl"

    def rows():
        for index, row in enumerate(ds):
            if index >= documents:
                break
            yield {
                "id": index,
                "url": row.get("url", ""),
                "timestamp": row.get("timestamp", ""),
                "text": row["text"],
            }

    _write_jsonl(path, rows())


def prepare_lambada(root: Path):
    load_dataset = _require_datasets()
    print("Preparing LAMBADA OpenAI English test...")
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    path = root / "lambada-openai" / "lambada_test_en.jsonl"

    def rows():
        for index, row in enumerate(ds):
            yield {"id": index, "text": row["text"]}

    _write_jsonl(path, rows())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("../benchmarks"),
        help="Benchmark output root (default: ../benchmarks).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["all", "ptb", "pg19", "c4", "lambada"],
        default=["all"],
        help="Datasets to prepare.",
    )
    parser.add_argument(
        "--c4-documents",
        type=int,
        default=5000,
        help="Number of deterministic leading C4 validation documents to snapshot.",
    )
    args = parser.parse_args()

    if args.c4_documents <= 0:
        raise ValueError("--c4-documents must be positive")

    selected = set(args.datasets)
    if "all" in selected:
        selected = {"ptb", "pg19", "c4", "lambada"}

    args.root.mkdir(parents=True, exist_ok=True)
    if "ptb" in selected:
        prepare_ptb(args.root)
    if "pg19" in selected:
        prepare_pg19(args.root)
    if "c4" in selected:
        prepare_c4(args.root, args.c4_documents)
    if "lambada" in selected:
        prepare_lambada(args.root)


if __name__ == "__main__":
    main()
