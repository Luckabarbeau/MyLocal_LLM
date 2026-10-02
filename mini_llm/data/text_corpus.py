"""Generic streaming access to text/code corpora used for pretraining.

The repository originally had a Cosmopedia-specific Parquet reader.  The
pretraining collection now contains several independently downloaded corpora,
so tokenizer training needs a small format-agnostic reader.  This module keeps
that concern separate from tokenization and model training.

Supported inputs are discovered recursively:
- Parquet (preferred for the prepared datasets)
- JSONL / NDJSON
- plain text / markdown
- common source-code files (useful when a GitHub corpus is unpacked as files)

Parquet support is imported lazily so the lightweight unit tests do not require
PyArrow to be installed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence


TEXT_COLUMN_CANDIDATES = (
    "text",
    "content",
    "code",
    "markdown",
    "body",
    "document",
    "raw_content",
    "source",
)

_TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".py",
    ".pyi",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".cu",
    ".cuh",
    ".rs",
    ".go",
    ".java",
    ".kt",
    ".kts",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".rb",
    ".php",
    ".swift",
    ".scala",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".sql",
    ".lua",
    ".r",
    ".R",
    ".jl",
    ".m",
    ".mm",
    ".cs",
    ".fs",
    ".fsx",
    ".vb",
    ".html",
    ".css",
    ".scss",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
}
_JSONL_SUFFIXES = {".jsonl", ".ndjson"}


@dataclass(frozen=True)
class CorpusSourceSpec:
    """Description of one logical corpus source."""

    name: str
    path: Path
    weight: float
    text_column: Optional[str] = None


@dataclass
class CorpusSampleStats:
    """Mutable statistics populated while a streaming sample is consumed."""

    documents: int = 0
    bytes: int = 0


class TextCorpusSource:
    """Recursively stream text documents from a prepared dataset directory."""

    def __init__(
        self,
        name: str,
        path: Path,
        text_column: Optional[str] = None,
        parquet_batch_size: int = 1024,
    ):
        self.name = str(name)
        self.path = Path(path).expanduser().resolve()
        self.text_column = text_column
        self.parquet_batch_size = int(parquet_batch_size)
        if self.parquet_batch_size <= 0:
            raise ValueError("parquet_batch_size must be positive")
        if not self.path.exists():
            raise FileNotFoundError(
                f"corpus source {self.name!r} does not exist: {self.path}"
            )

    @staticmethod
    def _is_supported_file(path: Path) -> bool:
        suffix = path.suffix.lower()
        return (
            suffix == ".parquet"
            or suffix in _JSONL_SUFFIXES
            or suffix in {s.lower() for s in _TEXT_SUFFIXES}
        )

    def discover_files(self) -> List[Path]:
        if self.path.is_file():
            files = [self.path] if self._is_supported_file(self.path) else []
        else:
            files = [
                path
                for path in self.path.rglob("*")
                if path.is_file() and self._is_supported_file(path)
            ]
        if not files:
            raise ValueError(
                f"no supported corpus files found for {self.name!r} under {self.path}"
            )
        return sorted(files)

    @staticmethod
    def choose_text_column(
        columns: Sequence[str], explicit: Optional[str] = None
    ) -> str:
        columns = list(columns)
        if explicit is not None:
            if explicit not in columns:
                raise ValueError(
                    f"requested text column {explicit!r} is not present; "
                    f"available columns: {columns}"
                )
            return explicit
        for candidate in TEXT_COLUMN_CANDIDATES:
            if candidate in columns:
                return candidate
        raise ValueError(
            "could not infer text/code column. "
            f"Available columns: {columns}; candidates: {TEXT_COLUMN_CANDIDATES}"
        )

    def _iter_parquet(self, path: Path) -> Iterator[str]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Parquet corpus reading requires pyarrow>=16.0"
            ) from exc

        parquet = pq.ParquetFile(path)
        column = self.choose_text_column(parquet.schema_arrow.names, self.text_column)
        for batch in parquet.iter_batches(
            batch_size=self.parquet_batch_size, columns=[column]
        ):
            values = batch.column(0).to_pylist()
            for value in values:
                if value is None:
                    continue
                text = value if isinstance(value, str) else str(value)
                if text:
                    yield text

    def _iter_jsonl(self, path: Path) -> Iterator[str]:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if isinstance(value, str):
                    text = value
                elif isinstance(value, dict):
                    column = self.choose_text_column(value.keys(), self.text_column)
                    raw = value.get(column)
                    text = "" if raw is None else str(raw)
                else:
                    continue
                if text:
                    yield text

    @staticmethod
    def _iter_text_file(path: Path) -> Iterator[str]:
        text = path.read_text(encoding="utf-8", errors="replace")
        if text:
            yield text

    def iter_texts(
        self,
        *,
        seed: int = 42,
        shuffle_files: bool = True,
    ) -> Iterator[str]:
        files = self.discover_files()
        if shuffle_files:
            rng = random.Random(f"{int(seed)}:{self.name}")
            rng.shuffle(files)
        for path in files:
            suffix = path.suffix.lower()
            if suffix == ".parquet":
                yield from self._iter_parquet(path)
            elif suffix in _JSONL_SUFFIXES:
                yield from self._iter_jsonl(path)
            else:
                yield from self._iter_text_file(path)

    def inspect(self) -> Dict[str, object]:
        files = self.discover_files()
        suffix_counts: Dict[str, int] = {}
        for path in files:
            suffix = path.suffix.lower()
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1

        result: Dict[str, object] = {
            "name": self.name,
            "path": str(self.path),
            "file_count": len(files),
            "suffix_counts": suffix_counts,
        }
        first_parquet = next((p for p in files if p.suffix.lower() == ".parquet"), None)
        if first_parquet is not None:
            try:
                import pyarrow.parquet as pq

                columns = pq.ParquetFile(first_parquet).schema_arrow.names
                result["parquet_columns"] = list(columns)
                result["selected_text_column"] = self.choose_text_column(
                    columns, self.text_column
                )
            except ImportError:
                result["parquet_columns"] = "pyarrow unavailable"
        return result


def normalized_weights(specs: Sequence[CorpusSourceSpec]) -> Dict[str, float]:
    if not specs:
        raise ValueError("at least one corpus source is required")
    if any(spec.weight < 0 for spec in specs):
        raise ValueError("corpus weights must be non-negative")
    total = sum(float(spec.weight) for spec in specs)
    if total <= 0:
        raise ValueError("sum of corpus weights must be positive")
    return {spec.name: float(spec.weight) / total for spec in specs}


def weighted_text_sample(
    specs: Sequence[CorpusSourceSpec],
    *,
    total_bytes: int,
    seed: int = 42,
    parquet_batch_size: int = 1024,
    stats: Optional[Dict[str, CorpusSampleStats]] = None,
) -> Iterator[str]:
    """Yield a deterministic approximately weighted sample without materializing it.

    Each source receives a byte budget proportional to its normalized weight.  The
    scheduler always advances the source furthest behind its target fraction, so
    corpora are interleaved rather than emitted in seven large contiguous runs.
    A document may take a source slightly past its byte quota; this avoids slicing
    code/documents at arbitrary byte boundaries.
    """

    if total_bytes <= 0:
        raise ValueError("total_bytes must be positive")
    weights = normalized_weights(specs)
    quotas = {
        name: max(1, int(round(total_bytes * weight)))
        for name, weight in weights.items()
    }
    # Correct rounding so the quota sum stays close to the requested total.
    delta = total_bytes - sum(quotas.values())
    if delta:
        largest = max(weights, key=weights.get)
        quotas[largest] = max(1, quotas[largest] + delta)

    if stats is None:
        stats = {}
    iterators: Dict[str, Iterator[str]] = {}
    active: Dict[str, bool] = {}
    for spec in specs:
        stats.setdefault(spec.name, CorpusSampleStats())
        source = TextCorpusSource(
            spec.name,
            spec.path,
            text_column=spec.text_column,
            parquet_batch_size=parquet_batch_size,
        )
        iterators[spec.name] = source.iter_texts(seed=seed, shuffle_files=True)
        active[spec.name] = True

    while any(active.values()):
        candidates = [
            spec.name
            for spec in specs
            if active[spec.name] and stats[spec.name].bytes < quotas[spec.name]
        ]
        if not candidates:
            break
        # 0 means no progress; 1 means quota reached. Select the most-behind source.
        name = min(
            candidates,
            key=lambda item: stats[item].bytes / float(quotas[item]),
        )
        try:
            text = next(iterators[name])
        except StopIteration:
            active[name] = False
            continue
        if not text:
            continue
        nbytes = len(text.encode("utf-8"))
        if nbytes == 0:
            continue
        stats[name].documents += 1
        stats[name].bytes += nbytes
        yield text
