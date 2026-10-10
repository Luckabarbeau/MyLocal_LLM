"""Generic streaming access to text/code corpora used for pretraining.

The repository originally had a Cosmopedia-specific Parquet reader.  The
pretraining collection now contains several independently downloaded corpora,
so tokenizer training needs a small format-agnostic reader.  This module keeps
that concern separate from tokenization and model training.

Supported inputs are discovered recursively:
- Parquet (preferred for the prepared datasets)
- JSON / JSONL / NDJSON
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
_JSON_SUFFIXES = {".json"}
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
            or suffix in _JSON_SUFFIXES
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

    @staticmethod
    def _iter_json_values(
        path: Path, chunk_size: int = 4 * 1024 * 1024
    ) -> Iterator[object]:
        """Stream values from either JSON-lines or a top-level JSON array.

        Some large prepared corpora (notably Dolma/GitHub exports) use a
        ``.json`` suffix even though the file contains one JSON object per
        line.  Others use an ordinary top-level JSON array.  Loading either
        form with ``json.load`` is unacceptable for multi-gigabyte corpora,
        so this parser incrementally feeds ``JSONDecoder.raw_decode``.
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        decoder = json.JSONDecoder()
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            buffer = ""
            pos = 0
            eof = False
            in_array: Optional[bool] = None

            def refill() -> bool:
                nonlocal buffer, pos, eof
                if eof:
                    return False
                # Discard already-consumed text before appending the next
                # chunk.  This bounds memory to roughly one chunk plus the
                # largest individual JSON record.
                if pos:
                    buffer = buffer[pos:]
                    pos = 0
                chunk = handle.read(chunk_size)
                if not chunk:
                    eof = True
                    return False
                buffer += chunk
                return True

            refill()
            while True:
                while True:
                    while pos < len(buffer) and buffer[pos].isspace():
                        pos += 1
                    if pos < len(buffer) or eof:
                        break
                    refill()

                if in_array is None:
                    if pos >= len(buffer):
                        return
                    in_array = buffer[pos] == "["
                    if in_array:
                        pos += 1

                if in_array:
                    while True:
                        while pos < len(buffer) and buffer[pos].isspace():
                            pos += 1
                        if pos >= len(buffer):
                            if refill():
                                continue
                            raise ValueError(f"unterminated JSON array in {path}")
                        if buffer[pos] == "]":
                            return
                        if buffer[pos] == ",":
                            pos += 1
                            continue
                        break
                else:
                    while pos >= len(buffer) and not eof:
                        refill()
                    if pos >= len(buffer):
                        return

                while True:
                    try:
                        value, end = decoder.raw_decode(buffer, pos)
                    except json.JSONDecodeError as exc:
                        if refill():
                            continue
                        raise ValueError(
                            f"invalid or truncated JSON corpus file {path}: {exc}"
                        ) from exc
                    pos = end
                    yield value
                    break

    def _iter_json(self, path: Path) -> Iterator[str]:
        selected_column: Optional[str] = self.text_column
        for value in self._iter_json_values(path):
            if isinstance(value, str):
                text = value
            elif isinstance(value, dict):
                if selected_column is None:
                    selected_column = self.choose_text_column(value.keys())
                elif selected_column not in value:
                    raise ValueError(
                        f"requested/inferred text column {selected_column!r} is not "
                        f"present in a JSON record from {path}; available columns: "
                        f"{list(value.keys())}"
                    )
                raw = value.get(selected_column)
                text = "" if raw is None else str(raw)
            else:
                continue
            if text:
                yield text

    def _iter_jsonl(self, path: Path) -> Iterator[str]:
        selected_column: Optional[str] = self.text_column
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL record in {path} at line {line_number}: {exc}"
                    ) from exc
                if isinstance(value, str):
                    text = value
                elif isinstance(value, dict):
                    if selected_column is None:
                        selected_column = self.choose_text_column(value.keys())
                    elif selected_column not in value:
                        raise ValueError(
                            f"requested/inferred text column {selected_column!r} is not "
                            f"present in a JSONL record from {path}; available columns: "
                            f"{list(value.keys())}"
                        )
                    raw = value.get(selected_column)
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
            elif suffix in _JSON_SUFFIXES:
                yield from self._iter_json(path)
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

        first_json = next((p for p in files if p.suffix.lower() == ".json"), None)
        if first_json is not None:
            try:
                sample = next(self._iter_json_values(first_json))
            except StopIteration:
                result["json_sample"] = "empty"
            else:
                result["json_sample_type"] = type(sample).__name__
                if isinstance(sample, dict):
                    columns = list(sample.keys())
                    result["json_fields"] = columns
                    result["selected_text_column"] = self.choose_text_column(
                        columns, self.text_column
                    )
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
