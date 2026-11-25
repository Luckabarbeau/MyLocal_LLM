"""Fail closed if the imported full source differs from its recorded file hashes."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOURCE = BASE.parent / "third_party" / "strata-712fec6b4297"
REPO = BASE.parents[2]
PIN = "712fec6b4297c9cfd32ab43daaa2979a9aa01edf"


def verify():
    provenance = json.loads((BASE / "source_provenance.json").read_text())
    assert provenance["revision"] == PIN
    expected = provenance["files"]
    actual = {p.relative_to(SOURCE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in SOURCE.rglob("*") if p.is_file()}
    if actual != expected:
        changed = sorted(k for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k))
        raise SystemExit(f"Strata source integrity failed: {changed[:20]}")
    reference = REPO / "references" / "strata-712fec6b4297"
    manifest = json.loads((reference / "manifest.json").read_text())
    assert manifest["revision"] == PIN
    for name, item in manifest["files"].items():
        assert actual[name] == item["sha256"], name
        assert (SOURCE / name).read_bytes() == (reference / item["snapshot"]).read_bytes(), name
    print(f"Verified {len(actual)} pinned source files and all {len(manifest['files'])} reference snapshots")


if __name__ == "__main__":
    verify()
