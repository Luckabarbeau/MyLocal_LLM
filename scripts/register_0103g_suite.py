#!/usr/bin/env python3
"""Register 0103G tests without replacing a locally customized regression runner.

Safe to rerun. This utility edits only the CPU_SUITES dictionary, keeping all
existing suites, CUDA-memory safeguards, and local modifications intact.
"""

from __future__ import annotations

import ast
import os
import stat
from pathlib import Path
import tempfile


TEST = "tests/test_unified_benchmark_0103g.py"
SUITE = "unified_benchmark"
ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_0103a_regressions.py"


def update_runner(path: Path) -> bool:
    contents = path.read_text(encoding="utf-8")
    tree = ast.parse(contents, filename=str(path))
    candidates = [
        stmt.value
        for stmt in tree.body
        if isinstance(stmt, (ast.Assign, ast.AnnAssign))
        and (any(isinstance(target, ast.Name) and target.id == "CPU_SUITES"
                 for target in stmt.targets)
             if isinstance(stmt, ast.Assign)
             else isinstance(stmt.target, ast.Name) and stmt.target.id == "CPU_SUITES")
    ]
    if len(candidates) != 1 or not isinstance(candidates[0], ast.Dict):
        raise RuntimeError("Expected exactly one literal CPU_SUITES dictionary; no changes made")
    suites = candidates[0]
    keys = [key.value if isinstance(key, ast.Constant) else None for key in suites.keys]
    if SUITE in keys:
        index = keys.index(SUITE)
        values = [node.value for node in ast.walk(suites.values[index])
                  if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        if TEST not in values:
            raise RuntimeError(f"Existing {SUITE} suite doesn't include {TEST}; no changes made")
        print(f"0103G regression tier already registered in {path}")
        return False
    # Preserve existing suite structure even if 0103A..0103F changed its order.
    lines = contents.splitlines(keepends=True)
    close_index = suites.end_lineno - 1  # AST line numbers are one-based.
    closing = lines[close_index]
    if closing.strip() != "}":
        raise RuntimeError("CPU_SUITES dictionary has an unusual layout; no changes made")
    indentation = (lines[suites.keys[0].lineno - 1][:suites.keys[0].col_offset]
                   if suites.keys else "    ")
    addition = (f'{indentation}"{SUITE}": (\n'
                f'{indentation}    "{TEST}",\n'
                f'{indentation}),\n')
    new_text = ''.join(lines[:close_index]) + addition + ''.join(lines[close_index:])
    compile(new_text, str(path), "exec")
    # Atomic replacement avoids leaving a truncated runner if interrupted.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".runner-0103g-", delete=False) as tmp:
        tmp.write(new_text)
        temp_path = Path(tmp.name)
    try:
        temp_path.chmod(stat.S_IMODE(path.stat().st_mode))
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"Registered {SUITE}: {TEST} in {path}")
    return True


if __name__ == "__main__":
    update_runner(RUNNER)
