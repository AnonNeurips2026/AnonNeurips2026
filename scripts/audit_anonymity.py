#!/usr/bin/env python3
"""Audit the review bundle for identity, path, and internal-name leaks."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def joined(*parts: str) -> str:
    return "".join(parts)


FORBIDDEN_TEXT = [
    joined("tak", "h04"),
    joined("/home/", "tak"),
    joined("/Users/", "tak"),
    joined("jee", "woo"),
    joined("jdub", "kim"),
    joined("convergence", "_ai"),
    joined("nvidia", "/workspace"),
    joined("netket", "_pro"),
    joined("NETKET", "_PRO"),
    joined("benign", "_overfitting_nqs"),
    joined("bening", "_overfitting_nqs2"),
    joined("hf", "_results"),
    joined("hugging", "face.co/"),
    joined("github.com/", "jdub"),
    joined("github.com/", "jee"),
]

FORBIDDEN_PDF_TEXT = [
    joined("Key", "note"),
    joined("Quartz", " PDFContext"),
    joined("mac", "OS Version"),
]

FORBIDDEN_PATH_PARTS = {
    ".codex",
    ".git",
    ".venv",
    "__pycache__",
    joined("hf", "_results"),
}

FORBIDDEN_FILENAMES = {
    "AGENTS.md",
}

FORBIDDEN_SUFFIXES = {
    ".pyc",
    ".log",
}


def iter_files() -> list[Path]:
    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        base = Path(dirpath)
        rel_parts = set(base.relative_to(ROOT).parts)
        bad_parts = sorted(rel_parts & FORBIDDEN_PATH_PARTS)
        if bad_parts:
            raise SystemExit(f"Forbidden directory present: {base.relative_to(ROOT)}")
        dirnames[:] = [d for d in dirnames if d not in FORBIDDEN_PATH_PARTS]
        for filename in filenames:
            path = base / filename
            if filename in FORBIDDEN_FILENAMES:
                raise SystemExit(f"Forbidden file present: {path.relative_to(ROOT)}")
            if path.suffix in FORBIDDEN_SUFFIXES:
                raise SystemExit(f"Forbidden generated file present: {path.relative_to(ROOT)}")
            if path.name.startswith("delta_bank") and path.suffix == ".pkl":
                raise SystemExit(f"Forbidden raw delta bank present: {path.relative_to(ROOT)}")
            paths.append(path)
    return paths


def find_in_file(path: Path, needles: list[str]) -> list[str]:
    data = path.read_bytes()
    hits = []
    for needle in needles:
        if needle.encode() in data:
            hits.append(needle)
    if path.suffix.lower() == ".pdf":
        for needle in FORBIDDEN_PDF_TEXT:
            if needle.encode() in data:
                hits.append(f"pdf-metadata:{needle}")
    return hits


def main() -> int:
    failures: list[str] = []
    for path in iter_files():
        hits = find_in_file(path, FORBIDDEN_TEXT)
        if hits:
            rel = path.relative_to(ROOT)
            labels = ", ".join(sorted(set(hits)))
            failures.append(f"{rel}: {labels}")

    if failures:
        print("Anonymity audit failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("Anonymity audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
