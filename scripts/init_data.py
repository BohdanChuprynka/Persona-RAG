"""Bootstrap the data/ directory tree.

Idempotent. Safe to re-run. Creates every directory the runtime expects,
so a fresh clone can immediately run `make ingest` without crashing on
missing dirs. Does NOT touch any existing content.
"""

from __future__ import annotations

from pathlib import Path

DIRS = [
    "data",
    "data/raw",
    "data/raw/telegram",
    "data/raw/instagram",
    "data/processed",
    "data/eval",
]


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    for d in DIRS:
        path = repo_root / d
        path.mkdir(parents=True, exist_ok=True)
        keep = path / ".gitkeep"
        if not keep.exists() and d != "data":
            keep.touch()
        print(f"ok: {d}")
    print("\ndata/ tree ready. Drop your exports into data/raw/telegram/ and data/raw/instagram/.")


if __name__ == "__main__":
    main()
