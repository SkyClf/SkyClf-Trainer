from pathlib import Path

def next_version_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("v")])
    if not existing:
        return base / "v0001"
    last = existing[-1].name[1:]
    n = int(last)
    return base / f"v{n+1:04d}"
