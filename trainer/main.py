import argparse
import json
from pathlib import Path

from trainer.config import TrainerConfig
from trainer.dataset import Dataset
from trainer.labels import SKYSTATE_TO_ID
from trainer.output import next_version_dir

def main():
    parser = argparse.ArgumentParser(description="SkyClf Trainer")
    parser.add_argument(
        "--data",
        type=Path,
        help="Override data directory (default: SKYCLF_DATA_DIR or /data)",
    )
    parser.add_argument("--export", action="store_true", help="Export manifest into /data/models")
    args = parser.parse_args()

    cfg = TrainerConfig(data_dir=args.data)
    cfg.validate()

    ds = Dataset(cfg)
    stats = ds.stats()

    print("🌌 SkyClf Trainer")
    print("----------------")
    print(f"Data dir: {cfg.data_dir}")
    print(f"Total labeled images: {stats['total']}")

    for k, v in stats["skystates"].items():
        print(f"  {k:>15}: {v}")

    print(f"  {'meteor':>15}: {stats['meteor_yes']}")

    if stats["total"] < cfg.min_samples:
        print(f"\n⚠️  Warning: only {stats['total']} samples "
              f"(recommended ≥ {cfg.min_samples})")

    if args.export:
        base = cfg.data_dir / "models" / "skystate"
        out_dir = next_version_dir(base)
        out_dir.mkdir(parents=True, exist_ok=True)

        # write classes mapping
        (out_dir / "classes.json").write_text(json.dumps(SKYSTATE_TO_ID, indent=2), encoding="utf-8")

        # write manifest
        n = ds.export_manifest(out_dir / "manifest.jsonl")

        # write meta
        meta = {"samples": n}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f"\n✅ Exported {n} samples to: {out_dir}")


if __name__ == "__main__":
    main()
