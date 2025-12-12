import json
import sqlite3
from pathlib import Path
from collections import Counter
from trainer.config import TrainerConfig
from trainer.labels import encode_skystate

class Dataset:
    def __init__(self, config: TrainerConfig | None = None):
        self.config = config or TrainerConfig()
        self.data_dir = self.config.data_dir
        self.images_dir = self.config.images_dir
        self.db_path = self.config.labels_db

    def load_labels(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        rows = cur.execute("""
            SELECT
              i.id,
              i.path,
              l.skystate,
              l.meteor
            FROM labels l
            JOIN images i ON i.id = l.image_id
        """).fetchall()

        conn.close()
        return rows

    def stats(self):
        rows = self.load_labels()
        total = len(rows)

        skystates = Counter()
        meteors = 0

        for _, _, skystate, meteor in rows:
            skystates[skystate] += 1
            if int(meteor) == 1:
                meteors += 1

        return {
            "total": total,
            "skystates": dict(skystates),
            "meteor_yes": meteors,
        }

    def export_manifest(self, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rows = self.load_labels()
        with out_path.open("w", encoding="utf-8") as f:
            for image_id, image_path, skystate, meteor in rows:
                rec = {
                    "image_id": image_id,
                    "path": image_path,
                    "skystate": skystate,
                    "skystate_id": encode_skystate(skystate),
                    "meteor": int(meteor),
                }
                f.write(json.dumps(rec) + "\n")

        return len(rows)