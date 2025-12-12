import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset

from trainer.labels import encode_skystate
from trainer.config import TrainerConfig


@dataclass
class Sample:
    image_id: str
    skystate: str
    meteor: int  # 0/1


def load_samples(cfg: TrainerConfig) -> List[Sample]:
    conn = sqlite3.connect(cfg.labels_db)
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT image_id, skystate, meteor
        FROM labels
        ORDER BY labeled_at ASC
    """).fetchall()
    conn.close()

    return [Sample(str(i), str(s), int(m)) for (i, s, m) in rows]


class SkyStateDataset(Dataset):
    def __init__(self, cfg: TrainerConfig, samples: List[Sample], transform=None):
        self.cfg = cfg
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[object, int]:
        s = self.samples[idx]
        # IMPORTANT: use image_id.jpg, so we ignore Windows paths stored in DB
        img_path = Path(self.cfg.images_dir) / f"{s.image_id}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        y = encode_skystate(s.skystate)
        return img, y
