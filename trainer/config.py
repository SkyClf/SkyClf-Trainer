import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present (safe no-op if missing)
load_dotenv()

def env(key: str, default=None):
    return os.getenv(key, default)

class TrainerConfig:
    def __init__(self, data_dir: Path | None = None):
        # Paths
        self.data_dir = data_dir or Path(env("SKYCLF_DATA_DIR", "/data"))
        self.images_dir = self.data_dir / "images"
        self.labels_db = self.data_dir / "labels" / "labels.db"
        self.models_dir = self.data_dir / "models"

        # Trainer behavior
        self.task = env("SKYCLF_TASK", "skystate")   # future: meteor, multi-task
        self.min_samples = int(env("SKYCLF_MIN_SAMPLES", "20"))
        self.verbose = env("SKYCLF_VERBOSE", "0") == "1"

    def validate(self):
        if not self.labels_db.exists():
            raise FileNotFoundError(f"labels.db not found at {self.labels_db}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"images dir not found at {self.images_dir}")
