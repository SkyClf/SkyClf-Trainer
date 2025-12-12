import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from trainer.config import TrainerConfig
from trainer.labels import SKYSTATE_TO_ID
from trainer.output import next_version_dir
from trainer.torch_dataset import load_samples, SkyStateDataset


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_samples(samples, val_ratio: float, seed: int):
    rnd = random.Random(seed)
    idx = list(range(len(samples)))
    rnd.shuffle(idx)
    n_val = max(1, int(len(samples) * val_ratio))
    val_idx = set(idx[:n_val])
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
    return train, val


def main():
    p = argparse.ArgumentParser(description="SkyClf Trainer (PyTorch)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val", type=float, default=0.2)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    cfg = TrainerConfig()
    cfg.validate()

    set_seed(args.seed)

    samples = load_samples(cfg)
    if len(samples) < 10:
        raise SystemExit(f"Not enough labeled samples: {len(samples)} (label more images)")

    train_samples, val_samples = split_samples(samples, args.val, args.seed)

    tf_train = transforms.Compose([
        transforms.Resize((args.img, args.img)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((args.img, args.img)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    ds_train = SkyStateDataset(cfg, train_samples, transform=tf_train)
    ds_val = SkyStateDataset(cfg, val_samples, transform=tf_val)

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} total={len(samples)} train={len(train_samples)} val={len(val_samples)}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(SKYSTATE_TO_ID))
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    models_root = Path(cfg.models_dir) / "skystate"
    models_root.mkdir(parents=True, exist_ok=True)
    latest_ckpt = models_root / "latest.pt"

    if args.resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        print(f"resumed from {latest_ckpt}")

    def eval_epoch():
        model.eval()
        total = 0
        correct = 0
        loss_sum = 0.0
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                loss_sum += float(loss.item()) * x.size(0)
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == y).sum().item())
                total += int(x.size(0))
        return loss_sum / max(1, total), correct / max(1, total)

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        seen = 0
        loss_sum = 0.0

        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            loss_sum += float(loss.item()) * x.size(0)
            seen += int(x.size(0))

        train_loss = loss_sum / max(1, seen)
        val_loss, val_acc = eval_epoch()
        print(f"epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc

        torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, latest_ckpt)

    out_dir = next_version_dir(models_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save weights
    torch.save(model.state_dict(), out_dir / "model.pt")

    # Save metadata + classes
    (out_dir / "classes.json").write_text(json.dumps(SKYSTATE_TO_ID, indent=2), encoding="utf-8")
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "samples_total": len(samples),
        "samples_train": len(train_samples),
        "samples_val": len(val_samples),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "img_size": args.img,
        "best_val_acc": best_acc,
        "device": str(device),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n✅ wrote: {out_dir}")


if __name__ == "__main__":
    main()
