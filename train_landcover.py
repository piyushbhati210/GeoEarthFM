import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from tqdm import tqdm

from geoearthfm.config import *
from geoearthfm.datasets import EuroSATDataset
from geoearthfm.models import LandCoverModel
from geoearthfm.utils import seed_everything, device, save_checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS_CLASSIFICATION)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_CLASSIFICATION)
    args = parser.parse_args()

    seed_everything(SEED)
    dev = device()
    print("Device:", dev)

    ds = EuroSATDataset(EUROSAT_DIR, train=True, image_size=IMAGE_SIZE)
    n_val = int(0.15 * len(ds))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    # validation should not use augmentation
    val_base = EuroSATDataset(EUROSAT_DIR, train=False, image_size=IMAGE_SIZE)
    val_ds = torch.utils.data.Subset(val_base, val_ds.indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=NUM_WORKERS)

    model = LandCoverModel(num_classes=len(ds.classes)).to(dev)
    opt = optim.AdamW(model.parameters(), lr=LR_CLASSIFICATION, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        correct = total = 0
        for x,y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} train"):
            x,y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logits,_ = model(x)
            loss = criterion(logits,y)
            loss.backward()
            opt.step()
            running += loss.item()*x.size(0)
            correct += (logits.argmax(1)==y).sum().item()
            total += y.numel()

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(dev), y.to(dev)
                logits,_ = model(x)
                vc += (logits.argmax(1)==y).sum().item()
                vt += y.numel()

        acc = vc/max(1,vt)
        print(f"loss={running/max(1,total):.4f} train_acc={correct/max(1,total):.4f} val_acc={acc:.4f}")

        if acc > best:
            best = acc
            save_checkpoint(model, OUTPUT_DIR/"landcover_best.pt",
                            classes=ds.classes, val_accuracy=best)

    print("Best validation accuracy:", best)
    print("Saved:", OUTPUT_DIR/"landcover_best.pt")

if __name__ == "__main__":
    main()
