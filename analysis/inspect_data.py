from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
print("\n=== GeoEarthFM DATA INSPECTOR ===\n")

eu = ROOT / "EuroSAT_RGB"
if eu.exists():
    classes = [p for p in eu.iterdir() if p.is_dir()]
    print(f"EuroSAT classes: {len(classes)}")
    total = 0
    for c in sorted(classes):
        n = sum(1 for p in c.iterdir() if p.is_file())
        total += n
        print(f"  {c.name:25s} {n:6d}")
    print("Total:", total)
else:
    print("EuroSAT_RGB not found:", eu)

st = ROOT / "dataset"
print("\nSTTORM root:", st)
for split in ["train", "test"]:
    split_dir = st / split
    print(f"\n{split.upper()}:")
    if not split_dir.exists():
        print("  MISSING")
        continue
    events = [p for p in split_dir.iterdir() if p.is_dir()]
    print("  events:", len(events))
    for event in events:
        seqs = [p for p in event.iterdir() if p.is_dir()]
        for seq in seqs:
            bands = seq / "all_bands"
            mask = seq / "mask.npy"
            if bands.exists():
                imgs = sorted(bands.glob("*.npy"))
                print(f"  {event.name}/{seq.name}: {len(imgs)} images, mask={mask.exists()}")
                if imgs:
                    arr = np.load(imgs[0], mmap_mode="r")
                    print("     first shape:", arr.shape, "dtype:", arr.dtype)
