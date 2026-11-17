# Step 4: write the small file that tells YOLO where the data is, then check the tiles.
# The checks matter: a model trained on wrong labels looks fine until you test it properly.
import os
import glob
import random
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

random.seed(config.seed)
os.makedirs(config.results_dir, exist_ok=True)

# 1. the yaml file (a plain text settings file)
lines = ["path: " + config.tile_dir, "train: images/train", "val: images/val", "names:"]
for i, name in enumerate(config.my_names):
    lines.append(f"  {i}: {name}")
yaml_path = os.path.join(config.tile_dir, "data.yaml")
open(yaml_path, "w").write("\n".join(lines) + "\n")
print("Wrote", yaml_path)

# 2. count tiles and objects in each split
originals = {}
for split in ["train", "val"]:
    tiles = sorted(glob.glob(os.path.join(config.tile_dir, "images", split, "*.jpg")))
    counts = [0] * len(config.my_names)
    for label_file in glob.glob(os.path.join(config.tile_dir, "labels", split, "*.txt")):
        for line in open(label_file):
            if line.strip():
                counts[int(line.split()[0])] += 1
    print(f"{split}: {len(tiles)} tiles, objects per class:", dict(zip(config.my_names, counts)))
    # tile names look like P0879_512_0, so everything before the first "_" is the original photo
    originals[split] = set(os.path.basename(t).split("_")[0] for t in tiles)

# 3. no original photo may appear in both train and val, otherwise the model sees its exam answers
shared = originals["train"] & originals["val"]
print("Photos in both train and val:", len(shared))
assert len(shared) == 0, "train and val share photos, fix the split"

# 4. draw boxes on a few random training tiles and look at them with my own eyes
train_tiles = sorted(glob.glob(os.path.join(config.tile_dir, "images", "train", "*.jpg")))
with_objects = [t for t in train_tiles
                if open(t.replace("images", "labels").replace(".jpg", ".txt")).read().strip()]
sample = random.sample(with_objects, min(8, len(with_objects)))
colors = [(255, 0, 0), (0, 120, 255), (0, 200, 0)]      # RGB
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, path in zip(axes.flatten(), sample):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    for line in open(path.replace("images", "labels").replace(".jpg", ".txt")):
        c, cx, cy, bw, bh = line.split()
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        cv2.rectangle(image, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)),
                      colors[int(c)], 2)
    ax.imshow(image)
    ax.set_title(os.path.basename(path), fontsize=8)
    ax.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(config.results_dir, "04_tile_check.jpg"), dpi=80)
print("Saved results/04_tile_check.jpg  (open it and check that the boxes sit on the objects)")
