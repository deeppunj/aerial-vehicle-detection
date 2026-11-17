# Step 8: does the model miss the small objects?
# For every real object on the validation tiles I check if the model found it
# (same class, boxes overlap by at least 50%), then I count the hits per object size.
# Expectation: tiny objects (few pixels) are found much less often than big ones.
import os
import sys
import glob
import random
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
import config
import tools

run_name = "optical_v1"
if len(sys.argv) > 1:
    run_name = sys.argv[1]

model = YOLO(os.path.join(config.runs_dir, run_name, "weights", "best.pt"))

random.seed(config.seed)
tiles = sorted(glob.glob(os.path.join(config.tile_dir, "images", "val", "*.jpg")))
tiles = random.sample(tiles, min(300, len(tiles)))

size_groups = [(0, 16), (16, 32), (32, 64), (64, 100000)]
group_names = ["under 16 px", "16-32 px", "32-64 px", "64 px and more"]
total = {}     # (class, group) -> number of real objects
found = {}     # (class, group) -> how many of them the model found

for path in tiles:
    h, w = cv2.imread(path).shape[:2]
    # the real objects (from the label file), converted back to pixels
    real = []
    for line in open(path.replace("images", "labels").replace(".jpg", ".txt")):
        if not line.strip():
            continue
        c, cx, cy, bw, bh = line.split()
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        real.append((int(c), (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)))

    # what the model says
    result = model.predict(path, imgsz=config.tile_size, conf=config.conf, verbose=False)[0]
    predicted = []
    for box, c in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
        predicted.append((int(c), tuple(box)))

    for c, box in real:
        long_side = max(box[2] - box[0], box[3] - box[1])
        for g, (low, high) in enumerate(size_groups):
            if low <= long_side < high:
                key = (config.my_names[c], group_names[g])
                total[key] = total.get(key, 0) + 1
                hit = any(pc == c and tools.iou(box, pb) >= 0.5 for pc, pb in predicted)
                if hit:
                    found[key] = found.get(key, 0) + 1

rows = []
for name in config.my_names:
    for g in group_names:
        n = total.get((name, g), 0)
        recall = found.get((name, g), 0) / n if n > 0 else None
        rows.append({"class": name, "object size": g, "objects": n, "recall": recall})
table = pd.DataFrame(rows)
print(table.round(2).to_string(index=False))
os.makedirs(config.results_dir, exist_ok=True)
table.round(3).to_csv(os.path.join(config.results_dir, "08_recall_by_size.csv"), index=False)

# bar chart: one group of bars per size, one bar per class
fig, ax = plt.subplots(figsize=(8, 4.5))
width = 0.25
for i, name in enumerate(config.my_names):
    part = table[table["class"] == name]
    xs = [j + i * width for j in range(len(group_names))]
    ax.bar(xs, part["recall"].fillna(0), width=width, label=name)
ax.set_xticks([j + width for j in range(len(group_names))])
ax.set_xticklabels(group_names)
ax.set_ylabel("recall (share of real objects found)")
ax.set_ylim(0, 1)
ax.set_title("Small objects are harder")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(config.results_dir, "08_recall_by_size.png"), dpi=120)
print("Saved results/08_recall_by_size.png")
