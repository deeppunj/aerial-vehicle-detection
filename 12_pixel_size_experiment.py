# Step 12: what happens when the camera pixels get bigger on the ground?
#
# The ground size of one pixel (GSD, ground sample distance) is
#     GSD = detector pixel pitch x height above the ground / focal length
# so flying twice as high, or using a lens with half the focal length, doubles the GSD.
# A plane that covers 40 pixels at 0.5 m per pixel covers only 20 pixels at 1 m per pixel.
#
# The experiment: I take the same 300 validation tiles and make each one look as if it was
# taken with bigger ground pixels (x1, x1.33, x2, x2.86, x4). For each pixel size I count how
# many real objects the trained model still finds (recall) and how many of its boxes are right
# (precision). Every tile is seen at every pixel size, so each tile is compared with itself
# (a paired experiment). That removes the tile-to-tile differences from the comparison.
#
# How a bigger pixel is made: shrink the tile with INTER_AREA, which averages the pixels in each
# block, just as one big detector pixel collects the light of the whole block. Then scale it back
# up to 640 x 640, so the model gets the same input size and the label boxes stay where they are.
# Only the detail is lost. Lens blur and sensor noise are not simulated.
#
# The 95% intervals come from a bootstrap: draw 300 tiles with replacement, recompute, repeat
# 1,000 times, and keep the middle 95% of the results.
import os
import sys
import glob
import random
import cv2
import numpy as np
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

# the model from training, or the copy in model/ if the runs folder is not there
model_path = os.path.join(config.runs_dir, run_name, "weights", "best.pt")
if not os.path.exists(model_path):
    model_path = os.path.join(config.here, "model", "best.pt")
print("Model:", model_path)
model = YOLO(model_path)

# bigger ground pixel = smaller shrink factor. x2 means each new pixel covers 2 x 2 old pixels.
shrink_factors = [1.0, 0.75, 0.5, 0.35, 0.25]
pixel_size_names = ["x1", "x1.33", "x2", "x2.86", "x4"]

random.seed(config.seed)
tiles = sorted(glob.glob(os.path.join(config.tile_dir, "images", "val", "*.jpg")))
tiles = random.sample(tiles, min(300, len(tiles)))
print("Tiles:", len(tiles))


def make_pixels_bigger(image, factor):
    if factor == 1.0:
        return image
    h, w = image.shape[:2]
    small = cv2.resize(image, (round(w * factor), round(h * factor)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def read_real_boxes(path, w, h):
    real = []
    for line in open(path.replace("images", "labels").replace(".jpg", ".txt")):
        if not line.strip():
            continue
        c, cx, cy, bw, bh = line.split()
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        real.append((int(c), (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)))
    return real


# counts[level][class] is a list with one entry per tile: [found, real, correct boxes, boxes]
n_classes = len(config.my_names)
counts = np.zeros((len(shrink_factors), len(tiles), n_classes, 4), dtype=int)

for t, path in enumerate(tiles):
    image = cv2.imread(path)
    h, w = image.shape[:2]
    real = read_real_boxes(path, w, h)
    for level, factor in enumerate(shrink_factors):
        test_image = make_pixels_bigger(image, factor)
        result = model.predict(test_image, imgsz=config.tile_size, conf=config.conf, verbose=False)[0]
        predicted = [(int(c), tuple(box)) for box, c in
                     zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy())]
        for c, box in real:
            counts[level, t, c, 1] += 1
            if any(pc == c and tools.iou(box, pb) >= 0.5 for pc, pb in predicted):
                counts[level, t, c, 0] += 1
        for pc, pb in predicted:
            counts[level, t, pc, 3] += 1
            if any(c == pc and tools.iou(box, pb) >= 0.5 for c, box in real):
                counts[level, t, pc, 2] += 1
    if (t + 1) % 50 == 0:
        print(t + 1, "tiles done")


def share(a, b):
    return a / b if b > 0 else np.nan


# bootstrap: the same random tile draws are used for every pixel size, to keep the pairing
rng = np.random.default_rng(config.seed)
draws = [rng.integers(0, len(tiles), len(tiles)) for _ in range(1000)]

rows = []
groups = [(name, [c]) for c, name in enumerate(config.my_names)] + [("all", list(range(n_classes)))]
for name, class_list in groups:
    for level, size_name in enumerate(pixel_size_names):
        per_tile = counts[level][:, class_list, :].sum(axis=1)     # one row per tile
        base = counts[0][:, class_list, :].sum(axis=1)              # the same tiles at x1
        recall = share(per_tile[:, 0].sum(), per_tile[:, 1].sum())
        precision = share(per_tile[:, 2].sum(), per_tile[:, 3].sum())
        boot_recall = []
        boot_drop = []
        for d in draws:
            r = share(per_tile[d, 0].sum(), per_tile[d, 1].sum())
            r0 = share(base[d, 0].sum(), base[d, 1].sum())
            boot_recall.append(r)
            boot_drop.append(r - r0)
        rows.append({
            "class": name,
            "pixel size": size_name,
            "objects": int(per_tile[:, 1].sum()),
            "recall": recall,
            "recall low 95%": np.nanpercentile(boot_recall, 2.5),
            "recall high 95%": np.nanpercentile(boot_recall, 97.5),
            "change vs x1": recall - share(base[:, 0].sum(), base[:, 1].sum()),
            "change low 95%": np.nanpercentile(boot_drop, 2.5),
            "change high 95%": np.nanpercentile(boot_drop, 97.5),
            "precision": precision,
        })

table = pd.DataFrame(rows)
print(table.round(3).to_string(index=False))
os.makedirs(config.results_dir, exist_ok=True)
table.round(3).to_csv(os.path.join(config.results_dir, "12_pixel_size.csv"), index=False)

# chart, left: recall against pixel size, one line per class, 95% interval as error bars.
# right: the paired change against x1. The left intervals overlap because some tiles are much
# easier than others; the right panel compares each tile with itself, so its intervals are narrower.
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
xs = np.array([1 / f for f in shrink_factors])
for i, (name, _) in enumerate(groups):
    part = table[table["class"] == name]
    shift = (i - 1.5) * 0.04          # small sideways shift so the error bars do not hide each other
    style = "k--" if name == "all" else "-o"
    ax.errorbar(xs + shift, part["recall"],
                yerr=[part["recall"] - part["recall low 95%"], part["recall high 95%"] - part["recall"]],
                fmt=style, capsize=3, label=name)
    ax2.errorbar(xs + shift, part["change vs x1"] * 100,
                 yerr=[(part["change vs x1"] - part["change low 95%"]) * 100,
                       (part["change high 95%"] - part["change vs x1"]) * 100],
                 fmt=style, capsize=3, label=name)
for a in (ax, ax2):
    a.set_xticks(xs)
    a.set_xticklabels(pixel_size_names)
    a.set_xlabel("ground pixel size compared with the original photo")
ax.set_ylabel("recall (share of real objects found)")
ax.set_ylim(0, 1)
ax.set_title("Recall at each pixel size")
ax.legend(loc="lower left")
ax2.axhline(0, color="grey", linewidth=0.8)
ax2.set_ylabel("change in recall against x1 (percentage points)")
ax2.set_title("Same tiles, compared with themselves")
fig.tight_layout()
fig.savefig(os.path.join(config.results_dir, "12_pixel_size.png"), dpi=120)
print("Saved results/12_pixel_size.png")

# example picture: the same small area of one tile at every pixel size
best_tile = max(range(len(tiles)), key=lambda t: counts[0, t, :, 1].sum())
image = cv2.imread(tiles[best_tile])
real = read_real_boxes(tiles[best_tile], image.shape[1], image.shape[0])
cx = int(np.mean([(b[0] + b[2]) / 2 for _, b in real]))
cy = int(np.mean([(b[1] + b[3]) / 2 for _, b in real]))
x0 = min(max(cx - 100, 0), image.shape[1] - 200)
y0 = min(max(cy - 100, 0), image.shape[0] - 200)
pieces = []
for factor, size_name in zip(shrink_factors, pixel_size_names):
    crop = make_pixels_bigger(image, factor)[y0:y0 + 200, x0:x0 + 200]
    crop = cv2.resize(crop, (400, 400), interpolation=cv2.INTER_NEAREST)
    cv2.putText(crop, size_name, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    pieces.append(crop)
cv2.imwrite(os.path.join(config.results_dir, "12_pixel_size_examples.jpg"), np.hstack(pieces))
print("Saved results/12_pixel_size_examples.jpg")
