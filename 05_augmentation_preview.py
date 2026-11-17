# Step 5: see what data augmentation does.
# Augmentation means showing the model changed copies of the same picture (flipped, brighter...)
# so that it sees more variety than the photos alone give it.
# The important part: when the picture is flipped, the boxes have to move with it.
# YOLO does all this by itself during training. This step only shows the idea with my own code.
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

# pick a training tile that has at least 4 objects
tiles = sorted(glob.glob(os.path.join(config.tile_dir, "images", "train", "*.jpg")))
random.shuffle(tiles)
for path in tiles:
    label_lines = [l for l in open(path.replace("images", "labels").replace(".jpg", ".txt")) if l.strip()]
    if len(label_lines) >= 4:
        break
image = cv2.imread(path)
T = image.shape[0]

# boxes in pixels: x1, y1, x2, y2
boxes = []
for line in label_lines:
    c, cx, cy, bw, bh = [float(v) for v in line.split()]
    boxes.append((cx * T - bw * T / 2, cy * T - bh * T / 2, cx * T + bw * T / 2, cy * T + bh * T / 2))

# each change returns a new picture and the new boxes
flip_left_right = (cv2.flip(image, 1), [(T - x2, y1, T - x1, y2) for x1, y1, x2, y2 in boxes])
flip_up_down = (cv2.flip(image, 0), [(x1, T - y2, x2, T - y1) for x1, y1, x2, y2 in boxes])
turn_90 = (cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), [(T - y2, x1, T - y1, x2) for x1, y1, x2, y2 in boxes])
brighter = (cv2.convertScaleAbs(image, alpha=1.0, beta=50), boxes)     # boxes stay where they are
darker = (cv2.convertScaleAbs(image, alpha=1.0, beta=-50), boxes)

versions = [("original", (image, boxes)), ("flip left-right", flip_left_right), ("flip up-down", flip_up_down),
            ("turn 90 degrees", turn_90), ("brighter", brighter), ("darker", darker)]

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, (title, (img, bxs)) in zip(axes.flatten(), versions):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for x1, y1, x2, y2 in bxs:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(config.results_dir, "05_augmentation.jpg"), dpi=80)
print("Saved results/05_augmentation.jpg  (the red boxes must stay on the same objects)")
