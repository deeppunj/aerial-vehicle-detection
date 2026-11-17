# Step 2: look at the data before doing anything with it.
# Prints how many objects of each class there are and how big they are in pixels,
# and saves two pictures into the results folder.
import os
import glob
import statistics
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import config
import tools

os.makedirs(config.results_dir, exist_ok=True)

image_files = sorted(glob.glob(os.path.join(config.raw_dir, "images", "train", "*")))
label_folder = os.path.join(config.raw_dir, "labels", "train")
print("Training images:", len(image_files))

counts = {}    # class name -> number of objects (all 15 classes)
sizes = {}     # class name -> long side of every box in pixels (my 3 classes only)
busiest = (0, None)

for path in image_files:
    name = os.path.splitext(os.path.basename(path))[0]
    w, h = Image.open(path).size           # only reads the file header, so it is fast
    objects = tools.read_dota_label(os.path.join(label_folder, name + ".txt"), w, h)
    mine = 0
    for class_id, corners in objects:
        class_name = config.dota_names[class_id]
        counts[class_name] = counts.get(class_name, 0) + 1
        if class_id in config.my_class_ids:
            x1, y1, x2, y2 = tools.box_from_corners(corners)
            sizes.setdefault(class_name, []).append(max(x2 - x1, y2 - y1))
            mine += 1
    if mine > busiest[0]:
        busiest = (mine, path)

print("\nObjects per class in the training set (all 15 classes):")
for class_name, n in sorted(counts.items(), key=lambda item: -item[1]):
    marker = "  <- I use this one" if class_name in config.my_names else ""
    print(f"  {class_name:20s} {n:7d}{marker}")

print("\nSize of my three classes (long side of the box, in pixels):")
for class_name in config.my_names:
    s = sizes[class_name]
    small = 100 * sum(1 for v in s if v < 16) / len(s)
    print(f"  {class_name:15s} median {statistics.median(s):5.0f} px   smallest {min(s):4.0f} px   "
          f"largest {max(s):5.0f} px   under 16 px: {small:.0f}%")

# Picture 1: the image with the most objects of my classes, boxes drawn on top
image = cv2.imread(busiest[1])
colors = [(0, 0, 255), (255, 120, 0), (0, 200, 0)]
thickness = max(2, image.shape[1] // 600)
name = os.path.splitext(os.path.basename(busiest[1]))[0]
for class_id, corners in tools.read_dota_label(os.path.join(label_folder, name + ".txt"), image.shape[1], image.shape[0]):
    if class_id in config.my_class_ids:
        color = colors[config.my_class_ids.index(class_id)]
        points = [[int(x), int(y)] for x, y in corners]
        cv2.polylines(image, [np.array(points)], True, color, thickness)
scale = 1600 / max(image.shape[:2])
image = cv2.resize(image, None, fx=scale, fy=scale)
cv2.imwrite(os.path.join(config.results_dir, "02_example_image.jpg"), image)

# Picture 2: how big are the objects?
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 7))
for ax, class_name in zip(axes, config.my_names):
    ax.hist(sizes[class_name], bins=range(0, 160, 4))
    ax.set_ylabel(class_name)
axes[-1].set_xlabel("long side of the box in pixels")
axes[0].set_title("How many pixels do the objects cover?")
fig.tight_layout()
fig.savefig(os.path.join(config.results_dir, "02_object_sizes.png"), dpi=120)
print("\nSaved two pictures in", config.results_dir)
