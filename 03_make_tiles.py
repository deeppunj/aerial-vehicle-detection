# Step 3: cut the big aerial images into small square tiles and write new label files.
#
# Why: the photos are up to 4000 pixels wide. The model works on 640 x 640 pictures.
# If I shrink a whole photo to 640 px, a car shrinks to a dot. Cutting keeps the full detail.
#
# What happens to the labels:
#   - DOTA marks each object with four corners (a tilted box). YOLO wants a straight box,
#     so I take the smallest straight box around the four corners.
#   - Only plane, ship and large vehicle are kept. They get the new ids 0, 1, 2.
#   - Each tile gets its own label file with positions relative to the tile.
import os
import glob
import random
import cv2
import config
import tools

random.seed(config.seed)


def make_tiles(split, max_images):
    image_files = sorted(glob.glob(os.path.join(config.raw_dir, "images", split, "*")))
    label_folder = os.path.join(config.raw_dir, "labels", split)
    out_images = os.path.join(config.tile_dir, "images", split)
    out_labels = os.path.join(config.tile_dir, "labels", split)
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    if max_images:
        image_files = image_files[:max_images]

    T = config.tile_size
    n_tiles = 0
    n_objects = 0

    for number, path in enumerate(image_files):
        name = os.path.splitext(os.path.basename(path))[0]
        image = cv2.imread(path)
        h, w = image.shape[:2]

        # all objects of my classes as straight boxes in pixels: (new_class, x1, y1, x2, y2)
        boxes = []
        for class_id, corners in tools.read_dota_label(os.path.join(label_folder, name + ".txt"), w, h):
            if class_id in config.my_class_ids:
                x1, y1, x2, y2 = tools.box_from_corners(corners)
                boxes.append((config.my_class_ids.index(class_id), x1, y1, x2, y2))
        if len(boxes) == 0:
            continue

        # a small image gets black padding so that it fits at least one tile
        if h < T or w < T:
            image = cv2.copyMakeBorder(image, 0, max(T - h, 0), 0, max(T - w, 0), cv2.BORDER_CONSTANT, value=(0, 0, 0))
            h, w = image.shape[:2]

        for ty in tools.tile_starts(h, T, config.tile_step):
            for tx in tools.tile_starts(w, T, config.tile_step):
                lines = []
                for new_class, x1, y1, x2, y2 in boxes:
                    # the part of the box that is inside this tile
                    cx1 = max(x1, tx)
                    cy1 = max(y1, ty)
                    cx2 = min(x2, tx + T)
                    cy2 = min(y2, ty + T)
                    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
                        continue
                    inside = ((cx2 - cx1) * (cy2 - cy1)) / ((x2 - x1) * (y2 - y1))
                    if inside < config.min_inside:
                        continue       # most of the object is in another tile
                    # YOLO format: class, centre x, centre y, width, height (all between 0 and 1)
                    cx = ((cx1 + cx2) / 2 - tx) / T
                    cy = ((cy1 + cy2) / 2 - ty) / T
                    bw = (cx2 - cx1) / T
                    bh = (cy2 - cy1) / T
                    lines.append(f"{new_class} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                if len(lines) == 0 and random.random() > config.empty_tile_share:
                    continue           # most empty tiles are skipped

                tile_name = f"{name}_{tx}_{ty}"
                cv2.imwrite(os.path.join(out_images, tile_name + ".jpg"), image[ty:ty + T, tx:tx + T],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                open(os.path.join(out_labels, tile_name + ".txt"), "w").write("\n".join(lines))
                n_tiles += 1
                n_objects += len(lines)

        if (number + 1) % 50 == 0:
            print(f"  {split}: {number + 1} of {len(image_files)} images done")

    print(f"{split}: {n_tiles} tiles, {n_objects} objects")


make_tiles("train", config.max_train_images)
make_tiles("val", config.max_val_images)
print("Tiles are in", config.tile_dir)
