# Small helper functions that several steps use.
import os
import glob
import math
import cv2
import numpy as np
from PIL import Image


def read_dota_label(path, img_w, img_h):
    """Read one label file from the DOTA download.
    Every line is: class_id x1 y1 x2 y2 x3 y3 x4 y4 (the four corners of a tilted box,
    each number between 0 and 1). Returns a list of (class_id, corners), corners in pixels."""
    objects = []
    if not os.path.exists(path):
        return objects
    for line in open(path):
        parts = line.split()
        if len(parts) != 9:
            continue
        class_id = int(parts[0])
        numbers = [float(p) for p in parts[1:]]
        corners = []
        for i in range(0, 8, 2):
            corners.append((numbers[i] * img_w, numbers[i + 1] * img_h))
        objects.append((class_id, corners))
    return objects


def box_from_corners(corners):
    """Turn four corners into the straight box around them: x1, y1, x2, y2."""
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return min(xs), min(ys), max(xs), max(ys)


def iou(a, b):
    """Overlap of two boxes (x1, y1, x2, y2): area of overlap divided by area of both together."""
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / (area_a + area_b - overlap)


def tile_starts(length, tile, step):
    """Where tiles start along one side of the image. The last tile is pushed back
    so that it ends exactly at the image edge."""
    starts = list(range(0, max(length - tile, 0) + 1, step))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def find_demo_image(raw_dir, split, class_ids):
    """Pick a photo for the demo pictures: one that has as many of my classes as possible,
    and among those, plenty of objects (but not a 1000-car parking lot)."""
    best_path = None
    best_score = -1
    best_count = 0
    for path in sorted(glob.glob(os.path.join(raw_dir, "images", split, "*"))):
        name = os.path.splitext(os.path.basename(path))[0]
        w, h = Image.open(path).size
        objects = read_dota_label(os.path.join(raw_dir, "labels", split, name + ".txt"), w, h)
        found_classes = [o[0] for o in objects if o[0] in class_ids]
        score = len(set(found_classes)) * 1000 + min(len(found_classes), 100)
        if score > best_score:
            best_path = path
            best_score = score
            best_count = len(found_classes)
    return best_path, best_count


def remove_duplicates(found, iou_limit=0.5):
    """The same object can be found in two overlapping tiles. For each class, keep the most
    confident box and drop boxes that overlap it too much (non-maximum suppression)."""
    kept = []
    for class_id in set(f[5] for f in found):
        these = [f for f in found if f[5] == class_id]
        boxes = [[f[0], f[1], f[2] - f[0], f[3] - f[1]] for f in these]   # x, y, width, height
        scores = [f[4] for f in these]
        keep = cv2.dnn.NMSBoxes(boxes, scores, 0.0, iou_limit)
        for k in np.array(keep).flatten():
            kept.append(these[int(k)])
    return kept


def detect_big_image(model, image, tile=640, step=512, conf=0.25):
    """Run the model on a big image by cutting it into tiles, the same way as for training.
    Returns a list of [x1, y1, x2, y2, confidence, class_id] in pixels of the big image."""
    h, w = image.shape[:2]
    # if the image is smaller than one tile, pad it with black
    pad_bottom = max(tile - h, 0)
    pad_right = max(tile - w, 0)
    if pad_bottom or pad_right:
        image = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        h, w = image.shape[:2]

    crops = []
    offsets = []
    for y in tile_starts(h, tile, step):
        for x in tile_starts(w, tile, step):
            crops.append(image[y:y + tile, x:x + tile])
            offsets.append((x, y))

    found = []
    # a few tiles at a time, so the memory stays small
    for i in range(0, len(crops), 16):
        results = model.predict(crops[i:i + 16], conf=conf, imgsz=tile, verbose=False)
        for r, (ox, oy) in zip(results, offsets[i:i + 16]):
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            for box, score, class_id in zip(boxes, scores, classes):
                # float() turns numpy numbers into plain Python numbers, which tables and json can handle
                found.append([float(box[0]) + ox, float(box[1]) + oy, float(box[2]) + ox, float(box[3]) + oy,
                              float(score), int(class_id)])
    return remove_duplicates(found)


def draw_detections(image, detections, names):
    """Draw boxes and labels on a copy of the image."""
    colors = [(0, 0, 255), (255, 120, 0), (0, 200, 0), (0, 200, 255)]   # BGR
    out = image.copy()
    thickness = max(2, image.shape[1] // 800)
    for x1, y1, x2, y2, score, class_id in detections:
        color = colors[class_id % len(colors)]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        text = names[class_id] + " " + str(round(score, 2))
        cv2.putText(out, text, (int(x1), max(int(y1) - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * thickness, color, thickness)
    return out


# One degree of latitude is about 111,195 metres (earth radius 6371 km, times pi, divided by 180).
metres_per_degree = 111195


def pixel_to_latlon(x, y, lat0, lon0, metres_per_pixel):
    """Position of pixel (x, y) on a north-up image whose top-left corner is at (lat0, lon0).
    Works for small areas only, where the earth is flat enough."""
    lat = lat0 - (y * metres_per_pixel) / metres_per_degree
    # a degree of longitude gets shorter the further you are from the equator
    metres_per_degree_lon = metres_per_degree * math.cos(math.radians(lat0))
    lon = lon0 + (x * metres_per_pixel) / metres_per_degree_lon
    return lat, lon


def metres_between(lat1, lon1, lat2, lon2):
    """Distance between two points on the earth in metres (haversine formula). Used to check the above."""
    r = 6371000
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
