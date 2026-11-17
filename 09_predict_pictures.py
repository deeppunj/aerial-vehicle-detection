# Step 9: look at what the model actually does.
#   a) a few small validation tiles with the model's boxes drawn on them
#   b) one full-size photo: it is cut into tiles, every tile goes through the model,
#      the boxes are moved back to full-size positions, and doubles from the overlaps are removed
import os
import sys
import glob
import random
import cv2
from ultralytics import YOLO
import config
import tools

run_name = "optical_v1"
if len(sys.argv) > 1:
    run_name = sys.argv[1]

model = YOLO(os.path.join(config.runs_dir, run_name, "weights", "best.pt"))
out_folder = os.path.join(config.results_dir, "09_predictions")
os.makedirs(out_folder, exist_ok=True)

# a) small tiles
random.seed(config.seed)
tiles = sorted(glob.glob(os.path.join(config.tile_dir, "images", "val", "*.jpg")))
for path in random.sample(tiles, min(8, len(tiles))):
    result = model.predict(path, imgsz=config.tile_size, conf=config.conf, verbose=False)[0]
    cv2.imwrite(os.path.join(out_folder, os.path.basename(path)), result.plot())    # .plot() draws the boxes
print("Saved tile predictions in", out_folder)

# b) one big photo: a validation photo with a mix of classes
path, count = tools.find_demo_image(config.raw_dir, "val", config.my_class_ids)
print("Big photo:", os.path.basename(path), "with", count, "real objects of my classes")
image = cv2.imread(path)
detections = tools.detect_big_image(model, image, config.tile_size, config.tile_step, config.conf)
print("The model found", len(detections), "objects")
drawn = tools.draw_detections(image, detections, config.my_names)
scale = 1800 / max(drawn.shape[:2])
drawn = cv2.resize(drawn, None, fx=scale, fy=scale)
cv2.imwrite(os.path.join(config.results_dir, "09_big_photo.jpg"), drawn)
print("Saved results/09_big_photo.jpg")
