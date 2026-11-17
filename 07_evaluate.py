# Step 7: measure how good the trained model is on the validation tiles, and how fast it is.
#
# Words used here:
#   precision = of everything the model marked, how much was really an object
#   recall    = of all real objects, how many the model found
#   mAP50     = one overall score from 0 to 1 that combines both (higher is better);
#               a detection counts as correct when its box overlaps the true box by at least 50%
#
# Same arguments as step 6:  python 07_evaluate.py [data.yaml] [run name]
import os
import sys
import glob
import random
import pandas as pd
import torch
from ultralytics import YOLO
import config

data_yaml = os.path.join(config.tile_dir, "data.yaml")
run_name = "optical_v1"
if len(sys.argv) > 1:
    data_yaml = sys.argv[1]
if len(sys.argv) > 2:
    run_name = sys.argv[2]

os.makedirs(config.results_dir, exist_ok=True)
model = YOLO(os.path.join(config.runs_dir, run_name, "weights", "best.pt"))

metrics = model.val(data=data_yaml, imgsz=config.tile_size, split="val", plots=True,
                    project=config.runs_dir, name=run_name + "_val", exist_ok=True)

rows = []
for i, class_index in enumerate(metrics.box.ap_class_index):
    p, r, ap50, ap = metrics.box.class_result(i)
    rows.append({"class": model.names[int(class_index)], "precision": p, "recall": r, "mAP50": ap50})
rows.append({"class": "ALL", "precision": metrics.box.mp, "recall": metrics.box.mr, "mAP50": metrics.box.map50})
table = pd.DataFrame(rows).round(3)
print("\n", table.to_string(index=False))
table.to_csv(os.path.join(config.results_dir, "07_metrics_" + run_name + ".csv"), index=False)

# Speed: time the model on single tiles, the way it would run on a stream of images
random.seed(config.seed)
tiles = sorted(glob.glob(os.path.join(os.path.dirname(data_yaml), "images", "val", "*.jpg")))
tiles = random.sample(tiles, min(50, len(tiles)))
times = []
for path in tiles:
    result = model.predict(path, imgsz=config.tile_size, verbose=False)[0]
    times.append(result.speed["inference"])      # milliseconds spent in the network
times = times[5:]                                # the first few runs are slower (warm-up), skip them
device = "GPU" if torch.cuda.is_available() else "CPU"
if len(times) > 0:
    print(f"\nSpeed on {device}: {sum(times) / len(times):.1f} ms per 640x640 tile")
else:
    print("\nNo validation tiles found next to the yaml file, so the speed test was skipped.")
