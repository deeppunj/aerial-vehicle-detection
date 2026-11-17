# Step 6: train the detector.
# I do not train from zero. yolo11n.pt is a small YOLO model that already learned to see
# edges, shapes and everyday objects on a large photo set (COCO). I start from that and
# keep training on my aerial tiles. This is called fine-tuning.
#
# Run it like this:
#   python 06_train.py                          -> trains on the aerial tiles, run name optical_v1
#   python 06_train.py path/to/data.yaml name   -> trains on another data set (for example radar images)
import os
import sys
from ultralytics import YOLO
import config

data_yaml = os.path.join(config.tile_dir, "data.yaml")
run_name = "optical_v1"
if len(sys.argv) > 1:
    data_yaml = sys.argv[1]
if len(sys.argv) > 2:
    run_name = sys.argv[2]

model = YOLO("yolo11n.pt")      # downloads the starting weights the first time

model.train(
    data=data_yaml,
    epochs=config.epochs,       # one epoch = the model sees every training tile once
    imgsz=config.tile_size,
    batch=config.batch,
    flipud=0.5,                 # from above there is no "up", so flipping top-bottom is fine too
    fliplr=0.5,
    patience=10,                # stop early if the validation score has not improved for 10 epochs
    seed=config.seed,
    workers=2,
    project=config.runs_dir,
    name=run_name,
    exist_ok=True,              # train again with the same name -> overwrite the old run
)

print("Done. The best model is saved at:")
print(os.path.join(config.runs_dir, run_name, "weights", "best.pt"))
