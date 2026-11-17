# Settings used by all the steps. If you want to change something, change it here.
import os

# the folder this file is in (the project folder)
here = os.path.dirname(os.path.abspath(__file__))

# On Colab the fast disk is /content. The big data goes there because Google Drive is slow.
on_colab = os.path.exists("/content")
if on_colab:
    data_root = "/content/data"
else:
    data_root = os.path.join(here, "data")

zip_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip"
raw_dir = os.path.join(data_root, "DOTAv1")    # the download after unzipping
tile_dir = os.path.join(data_root, "tiles")    # small pictures cut from the big ones

# pictures, tables and trained models stay in the project folder
results_dir = os.path.join(here, "results")
runs_dir = os.path.join(here, "runs")

# DOTA has 15 classes, numbered 0 to 14. I only use three of them.
dota_names = {
    0: "plane", 1: "ship", 2: "storage tank", 3: "baseball diamond",
    4: "tennis court", 5: "basketball court", 6: "ground track field",
    7: "harbor", 8: "bridge", 9: "large vehicle", 10: "small vehicle",
    11: "helicopter", 12: "roundabout", 13: "soccer ball field", 14: "swimming pool",
}
my_class_ids = [0, 1, 9]
my_names = [dota_names[i] for i in my_class_ids]

# how the big images are cut up (in pixels)
tile_size = 640
tile_step = 512            # smaller than tile_size, so neighbouring tiles overlap by 128 px
min_inside = 0.6           # keep an object in a tile only if 60% of it is inside the tile
empty_tile_share = 0.1     # keep 1 in 10 tiles that have no objects, as background examples
seed = 42

# None means use everything. Put a small number (like 40) for a quick test run.
max_train_images = None  
max_val_images = None

# training
epochs = 30
#epochs = 3
batch = 16

# detections below this confidence are thrown away
conf = 0.25
