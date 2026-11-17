# Step 1: download the DOTA aerial image set (about 2 GB) and unzip it.
# DOTA is a public set of aerial photos with every object marked by hand.
# The copy used here is the one hosted by Ultralytics, the makers of the YOLO library.
import os
import glob
import zipfile
import urllib.request
import config

os.makedirs(config.data_root, exist_ok=True)
zip_path = os.path.join(config.data_root, "DOTAv1.zip")
done_marker = os.path.join(config.raw_dir, "unzip_finished.txt")

if os.path.exists(done_marker):
    print("Already downloaded and unzipped, nothing to do.")
else:
    if not os.path.exists(zip_path):
        print("Downloading about 2 GB, this takes a few minutes...")
        urllib.request.urlretrieve(config.zip_url, zip_path)
    print("Unzipping...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(config.data_root)
    open(done_marker, "w").write("ok")
    os.remove(zip_path)       # the zip is not needed any more

for split in ["train", "val", "test"]:
    images = glob.glob(os.path.join(config.raw_dir, "images", split, "*"))
    print(split, "images:", len(images))
print("Data is in", config.raw_dir)
