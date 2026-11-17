# Aerial object detection: planes, ships and large vehicles

Personal project. I trained a small YOLO model to find planes, ships and large vehicles in aerial photos, then built a web page where you upload a photo, change what is shown, and see the detections, with a made-up lat/lon for each one, and zoom into any part of the photo.

## Data

DOTA v1.0, a public set of aerial photos with hand-drawn boxes (Wuhan University). I use 3 of its 15 classes: plane, ship, large vehicle. I did not label anything myself. The copy I download is the one hosted by Ultralytics: 1,411 training photos and 458 validation photos. DOTA is released for academic use, see its page for the terms.

## How it works

The photos are up to 4,000 pixels wide, so step 3 cuts them into 640 x 640 tiles with some overlap and rewrites the labels to match. The model is YOLO11n (the smallest YOLO11), fine-tuned from its COCO weights. For a full photo, the tiles go through the model one batch at a time and duplicate boxes from the overlaps are removed.

| File | What it does |
|---|---|
| `config.py` | all settings in one place |
| `tools.py` | helper functions (labels, tiling, big-photo detection, pixel to lat/lon) |
| `01_download_data.py` | downloads and unzips DOTA |
| `02_look_at_data.py` | counts objects, measures their size in pixels, saves two pictures |
| `03_make_tiles.py` | cuts photos into tiles, converts labels to YOLO format |
| `04_write_yaml_and_check.py` | writes the data file, checks the split, draws labels on sample tiles |
| `05_augmentation_preview.py` | shows flips and brightness changes with the boxes following |
| `06_train.py` | fine-tunes the model |
| `07_evaluate.py` | precision, recall, mAP50 per class, speed per tile |
| `08_recall_by_size.py` | how often objects of each size are found |
| `09_predict_pictures.py` | prediction pictures, tiles and one full photo |
| `10_pixel_to_latlon.py` | pixel position to latitude and longitude |
| `11_app.py` | Streamlit web page |

Steps 1 to 10 were run on Google Colab. The web page (step 11) runs locally, because it opens in a browser on your own computer. Copy `runs/optical_v1/weights/best.pt` to `model/best.pt` first.

```
# On Colab, steps 1 to 10
pip install -r requirements.txt
python 01_download_data.py
... (steps 2 to 10 in order)

# On a laptop, step 11
pip install -r requirements.txt
streamlit run 11_app.py        # needs model/best.pt
```

## Results

Validation tiles, confidence 0.25.

| Class | Precision | Recall | mAP50 |
|---|---|---|---|
| plane | 0.929 | 0.912 | 0.950 |
| ship | 0.924 | 0.928 | 0.966 |
| large vehicle | 0.889 | 0.820 | 0.895 |
| all | 0.914 | 0.887 | 0.937 |

Speed: 8.1 ms per 640 x 640 tile on a Colab GPU (from step 7).

### Recall by object size

A random sample of 300 validation tiles, confidence 0.25. An object counts as found if the model draws a box of the same class that overlaps it with IoU of at least 0.5.

| Class | Under 16 px | 16-32 px | 32-64 px | 64 px and more |
|---|---|---|---|---|
| plane | 0.00 (5 objects) | 0.42 (43) | 0.98 (128) | 0.98 (242) |
| ship | 1.00 (4) | 0.91 (291) | 0.97 (2,053) | 0.96 (297) |
| large vehicle | 0.00 (2) | 0.17 (35) | 0.87 (485) | 0.91 (556) |

Objects over 32 px are found most of the time. Recall drops for small planes and small large vehicles, and I do not have a tested reason for it. The under-16 px groups hold only 2 to 5 objects, so those numbers say little. The chart is in `results/08_recall_by_size.png`.

## Web page

`streamlit run 11_app.py`. Upload an aerial photo, set the minimum confidence, pick which classes to show and which one is listed first, and zoom into any part of the photo at full resolution. Screenshot with a DOTA validation photo at confidence 0.50 (103 objects found), in two parts:

![Web page, top: settings, upload and detections](results/streamlit_app_1.png)

![Web page, bottom: zoom and detection table](results/streamlit_app_2.png)

The lat/lon columns come from a made-up top-left corner (0, 0) and 0.5 m per pixel, because DOTA photos have no coordinates.

## Limits

- The DOTA photos have no coordinates. The latitude and longitude in step 10 and in the web page come from a made-up top-left corner and pixel size that you can change. With a georeferenced image these would be read from the file.
- I chose the best model by its score on the validation photos and have no separate test set, so the scores are a little optimistic.
- DOTA marks objects with tilted boxes. I turn them into straight boxes, so a tilted ship gets some extra background in its box.
- Only normal (optical) photos are used here. No infrared, hyperspectral or segmentation.
- The photos come from different sources and have different pixel sizes, so object sizes in metres in the web page depend on the pixel size you type in.

## Credits

DOTA dataset (Xia et al., Wuhan University). YOLO11 and the training library by Ultralytics (AGPL-3.0). The sample photo in test_for_streamlit_image/ is from the DOTA validation set, used for demonstration.
