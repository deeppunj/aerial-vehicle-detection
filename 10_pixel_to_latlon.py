# Step 10: turn the pixel position of each detection into latitude and longitude.
#
# A satellite or drone image that is georeferenced knows two things:
#   - the earth coordinates of its top-left corner
#   - how many metres one pixel covers on the ground (the "ground sampling distance", GSD)
# With those, a pixel position becomes a map position (see pixel_to_latlon in tools.py).
#
# IMPORTANT: the DOTA photos have no georeference. The corner and GSD below are MADE-UP values
# to show the maths. With a real georeferenced image (a GeoTIFF) they would be read from the file.
import os
import sys
import cv2
import pandas as pd
from ultralytics import YOLO
import config
import tools

run_name = "optical_v1"
if len(sys.argv) > 1:
    run_name = sys.argv[1]

top_left_lat = 0.0          # made-up values: DOTA photos have no real location
top_left_lon = 0.0
metres_per_pixel = 0.5

# First check the maths: two points 1000 pixels apart in a straight line should be 500 m apart.
lat_a, lon_a = tools.pixel_to_latlon(0, 0, top_left_lat, top_left_lon, metres_per_pixel)
lat_b, lon_b = tools.pixel_to_latlon(1000, 0, top_left_lat, top_left_lon, metres_per_pixel)
lat_c, lon_c = tools.pixel_to_latlon(0, 1000, top_left_lat, top_left_lon, metres_per_pixel)
print("1000 px to the east :", round(tools.metres_between(lat_a, lon_a, lat_b, lon_b), 1), "m (should be 500)")
print("1000 px to the south:", round(tools.metres_between(lat_a, lon_a, lat_c, lon_c), 1), "m (should be 500)")

# Now detect on one big photo and convert every box centre
model = YOLO(os.path.join(config.runs_dir, run_name, "weights", "best.pt"))
path, count = tools.find_demo_image(config.raw_dir, "val", config.my_class_ids)
image = cv2.imread(path)
detections = tools.detect_big_image(model, image, config.tile_size, config.tile_step, config.conf)

rows = []
for x1, y1, x2, y2, score, class_id in detections:
    centre_x = (x1 + x2) / 2
    centre_y = (y1 + y2) / 2
    lat, lon = tools.pixel_to_latlon(centre_x, centre_y, top_left_lat, top_left_lon, metres_per_pixel)
    rows.append({
        "class": config.my_names[class_id],
        "confidence": round(score, 2),
        "size_m": round(max(x2 - x1, y2 - y1) * metres_per_pixel, 1),
        "lat": round(lat, 6),
        "lon": round(lon, 6),
    })
table = pd.DataFrame(rows).sort_values("confidence", ascending=False)
print(table.head(10).to_string(index=False))
os.makedirs(config.results_dir, exist_ok=True)
table.to_csv(os.path.join(config.results_dir, "10_detections_with_coordinates.csv"), index=False)
print("Saved", len(table), "detections in results/10_detections_with_coordinates.csv")
