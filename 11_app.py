# Step 11: a small web page where you upload an image and see the detections.
# Run it on your own computer (not on Colab):
#   streamlit run 11_app.py
# It needs the trained model at model/best.pt (copy it from runs/optical_v1/weights/best.pt).
#
# The person using it can change what is shown: how sure the model must be, which classes,
# and which class comes first in the list. That is the "operator input".
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import config
import tools

st.title("Aerial object detector")
st.caption("Personal project. Trained on the public DOTA aerial image set (planes, ships, large vehicles). "
           "The lat/lon columns use a made-up georeference (DOTA photos have none) that you can change on the left.")


@st.cache_resource
def load_model():
    return YOLO(os.path.join(config.here, "model", "best.pt"))


model = load_model()

# ---- settings on the left ----
st.sidebar.header("What to show")
conf = st.sidebar.slider("Minimum confidence", 0.05, 0.95, 0.25, 0.05)
shown_classes = st.sidebar.multiselect("Classes", config.my_names, default=config.my_names)
first_class = st.sidebar.selectbox("List this class first", ["none"] + config.my_names)

st.sidebar.header("Made-up georeference")
lat0 = st.sidebar.number_input("Top-left latitude", value=0.0, format="%.4f")
lon0 = st.sidebar.number_input("Top-left longitude", value=0.0, format="%.4f")
gsd = st.sidebar.number_input("Metres per pixel", value=0.5, min_value=0.05, step=0.05)

# ---- the image ----
uploaded = st.file_uploader("Upload an aerial image (jpg or png)", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload an image to start.")
    st.stop()

image = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
st.write("Image size:", image.shape[1], "x", image.shape[0], "pixels")

with st.spinner("Looking for objects..."):
    detections = tools.detect_big_image(model, image, config.tile_size, config.tile_step, conf)

# keep only the classes that were ticked
detections = [d for d in detections if config.my_names[d[5]] in shown_classes]

rows = []
for x1, y1, x2, y2, score, class_id in detections:
    lat, lon = tools.pixel_to_latlon((x1 + x2) / 2, (y1 + y2) / 2, lat0, lon0, gsd)
    rows.append({
        "class": config.my_names[class_id],
        "confidence": round(score, 2),
        "size (m)": round(max(x2 - x1, y2 - y1) * gsd, 1),
        "lat": lat,
        "lon": lon,
    })

if len(rows) == 0:
    st.warning("Nothing found with these settings.")
    st.stop()

table = pd.DataFrame(rows)
# the chosen class goes first, inside each group the most confident one is on top
table["first"] = table["class"] != first_class
table = table.sort_values(["first", "confidence"], ascending=[True, False]).drop(columns="first")
table.insert(0, "rank", range(1, len(table) + 1))

drawn = tools.draw_detections(image, [d for d in detections], config.my_names)
drawn_rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
st.image(drawn_rgb, caption=str(len(table)) + " objects found (hover over the image and click the arrows for full screen)")

# ---- zoom: show one part of the photo at full resolution ----
st.subheader("Zoom")
height, width = drawn_rgb.shape[:2]
window = st.select_slider("Window size (pixels)", options=[256, 512, 1024, 2048], value=512)
window = min(window, width, height)
left = st.slider("Left edge (x)", 0, max(width - window, 0), 0, step=32)
top = st.slider("Top edge (y)", 0, max(height - window, 0), 0, step=32)
st.image(drawn_rgb[top:top + window, left:left + window], width="stretch")

st.subheader("Detections")
st.dataframe(table, hide_index=True)
