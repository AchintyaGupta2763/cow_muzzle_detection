import streamlit as st
import cv2
import numpy as np
import time
from collections import Counter
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image

st.set_page_config(initial_sidebar_state="collapsed")
st.markdown("<h1 style='text-align: center;font-size:30px; color: red;'>Muzzle Detection With Camera</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;font-size:15px; color: green;'>Loading may take some time</h1>", unsafe_allow_html=True)

class YOLOTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.model = YOLO('assets/model2.tflite')

    def transform(self, frame):
        image = Image.fromarray(frame)
        frame_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        frame_detected = self.detect_objects(frame_rgb)
        return frame_detected

    def detect_objects(self, frame):
        results = self.model(frame, task='detect', show=False)
        detected_classes = []

        for r in results.pred:
            for box in r:
                x1, y1, x2, y2 = [int(coord) for coord in box[:4]]
                conf = float(box[4])
                cls = int(box[5])
                class_name = self.model.names[cls]

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f'{class_name}: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                detected_classes.append(class_name)

        return frame

def main():
    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=YOLOTransformer)

    if not webrtc_ctx:
        st.error("Error: Webcam not found or could not be accessed.")
        return

    detected_classes = []
    start_time = time.time()
    detection_duration = 15  # duration for detection in seconds

    while webrtc_ctx.video_receiver.running:
        if time.time() - start_time > detection_duration:
            break

        time.sleep(0.1)

    if detected_classes:
        most_common_class = Counter(detected_classes).most_common(1)[0][0]
        st.write(f"The most occurred class: {most_common_class}")
        st.write(f"Navigating to page: pages/{most_common_class}.py")

    _, back, _ = st.columns(3, vertical_alignment="bottom")
    back_button = back.button("Back Home", key="back_button")
    if back_button:
        st.write("Navigating back to main page: main.py")

if __name__ == '__main__':
    main()
