import streamlit as st
import cv2
import numpy as np
import time
from collections import Counter
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(initial_sidebar_state="collapsed")
st.markdown("<h1 style='text-align: center;font-size:30px; color: red;'>Muzzle Detection With Camera</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;font-size:15px; color: green;'>loading may take sometime</h1>", unsafe_allow_html=True)

model = YOLO('assets/model2.tflite')
detected_classes = []
start_time = None
detection_duration = 15  # duration for detection in seconds

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_predictions = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        results = model(img, task='detect', show=False)

        self.last_predictions = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = box.cls[0]
                class_name = model.names[int(cls)]

                self.last_predictions.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': conf, 'class': class_name
                })

        for prediction in self.last_predictions:
            x1, y1, x2, y2 = int(prediction['x1']), int(prediction['y1']), int(prediction['x2']), int(prediction['y2'])
            confidence = prediction['confidence']
            class_name = prediction['class']

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        global detected_classes, start_time
        if start_time is None:
            start_time = time.time()

        for prediction in self.last_predictions:
            detected_classes.append(prediction['class'])

        if time.time() - start_time > detection_duration:
            return None

        return img

def main():
    global detected_classes, start_time

    with st.container(height=730):
        ctx = webrtc_streamer(
            key="example",
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
        )

        if ctx.video_transformer:
            if st.button("Stop Camera"):
                ctx.video_transformer.stop()

    if not ctx.state.playing and detected_classes:
        most_common_class = Counter(detected_classes).most_common(1)[0][0]
        st.write(f"The most occurred class: {most_common_class}")
        st.switch_page(f"pages/{most_common_class}.py")

    _, back, _ = st.columns(3, vertical_alignment="bottom")
    back = st.button("Back Home", use_container_width=True)
    if back:
        st.switch_page("main.py")

    # Reset global variables
    detected_classes = []
    start_time = None

if __name__ == '__main__':
    main()
