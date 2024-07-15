import streamlit as st
import cv2
import numpy as np
import time
from collections import Counter
from ultralytics import YOLO
st.set_page_config(initial_sidebar_state="collapsed")
st.markdown("<h1 style='text-align: center;font-size:30px; color: red;'>Muzzle Detection With Camera</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;font-size:15px; color: green;'>loading may take sometime</h1>", unsafe_allow_html=True)
with st.container(height=730):
    model = YOLO('assets/model2.tflite')
    last_predictions = []

    def detect_objects(frame):
        global last_predictions

        results = model(frame, task='detect', show=False)

        last_prediction = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = box.cls[0]
                class_name = model.names[int(cls)]

                last_predictions.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': conf, 'class': class_name
                })

        for prediction in last_predictions:
            x1, y1, x2, y2 = int(prediction['x1']), int(prediction['y1']), int(prediction['x2']), int(prediction['y2'])
            confidence = prediction['confidence']
            class_name = prediction['class']

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 12)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)

        return frame

    def main():
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            st.error('Error: Could not open video device')
            return

        start_button, stop_button = st.columns(2, vertical_alignment="top")
        start_button = st.button("Start Camera and Wait for 15 sec", use_container_width=True)
        stop_button = st.button("Stop Camera", use_container_width=True)

        camera_started = False

        if start_button:
            camera_started = True

        if stop_button:
            camera_started = False

        frame_placeholder = st.empty()
        detected_classes = []

        start_time = time.time()
        detection_duration = 15  # duration for detection in seconds

        while camera_started:
            ret, frame = video_capture.read()

            if not ret:
                st.warning('No frame is available, check camera connection')
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_detected = detect_objects(frame_rgb)

            for prediction in last_predictions:
                detected_classes.append(prediction['class'])

            frame_placeholder.image(frame_detected, channels='RGB', use_column_width=True)

            if time.time() - start_time > detection_duration:
                camera_started = False

            time.sleep(0.1)
            last_predictions.clear()
            st.cache_data.clear()

        video_capture.release()
        cv2.destroyAllWindows()

        if detected_classes:
            most_common_class = Counter(detected_classes).most_common(1)[0][0]
            st.write(f"The most occurred class: {most_common_class}")
            st.switch_page(f"pages/{most_common_class}.py")

        _, back, _ = st.columns(3, vertical_alignment="bottom")
        back = st.button("Back Home", use_container_width=True)
        if back:
            st.switch_page("main.py")

    if __name__ == '__main__':
        main()
