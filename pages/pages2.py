import cv2
import numpy as np
import streamlit as st
from camera_input_live import camera_input_live
from ultralytics import YOLO

# Initialize the model once
model = YOLO('assets/model2.tflite')

def detect_objects(frame):
    results = model(frame, task='detect', show=False)

    last_predictions = []  # Clear previous predictions
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
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, last_predictions

# Streamlit camera input
image = st.camera_input("")

if image:
    bytes_data = image.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    frame_detected, predictions = detect_objects(frame)
    st.image(frame_detected)

    # Display detected classes and confidence scores
    for prediction in predictions:
        class_name = prediction['class']
        confidence = prediction['confidence']
        st.text(f"Class: {class_name}, Confidence: {confidence:.2f}")

        if st.button(f"GO TO {class_name}", use_container_width=True):
            st.switch_page(f"pages/{class_name}.py")

back = st.button("Back Home", use_container_width=True)
if back:
    st.switch_page("main.py")
