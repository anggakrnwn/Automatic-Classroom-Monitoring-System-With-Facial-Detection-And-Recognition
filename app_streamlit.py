import streamlit as st
import cv2
import time
from ultralytics import YOLO
import face_recognition
import numpy as np
import os

model = YOLO("yolov8m.pt")
known_face_encodings = []
known_face_names = []

def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    folder_path = "ktm_images"
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(os.path.splitext(filename)[0])

load_known_faces()

def detect_people_and_faces(ip_address):
    cap = cv2.VideoCapture(0 if ip_address == "0" else ip_address)
    st_frame = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        result = results[0]
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        idx = [i for i in range(len(classes)) if classes[i] == 0]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        count_people = len(idx)
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        for box in bboxes[idx]:
            (x, y, x2, y2) = box
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f'People Count: {count_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        st_frame.image(frame, channels="BGR")
        time.sleep(0.1)  # Mengontrol frame rate

        if st.session_state.get("stop", False):
            cap.release()
            break

# UI Streamlit
st.title("Sistem Monitoring Kelas Otomatis dengan Deteksi dan Pengenalan Wajah")
ip_address = st.sidebar.text_input("Enter IP Address for Video Feed", "0")

if st.sidebar.button("Start Monitoring", key="start_button"):
    st.session_state["stop"] = False
    detect_people_and_faces(ip_address)

if st.sidebar.button("Stop Monitoring", key="stop_button"):
    st.session_state["stop"] = True
    st.success("Monitoring Stopped!")
