import streamlit as st
import cv2
import time
from ultralytics import YOLO
import face_recognition
import numpy as np
import os
import sqlite3

# Load model YOLO
model = YOLO("yolov8m.pt")

# Inisialisasi database SQLite
def init_db():
    conn = sqlite3.connect("absensi.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS absensi (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nama TEXT,
        nim TEXT,
        mata_kuliah TEXT,
        jam_kuliah TEXT,
        waktu DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

init_db()

# Fungsi untuk menghapus data absensi lama
def clear_old_absensi():
    try:
        conn = sqlite3.connect("absensi.db")
        cursor = conn.cursor()
        
        # Hapus data absensi yang lebih lama dari 1 hari
        cursor.execute("DELETE FROM absensi WHERE waktu < datetime('now', '-1 day')")
        
        conn.commit()
        conn.close()
        print("Data absensi lama berhasil dihapus!")  # Debugging
    except sqlite3.Error as e:
        print(f"Database error: {e}")  # Debugging

# Fungsi untuk menyimpan absensi ke database
def insert_absensi(nama, nim, mata_kuliah, jam_kuliah):
    try:
        conn = sqlite3.connect("absensi.db")
        cursor = conn.cursor()
        
        # Cek apakah sudah ada absensi untuk mahasiswa ini pada mata kuliah dan jam yang sama
        cursor.execute("""
        SELECT * FROM absensi 
        WHERE nama = ? AND nim = ? AND mata_kuliah = ? AND jam_kuliah = ?
        """, (nama, nim, mata_kuliah, jam_kuliah))
        
        if cursor.fetchone() is None:
            # Jika tidak ada, simpan absensi baru
            cursor.execute("""
            INSERT INTO absensi (nama, nim, mata_kuliah, jam_kuliah) 
            VALUES (?, ?, ?, ?)
            """, (nama, nim, mata_kuliah, jam_kuliah))
            conn.commit()
            print(f"Data inserted: {nama}, {nim}, {mata_kuliah}, {jam_kuliah}")  # Debugging
        else:
            print(f"Absensi sudah ada: {nama}, {nim}, {mata_kuliah}, {jam_kuliah}")  # Debugging
        
        conn.close()
    except sqlite3.Error as e:
        print(f"Database error: {e}")  # Debugging

# Load wajah yang dikenali
known_face_encodings = []
known_face_names = []

def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    folder_path = "ktm_images"
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(os.path.splitext(filename)[0])
                print(f"Loaded: {filename}")  # Debugging
            else:
                print(f"No face found in: {filename}")  # Debugging

load_known_faces()

# Fungsi deteksi wajah dan penyimpanan absensi
def detect_people_and_faces(ip_address, mata_kuliah, jam_kuliah):
    cap = cv2.VideoCapture(0 if ip_address == "0" else ip_address)
    st_frame = st.empty()
    detected_students = set()

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
            nim = ""

            if True in matches:
                match_index = matches.index(True)
                full_name = known_face_names[match_index]
                print(f"Detected: {full_name}")  # Debugging
                try:
                    name, nim = full_name.split("_")
                    print(f"Name: {name}, NIM: {nim}")  # Debugging
                except ValueError:
                    name = full_name
                    print(f"Name: {name}, NIM: Not available")  # Debugging
                
                # Simpan ke database hanya jika belum terdeteksi dalam sesi ini
                if full_name not in detected_students:
                    insert_absensi(name, nim, mata_kuliah, jam_kuliah)
                    detected_students.add(full_name)
                    print(f"Inserted into database: {name}, {nim}, {mata_kuliah}, {jam_kuliah}")  # Debugging

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        for box in bboxes[idx]:
            (x, y, x2, y2) = box
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f'People Count: {count_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        st_frame.image(frame, channels="BGR")
        time.sleep(0.1)
        
        if st.session_state.get("stop", False):
            cap.release()
            break

# UI Streamlit
st.title("Sistem Monitoring Kelas Otomatis dengan Deteksi dan Pengenalan Wajah")
ip_address = st.sidebar.text_input("Enter IP Address for Video Feed", "0")
mata_kuliah = st.sidebar.text_input("Mata Kuliah", "Pemrograman")
jam_kuliah = st.sidebar.text_input("Jam Kuliah", "08:00-10:00")

if st.sidebar.button("Start Monitoring", key="start_button"):
    st.session_state["stop"] = False
    clear_old_absensi()  # Hapus data lama sebelum memulai
    detect_people_and_faces(ip_address, mata_kuliah, jam_kuliah)

if st.sidebar.button("Stop Monitoring", key="stop_button"):
    st.session_state["stop"] = True
    st.success("Monitoring Stopped!")

# Menampilkan Data Absensi
if st.sidebar.button("Lihat Absensi"):
    conn = sqlite3.connect("absensi.db")
    cursor = conn.cursor()
    
    # Tampilkan data absensi untuk mata kuliah dan jam kuliah yang sedang dipantau
    cursor.execute("""
    SELECT * FROM absensi 
    WHERE mata_kuliah = ? AND jam_kuliah = ?
    ORDER BY waktu DESC
    """, (mata_kuliah, jam_kuliah))
    
    data = cursor.fetchall()
    conn.close()

    if data:
        st.write("### Data Absensi")
        st.table(data)
    else:
        st.warning("Belum ada data absensi untuk mata kuliah dan jam ini.")