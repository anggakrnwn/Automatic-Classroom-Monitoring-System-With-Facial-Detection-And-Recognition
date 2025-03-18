import face_recognition
import os
import pickle

# Folder dataset
dataset_folder = "dataset/"

# List untuk menyimpan encoding dan nama wajah
known_face_encodings = []
known_face_names = []

# Loop melalui semua folder di dataset
for student_folder in os.listdir(dataset_folder):
    student_path = os.path.join(dataset_folder, student_folder)
    
    # Pastikan ini adalah folder
    if not os.path.isdir(student_path):
        continue

    # Ekstraksi nama dan NIM dari nama folder
    name, nim = student_folder.split("_")

    # Loop melalui semua gambar wajah di folder mahasiswa
    for face_image_name in os.listdir(student_path):
        face_image_path = os.path.join(student_path, face_image_name)
        
        # Load gambar wajah
        image = face_recognition.load_image_file(face_image_path)
        
        # Buat encoding wajah
        encodings = face_recognition.face_encodings(image)
        
        # Jika encoding berhasil dibuat, simpan ke list
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(f"{name}_{nim}")

# Simpan encoding dan nama ke file pickle
with open("encodings.pkl", "wb") as f:
    pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)

print("Encoding wajah selesai dan disimpan ke encodings.pkl!")