import face_recognition
import os
import pickle

dataset_folder = "dataset/"
encodings_file = "encodings.pkl"

known_encodings = []
known_names = []

for student_folder in os.listdir(dataset_folder):
    student_path = os.path.join(dataset_folder, student_folder)
    if not os.path.isdir(student_path):
        continue

    for image_name in os.listdir(student_path):
        image_path = os.path.join(student_path, image_name)
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        
        if face_encodings:
            known_encodings.append(face_encodings[0])
            known_names.append(student_folder)

data = {"encodings": known_encodings, "names": known_names}

with open(encodings_file, "wb") as f:
    pickle.dump(data, f)

print("Encodings berhasil dibuat!")
