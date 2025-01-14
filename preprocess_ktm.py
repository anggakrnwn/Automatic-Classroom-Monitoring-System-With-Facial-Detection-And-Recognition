import cv2
import os

input_folder = "ktm_images/"
output_folder = "dataset/"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Gambar {image_name} tidak dapat dibaca.")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for i, (x, y, w, h) in enumerate(faces):
        face = image[y:y+h, x:x+w]
        
        # Ekstraksi nama dan NIM dari nama file
        try:
            name, nim = image_name.split("_")[0], image_name.split("_")[1].split(".")[0]
        except IndexError:
            print(f"Format nama file salah: {image_name}")
            continue

        student_folder = os.path.join(output_folder, f"{name}_{nim}")
        os.makedirs(student_folder, exist_ok=True)
        
        output_path = os.path.join(student_folder, f"face_{i}.jpg")
        cv2.imwrite(output_path, face)

print("Proses cropping selesai!")
