import cv2
import os

# Folder input dan output
input_folder = "ktm_images/"
output_folder = "dataset/"

# Load Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Buat folder output jika belum ada
os.makedirs(output_folder, exist_ok=True)

# Loop melalui semua gambar di folder input
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(image_path)

    # Jika gambar tidak dapat dibaca, lanjut ke gambar berikutnya
    if image is None:
        print(f"Gambar {image_name} tidak dapat dibaca.")
        continue

    # Konversi gambar ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Jika wajah terdeteksi, simpan wajah ke folder dataset
    for i, (x, y, w, h) in enumerate(faces):
        face = image[y:y+h, x:x+w]
        
        # Ekstraksi nama dan NIM dari nama file
        try:
            name, nim = image_name.split("_")[0], image_name.split("_")[1].split(".")[0]
        except IndexError:
            print(f"Format nama file salah: {image_name}")
            continue

        # Buat folder untuk setiap mahasiswa
        student_folder = os.path.join(output_folder, f"{name}_{nim}")
        os.makedirs(student_folder, exist_ok=True)
        
        # Simpan wajah yang terdeteksi
        output_path = os.path.join(student_folder, f"face_{i}.jpg")
        cv2.imwrite(output_path, face)

print("Proses cropping selesai!")