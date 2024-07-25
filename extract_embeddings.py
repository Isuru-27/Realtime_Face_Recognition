# python extract_embeddings.py

import os
import cv2
import face_recognition
import pickle

# Training set directory
train_images_directory = "D:\\face\\train"

# Dictionary to store known faces
known_faces = {
    "SP_001": []
}

# Load all images from the training directory
for image_file in os.listdir(train_images_directory):
    image_path = os.path.join(train_images_directory, image_file)
    if os.path.isfile(image_path):
        image = face_recognition.load_image_file(image_path)
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame)
        if len(face_encodings) > 0:
            for encoding in face_encodings:
                known_faces["SP_001"].append(encoding)
    else:
        print(f"Training image file not found: {image_path}")

# Save the known faces to a file
with open("known_faces.pkl", "wb") as file:
    pickle.dump(known_faces, file)

print("Face embeddings extracted and saved successfully.")

