# python train_model.py

import os
import cv2
import face_recognition
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Function to evaluate accuracy
def evaluate_accuracy(known_faces, test_images_directory, true_labels):
    predicted_labels = []

    for image_file in os.listdir(test_images_directory):
        image_path = os.path.join(test_images_directory, image_file)
        if os.path.isfile(image_path):
            image = face_recognition.load_image_file(image_path)
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            frame_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if frame_encodings:
                for face_encoding in frame_encodings:
                    matches = face_recognition.compare_faces(known_faces["SP_001"], face_encoding, tolerance=0.5)
                    name = "Unknown"
                    if True in matches:
                        name = "SP_001"
                    predicted_labels.append(name)
            else:
                print(f"No face found in test image: {image_path}")
                predicted_labels.append("Unknown")
        else:
            print(f"Test image file not found: {image_path}")
            predicted_labels.append("Unknown")

    # Metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, pos_label="SP_001", zero_division=0)
    recall = recall_score(true_labels, predicted_labels, pos_label="SP_001", zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, pos_label="SP_001", zero_division=0)
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=["SP_001", "Unknown"])

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix: \n{conf_matrix}")

# Load known faces from file
with open("known_faces.pkl", "rb") as file:
    known_faces = pickle.load(file)

# Test set directory
test_images_directory = "D:\\face\\test"
true_labels = ["SP_001"] * len(os.listdir(test_images_directory))  # Assuming all test images are of SP_001

# Evaluate accuracy
evaluate_accuracy(known_faces, test_images_directory, true_labels)
