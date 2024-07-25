# Face Recognition System

This repository contains a Python-based face recognition system utilizing the `face_recognition` library. The system includes functionalities for training the model, evaluating its performance, and performing real-time face recognition with alert notifications.

## Scripts

### 1. `extract_embeddings.py`

Extracts face embeddings from training images and saves them to a file for later use.

**Usage:**

python extract_embeddings.py
Requirements:

Set the train_images_directory variable to the directory containing your training images.

### 2. recognize_video.py
Performs real-time face recognition using a webcam and plays an alert sound if a known face is detected.

Usage:

python recognize_video.py
Requirements:

Set the alert_sound_path variable to the path of your custom alert sound file.
Ensure the known_faces.pkl file is present in the same directory.

### 3. train_model.py
Evaluates the accuracy of the face recognition model using a set of test images.

Usage:

Copy code
python train_model.py
Requirements:

Set the test_images_directory variable to the directory containing your test images.
Ensure the known_faces.pkl file is present in the same directory.
Setup
Install Dependencies:

Install the required Python libraries:

Copy code
pip install opencv-python face_recognition scikit-learn pygame
Prepare Directories:

Create directories for training and test images.
* Place your training images in the directory specified by train_images_directory in extract_embeddings.py.
Place your test images in the directory specified by test_images_directory in train_model.py.
Ensure the alert sound file is in the correct path specified in recognize_video.py.
Run the Scripts:

Run extract_embeddings.py to generate the known_faces.pkl file.
Run train_model.py to evaluate the model's performance.
Run recognize_video.py for real-time face recognition and alerts.
Configuration
Training Directory: Set the train_images_directory variable in extract_embeddings.py.
Test Directory: Set the test_images_directory variable in train_model.py.
Alert Sound Path: Set the alert_sound_path variable in recognize_video.py.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
face_recognition for the face recognition library.
OpenCV for computer vision functionalities.
pygame for audio playback.
Feel free to modify the scripts and configurations to suit your needs. For any issues or contributions, please open an issue or pull request on this repository.
