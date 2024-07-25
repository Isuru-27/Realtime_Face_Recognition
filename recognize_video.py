# python recognize_video.py

import cv2
import face_recognition
import pickle
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load known faces from file
with open("known_faces.pkl", "rb") as file:
    known_faces = pickle.load(file)

# Open the webcam
video_capture = cv2.VideoCapture(0)

# Custom alert sound file path
alert_sound_path = "D:\\face\\alert_sound.mp3"

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to RGB for face recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find face locations in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    
    if face_locations:
        # Encode the faces in the frame
        frame_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face found in the frame
        for face_encoding, (top, right, bottom, left) in zip(frame_encodings, face_locations):
            # Compare the face encoding with known faces
            matches = face_recognition.compare_faces(known_faces["SP_001"], face_encoding, tolerance=0.5)
            name = "Unknown"

            # Check if any known face matches
            if True in matches:
                name = "SP_001"
                print("Face detected: SP_001")
                # Play custom alert sound using pygame
                try:
                    pygame.mixer.music.load(alert_sound_path)
                    pygame.mixer.music.play()
                except Exception as e:
                    print(f"Error playing sound: {e}")

            # Display the name as alert
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Video", frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
