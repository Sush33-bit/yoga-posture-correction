import cv2
import mediapipe as mp
import numpy as np
import os
import joblib

# Load the trained model
model = joblib.load(r"C:\Users\Sushant Kabra\Desktop\NEW PROJECT\IMAGE RELATED\yoga_pose_model.pkl")

# Mediapipe pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# Function to extract pose keypoints
def extract_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        keypoints = []

        for lm in mp_pose.PoseLandmark:
            keypoints.append(landmarks[lm.value].x)
            keypoints.append(landmarks[lm.value].y)

        return keypoints
    else:
        return None

# Function to test a single image
def test_image(image_path):
    image = cv2.imread(image_path)
    keypoints = extract_keypoints(image)

    if keypoints:
        prediction = model.predict([keypoints])[0]
        print(f"Predicted Pose for {os.path.basename(image_path)}: {prediction}")
        cv2.putText(image, f'{prediction}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow("Prediction", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"No pose detected in {os.path.basename(image_path)}")

# Test on all images in a folder
def test_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(folder_path, filename)
            test_image(full_path)

# ========== SET THIS ==========

# For a single image:
# test_image(r"C:\Users\Sushant Kabra\Desktop\NEW PROJECT\TEST IMAGES\test1.jpg")

# OR, test all images in a folder:
test_images_in_folder(r"C:\Users\Sushant Kabra\Desktop\NEW PROJECT\DATASET\TEST")
