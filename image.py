import os
import cv2
import numpy as np
import mediapipe as mp
import joblib
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Function to calculate angle between 3 points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# Feature extraction using 4 joint angles
def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return None
    landmarks = results.pose_landmarks.landmark
    h, w = image.shape[:2]
    angles = []

    # Elbow and knee angles
    joint_sets = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
    ]
    
    for a, b, c in joint_sets:
        angles.append(calculate_angle(
            [landmarks[a.value].x * w, landmarks[a.value].y * h],
            [landmarks[b.value].x * w, landmarks[b.value].y * h],
            [landmarks[c.value].x * w, landmarks[c.value].y * h]
        ))
    return angles

# Function to test a single image
def test_single_image(image_path):
    # Load the model
    model = joblib.load(r"C:\Users\Sushant Kabra\Desktop\NEW PROJECT\IMAGE RELATED\yoga_pose_model.pkl")

    # Extract features
    features = extract_features(image_path)
    if not features:
        print("Pose landmarks not detected or feature extraction failed.")
        return

    # Predict the pose
    pose_name = model.predict([features])[0]
    print(f"Predicted Pose: {pose_name}")

    # Get the ideal angles for the pose (you can manually define them for each pose class)
    ideal_angles = {
        "TreePose": [45, 60, 35, 60],  # Example ideal angles, replace with actual ones
        "DownwardDogPose": [30, 55, 45, 60],
        "WarriorPose": [60, 70, 40, 65],
        # Add other poses and their ideal angles
    }

    # Show ideal angles vs predicted angles
    print("Calculated Angles from Image:")
    for i, angle in enumerate(features):
        print(f"Angle {i+1}: {angle:.2f}°")

    print("\nIdeal Angles:")
    for i, ideal in enumerate(ideal_angles.get(pose_name, [])):
        print(f"Ideal Angle {i+1}: {ideal}°")

    # Show the image with landmarks and angles (for visualization)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for landmark in landmarks:
            x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    
    # Display the image with landmarks
    cv2.imshow(f"Pose: {pose_name}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test the function with a specific image
test_single_image(r"C:\Users\Sushant Kabra\Desktop\NEW PROJECT\DATASET\TEST\tree\00000000.jpg")
