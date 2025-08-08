import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Ideal angles (based on football postures like kicking, sprinting, balance)
ideal_angles = {
    'left_elbow': 165,     # Slightly bent while sprinting
    'right_elbow': 165,
    'left_knee': 90,       # Bent knee while kicking
    'right_knee': 160,
    'hip_angle': 150       # Torso-hip-leg alignment
}

# Define joints
joints = {
    'left_elbow': (mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                   mp_pose.PoseLandmark.LEFT_ELBOW.value,
                   mp_pose.PoseLandmark.LEFT_WRIST.value),
    'right_elbow': (mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                    mp_pose.PoseLandmark.RIGHT_WRIST.value),
    'left_knee': (mp_pose.PoseLandmark.LEFT_HIP.value,
                  mp_pose.PoseLandmark.LEFT_KNEE.value,
                  mp_pose.PoseLandmark.LEFT_ANKLE.value),
    'right_knee': (mp_pose.PoseLandmark.RIGHT_HIP.value,
                   mp_pose.PoseLandmark.RIGHT_KNEE.value,
                   mp_pose.PoseLandmark.RIGHT_ANKLE.value),
    'hip_angle': (mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                  mp_pose.PoseLandmark.LEFT_HIP.value,
                  mp_pose.PoseLandmark.LEFT_KNEE.value)
}

# Webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        detected_angles = {}
        accuracies = {}

        for joint, (a_idx, b_idx, c_idx) in joints.items():
            a = [landmarks[a_idx].x * w, landmarks[a_idx].y * h]
            b = [landmarks[b_idx].x * w, landmarks[b_idx].y * h]
            c = [landmarks[c_idx].x * w, landmarks[c_idx].y * h]

            angle = calculate_angle(a, b, c)
            detected_angles[joint] = angle
            ideal = ideal_angles[joint]
            accuracy = max(0, 100 - (abs(angle - ideal) / 180 * 100))
            accuracies[joint] = accuracy

        overall_accuracy = np.mean(list(accuracies.values()))
        cv2.putText(frame, f'Posture Accuracy: {overall_accuracy:.1f}%', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Football Posture Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') and results.pose_landmarks:
        joints_list = list(joints.keys())
        detected = [detected_angles[j] for j in joints_list]
        ideal = [ideal_angles[j] for j in joints_list]

        x = np.arange(len(joints_list))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, detected, width, label='Detected', color='#1f77b4')
        plt.bar(x + width/2, ideal, width, label='Ideal', color='#ff7f0e')
        plt.ylabel('Angle (degrees)')
        plt.title(f'Football Joint Angle Comparison - Accuracy: {overall_accuracy:.1f}%')
        plt.xticks(x, joints_list)
        plt.legend()
        plt.tight_layout()
        plt.show()

    elif key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
