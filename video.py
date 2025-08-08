import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Ideal angles for a yoga pose (Warrior II example)
ideal_angles = {
    'left_knee': 90,
    'right_knee': 180,
    'left_elbow': 180,
    'right_elbow': 180,
    'hip_angle': 90,
    'shoulder_angle': 180
}

# Define joint connections
joints = {
    'left_knee': (mp.solutions.pose.PoseLandmark.LEFT_HIP.value,
                  mp.solutions.pose.PoseLandmark.LEFT_KNEE.value,
                  mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value),
    'right_knee': (mp.solutions.pose.PoseLandmark.RIGHT_HIP.value,
                   mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value,
                   mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value),
    'left_elbow': (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
                   mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value,
                   mp.solutions.pose.PoseLandmark.LEFT_WRIST.value),
    'right_elbow': (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
                    mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value,
                    mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value),
    'hip_angle': (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
                  mp.solutions.pose.PoseLandmark.LEFT_HIP.value,
                  mp.solutions.pose.PoseLandmark.LEFT_KNEE.value),
    'shoulder_angle': (mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value,
                       mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
                       mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value)
}

# Load your yoga video
video_path = 'How to do Kapalbhati_ Step by step Tutorial & Benefits _ Detox your Body_ simple pranayam guide.mp4'
cap = cv2.VideoCapture(video_path)

# Setup MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

frame_count = 0
accuracy_list = []


if False:
    from tensorflow.keras.models import load_model

    # Load a pretrained yoga pose classifier model
    model = load_model('yoga_pose_classifier.h5')

    def predict_pose_from_landmarks(landmarks):
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten().reshape(1, -1)
        prediction = model.predict(keypoints)
        predicted_class = np.argmax(prediction)
        return predicted_class

    # Example usage (this won't run):
    # predicted_pose = predict_pose_from_landmarks(results.pose_landmarks.landmark)
    # print("Predicted Pose Class:", predicted_pose)

# Main video processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Draw landmarks
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        landmarks = results.pose_landmarks.landmark
        accuracies = {}

        for joint, (a_idx, b_idx, c_idx) in joints.items():
            a = [landmarks[a_idx].x * w, landmarks[a_idx].y * h]
            b = [landmarks[b_idx].x * w, landmarks[b_idx].y * h]
            c = [landmarks[c_idx].x * w, landmarks[c_idx].y * h]

            angle = calculate_angle(a, b, c)
            ideal = ideal_angles[joint]
            accuracy = max(0, 100 - (abs(angle - ideal) / 180 * 100))
            accuracies[joint] = accuracy

        overall_accuracy = np.mean(list(accuracies.values()))
        accuracy_list.append(overall_accuracy)

        # Text settings
        text = f'Yoga Pose Accuracy: {overall_accuracy:.1f}%'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x, text_y = 30, 60

        # Background rectangle
        cv2.rectangle(frame, (text_x - 10, text_y - 40), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

    frame_count += 1
    cv2.imshow('Yoga Pose Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to break
        break

cap.release()
cv2.destroyAllWindows()
pose.close()

# Plot the accuracy trend
if accuracy_list:
    plt.figure(figsize=(12, 5))
    plt.plot(accuracy_list, color='purple', linewidth=2)
    plt.title('Yoga Pose Accuracy Frame by Frame')
    plt.xlabel('Frame Number')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
