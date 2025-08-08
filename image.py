import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

# Load dataset and extract features
def load_dataset(dataset_path):
    X, y = [], []
    for pose_name in os.listdir(dataset_path):
        pose_dir = os.path.join(dataset_path, pose_name)
        if not os.path.isdir(pose_dir):
            continue
        for img_file in os.listdir(pose_dir):
            img_path = os.path.join(pose_dir, img_file)
            features = extract_features(img_path)
            if features:
                X.append(features)
                y.append(pose_name)
    return X, y

# ===================== MAIN LOGIC =====================
if __name__ == "__main__":
    # Get dataset paths from user
    train_path = r"C:\Users\Sushant Kabra\Desktop\NEW PROJECT\DATASET\TRAIN"
    test_path = r"C:\Users\Sushant Kabra\Desktop\NEW PROJECT\DATASET\TEST"




    print("\nLoading training data...")
    X_train, y_train = load_dataset(train_path)

    print("Loading test data...")
    X_test, y_test = load_dataset(test_path)

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Train model
    print("\nTraining model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "yoga_pose_model.pkl")
    print("Model saved as yoga_pose_model.pkl")

    # Prediction and Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {accuracy*100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_, cmap="YlGnBu")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Bar Graph for Correct vs Incorrect Predictions
    correct = sum(y_pred[i] == y_test[i] for i in range(len(y_test)))
    incorrect = len(y_test) - correct

    plt.figure(figsize=(6, 4))
    plt.bar(["Correct", "Incorrect"], [correct, incorrect], color=["green", "red"])
    plt.title("Correct vs Incorrect Predictions")
    plt.ylabel("Number of Predictions")
    plt.tight_layout()
    plt.show()

    pose.close()
