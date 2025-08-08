import cv2
import numpy as np
import mediapipe as mp
import joblib
import pickle

# ——— Load trained classifier & ideal keypoints ———
clf = joblib.load("pose_classifier2.pkl")
with open("ideal_keypoints.pkl", "rb") as f:
    pose_class_avg_keypoints = pickle.load(f)

# ——— MediaPipe setup ———
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5)
mp_connections = mp_pose.POSE_CONNECTIONS

# ——— Landmark subset & names ———
USE_LANDMARKS = [0, 11, 12, 13, 14, 23, 24, 25, 26]
landmark_names = {
    0: "Neck",
    11: "Left Shoulder", 12: "Right Shoulder",
    13: "Left Elbow",   14: "Right Elbow",
    23: "Left Hip",     24: "Right Hip",
    25: "Left Knee",    26: "Right Knee"
}
THRESHOLD = 15  # pixel threshold

# ——— Feature & coord extraction ———
def extract_full_keypoints(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img_rgb)
    if not res.pose_landmarks:
        return None, None
    h, w = frame.shape[:2]
    feats, coords = [], []
    for lm in res.pose_landmarks.landmark:
        feats.extend([lm.x, lm.y])           # 66-length vector
        coords.append((int(lm.x*w), int(lm.y*h)))
    return feats, coords

# ——— Drawing utilities ———
def draw_skeleton_subset(img, coords):
    for a,b in mp_connections:
        if a in USE_LANDMARKS and b in USE_LANDMARKS:
            ia, ib = USE_LANDMARKS.index(a), USE_LANDMARKS.index(b)
            cv2.line(img, coords[ia], coords[ib], (0,255,0), 2)
    for coord in coords:
        cv2.circle(img, coord, 4, (0,255,0), -1)

def draw_corrections_subset(img, user_coords, ideal_coords):
    for (ux,uy),(ix,iy), idx in zip(user_coords, ideal_coords, USE_LANDMARKS):
        dx,dy = ix-ux, iy-uy
        if abs(dx)>THRESHOLD or abs(dy)>THRESHOLD:
            cv2.arrowedLine(img, (ux,uy), (ix,iy), (0,0,255), 2, tipLength=0.3)
            # highlight joint
            cv2.circle(img, (ux,uy), 6, (0,0,255), -1)

# ——— Real-time analysis loop ———
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    feats, full_coords = extract_full_keypoints(frame)
    if feats is not None:
        # predict pose
        pose_name = clf.predict([feats])[0]

        # build subset coords
        user_coords = [full_coords[i] for i in USE_LANDMARKS]

        # get and convert ideal coords
        ideal_flat = pose_class_avg_keypoints.get(pose_name, [])
        h, w = frame.shape[:2]
        ideal_full = [(int(ideal_flat[i]*w), int(ideal_flat[i+1]*h))
                      for i in range(0, len(ideal_flat), 2)]
        ideal_coords = [ideal_full[i] for i in USE_LANDMARKS]

        # draw skeleton + corrections
        draw_skeleton_subset(frame, user_coords)
        draw_corrections_subset(frame, user_coords, ideal_coords)

        # overlay pose name
        cv2.putText(frame, f"Pose: {pose_name}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # overlay text corrections
        y0 = 60
        for (ux,uy),(ix,iy), idx in zip(user_coords, ideal_coords, USE_LANDMARKS):
            dx,dy = ix-ux, iy-uy
            if abs(dx)>THRESHOLD or abs(dy)>THRESHOLD:
                name = landmark_names.get(idx, str(idx))
                moves = []
                if dx>THRESHOLD: moves.append("right")
                if dx<-THRESHOLD: moves.append("left")
                if dy>THRESHOLD: moves.append("down")
                if dy<-THRESHOLD: moves.append("up")
                cv2.putText(frame, f"{name}: {', '.join(moves)}",
                            (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                y0 += 20

    cv2.imshow("Live Pose Correction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
