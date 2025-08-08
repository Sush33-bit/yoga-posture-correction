import os
import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from datetime import datetime

# ─── Mediapipe Pose setup ──────────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ─── Ideal pose angles ─────────────────────────────────────────────────────────
POSE_LIBRARY = {
    "Warrior II": {
        "left_knee": 90,
        "right_knee": 180,
        "left_hip": 180,
        "right_hip": 90,
        "shoulders": 180
    }
}

# ─── Angle definitions (A–B–C landmarks) ───────────────────────────────────────
ANGLE_DEFINITIONS = {
    "left_knee":    (mp_pose.PoseLandmark.LEFT_HIP,
                     mp_pose.PoseLandmark.LEFT_KNEE,
                     mp_pose.PoseLandmark.LEFT_ANKLE),
    "right_knee":   (mp_pose.PoseLandmark.RIGHT_HIP,
                     mp_pose.PoseLandmark.RIGHT_KNEE,
                     mp_pose.PoseLandmark.RIGHT_ANKLE),
    "left_hip":     (mp_pose.PoseLandmark.LEFT_SHOULDER,
                     mp_pose.PoseLandmark.LEFT_HIP,
                     mp_pose.PoseLandmark.LEFT_KNEE),
    "right_hip":    (mp_pose.PoseLandmark.RIGHT_SHOULDER,
                     mp_pose.PoseLandmark.RIGHT_HIP,
                     mp_pose.PoseLandmark.RIGHT_KNEE),
    "shoulders":    (mp_pose.PoseLandmark.LEFT_SHOULDER,
                     mp_pose.PoseLandmark.RIGHT_SHOULDER,
                     mp_pose.PoseLandmark.RIGHT_ELBOW),
}

class YogaAnalyzer:
    def __init__(self, video_path, pose_name):
        self.cap = cv2.VideoCapture(video_path)
        self.pose_name = pose_name
        self.ideal = POSE_LIBRARY[pose_name]
        self.current = {}
        self.history = []
        self.graph_w, self.graph_h = 400, 300
        self.graph = np.zeros((self.graph_h, self.graph_w, 3), np.uint8)
        self.prev_time = time.time()
        self.hold_start = None
        self.hold_duration = 0
        self.csv_data = []
        self.seeking = False
        self.progress_h = 20
        self.frame_num = 0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25

        # snapshot logic
        os.makedirs("snapshots", exist_ok=True)
        self.perfect_saved = False

        # summary report fields
        self.report = {
            "sum_acc": 0,
            "count": 0,
            "max_acc": 0,
            "min_acc": 100,
            "best_frame": 0,
            "worst_frame": 0,
            "max_hold": 0
        }

        cv2.namedWindow("Yoga Pose Analyzer")
        cv2.setMouseCallback("Yoga Pose Analyzer", self.on_mouse)

    def on_mouse(self, e, x, y, _flags, _param):
        if e == cv2.EVENT_LBUTTONDOWN and y > 480 - self.progress_h:
            frac = np.clip(x / 640, 0, 1)
            self.frame_num = int(frac * self.total)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)
            self.seeking = True

    def draw_progress(self, frame):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - self.progress_h), (w, h), (50, 50, 50), -1)
        p = self.frame_num / self.total
        cv2.rectangle(frame, (0, h - self.progress_h),
                      (int(w * p), h), (0, 255, 0), -1)
        t1 = self.frame_num / self.fps
        t2 = self.total / self.fps
        txt = f"{t1:.1f}s / {t2:.1f}s"
        cv2.putText(frame, txt, (10, h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def calc_angles(self, lms):
        for name, (A, B, C) in ANGLE_DEFINITIONS.items():
            a = np.array([lms[A.value].x, lms[A.value].y])
            b = np.array([lms[B.value].x, lms[B.value].y])
            c = np.array([lms[C.value].x, lms[C.value].y])
            BA, BC = a - b, c - b
            cosang = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
            ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            self.current[name] = ang

    def calc_accuracy(self):
        total, cnt = 0, 0
        errs = {}
        for k, ideal in self.ideal.items():
            actual = self.current.get(k, ideal)
            err = abs(ideal - actual)
            errs[k] = err
            total += max(0, 100 - err)
            cnt += 1
        return (int(total / cnt) if cnt else 0), errs

    def track_hold(self, acc):
        if acc > 90:
            if self.hold_start is None:
                self.hold_start = time.time()
            self.hold_duration = int(time.time() - self.hold_start)
            self.report["max_hold"] = max(self.report["max_hold"], self.hold_duration)
        else:
            self.hold_start = None
            self.hold_duration = 0

    def update_graph(self):
        self.graph[:] = (30, 30, 30)
        for v in range(0, 101, 20):
            y = int(self.graph_h * (1 - v / 100))
            cv2.line(self.graph, (0, y), (self.graph_w, y), (50, 50, 50), 1)
            cv2.putText(self.graph, str(v), (2, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(self.graph, "Accuracy (%)", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.graph, "Frame ->", (self.graph_w // 2 - 30, self.graph_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        hist = self.history[-self.graph_w:]
        for i in range(1, len(hist)):
            y1 = int(self.graph_h * (1 - hist[i - 1] / 100))
            y2 = int(self.graph_h * (1 - hist[i] / 100))
            cv2.line(self.graph, (i - 1, y1), (i, y2), (0, 255, 0), 1)

    def process(self, frame):
        frame = cv2.resize(frame, (640, 480))
        self.draw_progress(frame)

        acc = 0
        errs = {}
        if not self.seeking:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                self.calc_angles(res.pose_landmarks.landmark)
                acc, errs = self.calc_accuracy()
                self.history.append(acc)

                # summary report updates
                self.report["sum_acc"] += acc
                self.report["count"] += 1
                if acc > self.report["max_acc"]:
                    self.report["max_acc"], self.report["best_frame"] = acc, self.frame_num
                if acc < self.report["min_acc"]:
                    self.report["min_acc"], self.report["worst_frame"] = acc, self.frame_num

                self.track_hold(acc)
                self.csv_data.append((datetime.now().strftime("%H:%M:%S"), acc))

                # draw default landmarks & connections
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                # color-coded bones & ASCII angle labels
                for name, (A, B, C) in ANGLE_DEFINITIONS.items():
                    err = errs[name]
                    color = (0, 255, 0) if err <= 15 else (0, 0, 255)
                    pA = res.pose_landmarks.landmark[A.value]
                    pB = res.pose_landmarks.landmark[B.value]
                    pC = res.pose_landmarks.landmark[C.value]
                    ptA = (int(pA.x * 640), int(pA.y * 480))
                    ptB = (int(pB.x * 640), int(pB.y * 480))
                    ptC = (int(pC.x * 640), int(pC.y * 480))
                    cv2.line(frame, ptA, ptB, color, 2)
                    cv2.line(frame, ptB, ptC, color, 2)
                    cv2.putText(frame,
                                f"{int(self.current[name])}deg",
                                (ptB[0] + 5, ptB[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # highlight worst joint
                worst = max(errs, key=errs.get)
                if errs[worst] > 15:
                    B = ANGLE_DEFINITIONS[worst][1]
                    lm = res.pose_landmarks.landmark[B.value]
                    x_px, y_px = int(lm.x * 640), int(lm.y * 480)
                    cv2.circle(frame, (x_px, y_px), 15, (0, 0, 255), -1)

                # auto-snapshot on perfect hold
                if acc > 90 and self.hold_duration >= 5 and not self.perfect_saved:
                    fn = f"snapshots/{self.pose_name}_{datetime.now():%Y%m%d_%H%M%S}.png"
                    cv2.imwrite(fn, frame)
                    self.perfect_saved = True

        else:
            if self.history:
                acc = self.history[-1]
        self.seeking = False

        # overlay metrics
        cv2.putText(frame, f"Accuracy: {acc}%", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Hold: {self.hold_duration}s", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        return frame

    def save_csv(self):
        fn = f"{self.pose_name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with open(fn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Timestamp", "Accuracy"])
            w.writerows(self.csv_data)
        print("Saved CSV:", fn)

    def print_report(self):
        cnt = self.report["count"]
        avg = (self.report["sum_acc"] / cnt) if cnt else 0
        print("\n── SUMMARY REPORT ─────────────────")
        print(f"Average Accuracy : {avg:.1f}%")
        print(f"Best Frame       : {self.report['best_frame']} @ {self.report['max_acc']}%")
        print(f"Worst Frame      : {self.report['worst_frame']} @ {self.report['min_acc']}%")
        print(f"Max Hold Time    : {self.report['max_hold']}s")
        print("Snapshot saved   :", "Yes" if self.perfect_saved else "No")
        print("────────────────────────────────────\n")

    def run(self):
        while self.cap.isOpened():
            ok, frm = self.cap.read()
            if not ok:
                break
            self.frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            out = self.process(frm)

            # FPS overlay
            now = time.time()
            fps = int(1 / (now - self.prev_time + 1e-6))
            self.prev_time = now
            cv2.putText(out, f"FPS: {fps}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # combine video + graph
            self.update_graph()
            vis = np.zeros((480, 640 + self.graph_w, 3), np.uint8)
            vis[:, :640] = out
            vis[:self.graph_h, 640:] = self.graph

            cv2.imshow("Yoga Pose Analyzer", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

        self.cap.release()
        self.save_csv()
        self.print_report()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    YogaAnalyzer(
        "The BEST way to start your day!  _  10-Minute Morning Yoga.mp4",
        "Warrior II"
    ).run()
