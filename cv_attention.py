# cv_attention.py
import cv2
import numpy as np
import mediapipe as mp
import math
from collections import deque
import time

mp_face = mp.solutions.face_mesh

# Useful landmark indices (MediaPipe FaceMesh)
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]   # approximate eye landmarks for EAR
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# Eye corner indices for simple iris-to-eye geometry
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263

# Iris landmark indices introduced by MediaPipe Iris (468-473 region)
LEFT_IRIS_IDX = [468, 469, 470, 471]
RIGHT_IRIS_IDX = [473, 474, 475, 476]  # sometimes 473-476 depending on versions

# Parameters
EYE_AR_THRESH = 0.20     # threshold for EAR to consider eyes closed
EYE_AR_CONSEC_FRAMES = 3
GAZE_HORIZ_THRESHOLD = 0.35   # normalized threshold for left/right
GAZE_VERT_THRESHOLD = 0.30    # vertical threshold

class AttentionDetector:
    def __init__(self, max_history_seconds=10):
        self.face_mesh = mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,   # refine gives iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.blink_counter = 0
        self.closed_frames = 0
        self.blink_timestamps = deque()
        self.last_time = time.time()

    def _landmark_to_point(self, lm, shape):
        h, w = shape
        return np.array([int(lm.x * w), int(lm.y * h)], dtype=int)

    def eye_aspect_ratio(self, landmarks, indices, img_shape):
        # Using three vertical / one horizontal distances as approximation
        pts = [self._landmark_to_point(landmarks[i], img_shape) for i in indices]
        # pts order for our list: [p0 outer, p1 upper, p2 upper2, p3 inner, p4 lower, p5 lower2]
        # vertical distances
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h = np.linalg.norm(pts[0] - pts[3]) + 1e-6
        ear = (v1 + v2) / (2.0 * h)
        return ear, pts

    def iris_center(self, landmarks, iris_indices, img_shape):
        pts = [self._landmark_to_point(landmarks[i], img_shape) for i in iris_indices]
        pts = np.array(pts)
        center = pts.mean(axis=0).astype(int)
        return center, pts

    def estimate_gaze(self, landmarks, img_shape):
        # compute normalized iris offset inside eye horizontally and vertically
        h, w = img_shape
        # left eye
        try:
            left_outer = self._landmark_to_point(landmarks[LEFT_EYE_OUTER], img_shape)
            left_inner = self._landmark_to_point(landmarks[LEFT_EYE_INNER], img_shape)
            right_outer = self._landmark_to_point(landmarks[RIGHT_EYE_OUTER], img_shape)
            right_inner = self._landmark_to_point(landmarks[RIGHT_EYE_INNER], img_shape)

            left_iris_center, left_iris_pts = self.iris_center(landmarks, LEFT_IRIS_IDX, img_shape)
            right_iris_center, right_iris_pts = self.iris_center(landmarks, RIGHT_IRIS_IDX, img_shape)

            # normalized horizontal position: 0 (outer) .. 1 (inner)
            left_norm_h = (left_iris_center[0] - left_outer[0]) / (left_inner[0] - left_outer[0] + 1e-6)
            right_norm_h = (right_iris_center[0] - right_outer[0]) / (right_inner[0] - right_outer[0] + 1e-6)
            # normalized vertical relative to eye bbox
            eye_top = min(left_iris_pts[:,1].min(), right_iris_pts[:,1].min())
            eye_bottom = max(left_iris_pts[:,1].max(), right_iris_pts[:,1].max())
            left_norm_v = (left_iris_center[1] - eye_top) / (eye_bottom - eye_top + 1e-6)
            right_norm_v = (right_iris_center[1] - eye_top) / (eye_bottom - eye_top + 1e-6)

            horiz = (left_norm_h + right_norm_h) / 2.0
            vert = (left_norm_v + right_norm_v) / 2.0

            # decide categorical gaze
            horiz_cat = 'center'
            if horiz < 0.5 - GAZE_HORIZ_THRESHOLD:
                horiz_cat = 'left'
            elif horiz > 0.5 + GAZE_HORIZ_THRESHOLD:
                horiz_cat = 'right'

            vert_cat = 'center'
            if vert < 0.5 - GAZE_VERT_THRESHOLD:
                vert_cat = 'up'
            elif vert > 0.5 + GAZE_VERT_THRESHOLD:
                vert_cat = 'down'

            return {'h':horiz, 'v':vert, 'h_cat':horiz_cat, 'v_cat':vert_cat,
                    'left_iris': left_iris_center, 'right_iris': right_iris_center}
        except Exception as e:
            return None

    def process_frame(self, frame):
        img_h, img_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        info = {'present': False, 'blink': False, 'gaze': None, 'head_pose': None, 'attention_score': 0.0}

        if results.multi_face_landmarks:
            info['present'] = True
            lm = results.multi_face_landmarks[0].landmark

            # EAR left & right
            left_ear, left_pts = self.eye_aspect_ratio(lm, LEFT_EYE_LANDMARKS, (img_h, img_w))
            right_ear, right_pts = self.eye_aspect_ratio(lm, RIGHT_EYE_LANDMARKS, (img_h, img_w))
            ear = (left_ear + right_ear) / 2.0

            # blink detection logic
            if ear < EYE_AR_THRESH:
                self.closed_frames += 1
            else:
                if self.closed_frames >= EYE_AR_CONSEC_FRAMES:
                    self.blink_counter += 1
                    self.blink_timestamps.append(time.time())
                    info['blink'] = True
                self.closed_frames = 0

            # compute blink rate over last 30s
            now = time.time()
            while self.blink_timestamps and now - self.blink_timestamps[0] > 30.0:
                self.blink_timestamps.popleft()
            blink_rate = len(self.blink_timestamps) / 30.0  # blinks per second (avg)
            info['blink_rate'] = blink_rate

            # gaze estimate
            gaze = self.estimate_gaze(lm, (img_h, img_w))
            info['gaze'] = gaze

            # annotated drawing
            annotated = frame.copy()
            # draw eye landmarks and iris
            for p in left_pts:
                cv2.circle(annotated, tuple(p), 1, (0,255,0), -1)
            for p in right_pts:
                cv2.circle(annotated, tuple(p), 1, (0,255,0), -1)

            if gaze:
                cv2.circle(annotated, tuple(gaze['left_iris']), 3, (0,0,255), -1)
                cv2.circle(annotated, tuple(gaze['right_iris']), 3, (0,0,255), -1)
                cv2.putText(annotated, f"Gaze: {gaze['h_cat']},{gaze['v_cat']}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

            # attention score heuristic
            score = 0.0
            # presence counts for half
            score += 0.5
            # gaze centered adds up to 0.3
            if gaze:
                if gaze['h_cat']=='center' and gaze['v_cat']=='center':
                    score += 0.3
                elif gaze['h_cat']=='center' or gaze['v_cat']=='center':
                    score += 0.15
            # low blink rate adds up to 0.2
            if blink_rate < 0.5:  # <0.5 blinks/sec ~ normal
                score += 0.2
            info['attention_score'] = round(min(1.0, score), 2)

            # overlay score
            cv2.putText(annotated, f"Attention: {info['attention_score']}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,0) if info['attention_score']>0.6 else (0,0,255), 2)
            info['annotated_frame'] = annotated
            info['ear'] = ear
            info['blink_count'] = self.blink_counter
        else:
            # no face
            info['annotated_frame'] = frame
            info['attention_score'] = 0.0

        return info

def demo_camera_loop():
    cap = cv2.VideoCapture(0)
    det = AttentionDetector()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = det.process_frame(frame)
        annotated = res['annotated_frame']
        cv2.imshow("Attention demo", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    demo_camera_loop()
