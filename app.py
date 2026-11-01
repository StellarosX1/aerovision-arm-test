import cv2
import time
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

print("[INFO] Initializing High-Precision Multi-Modal Engine...")

# 1. Macro Layer: Full-Body Survivor Tracker
model = YOLO("yolo26n.pt")

# 2. Micro Layer A: Full-Range Face/Head Detector (model_selection=1 for >2m range)
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.3)

# 3. Micro Layer B: Unified 33-Landmark Pose Engine (Handles Hands, Legs & Body concurrently)
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,        # Balanced precision and high FPS execution
    smooth_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("[ERROR] Camera initialization failed.")
    exit()

print("[INFO] High-Reactivity Feed Active. Press 'q' to exit.")
prev_time = 0

def get_bbox_from_landmarks(landmarks, indices, img_w, img_h, padding=20, min_vis=0.3):
    """Calculates bounding box from specified landmark indices with confidence filtering."""
    valid_pts = []
    for idx in indices:
        lm = landmarks[idx]
        if lm.visibility > min_vis:
            valid_pts.append((int(lm.x * img_w), int(lm.y * img_h)))
    
    if not valid_pts:
        return None
    
    x_coords = [p[0] for p in valid_pts]
    y_coords = [p[1] for p in valid_pts]
    
    x1 = max(0, min(x_coords) - padding)
    y1 = max(0, min(y_coords) - padding)
    x2 = min(img_w, max(x_coords) + padding)
    y2 = min(img_h, max(y_coords) + padding)
    
    return (x1, y1, x2, y2)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    annotated_frame = frame.copy()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- 1. FULL-BODY SURVIVOR DETECTION (YOLO) ---
    results = model.track(
        frame, 
        persist=True, 
        tracker="bytetrack.yaml", 
        conf=0.30, 
        classes=[0], 
        verbose=False
    )
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, track_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
            label = f"SURVIVOR ID:{int(track_id)} [{conf:.2f}]"
            cv2.putText(annotated_frame, label, (x1 + 3, max(20, y1 - 6)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 2)

    # --- 2. FULL-RANGE FACE DETECTION ---
    face_results = face_detector.process(rgb_frame)
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            fx1 = max(0, int(bboxC.xmin * w))
            fy1 = max(0, int(bboxC.ymin * h))
            fw = int(bboxC.width * w)
            fh = int(bboxC.height * h)
            fx2, fy2 = min(w, fx1 + fw), min(h, fy1 + fh)

            cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), (255, 255, 0), 2)
            cv2.putText(annotated_frame, "FACE / HEAD", (fx1 + 3, max(14, fy1 - 6)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)

    # --- 3. UNIFIED POSE & LIMB PARSING (HANDS & LEGS) ---
    pose_results = pose_detector.process(rgb_frame)
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark

        # Left Hand (Wrist: 15, Pinky: 17, Index: 19, Thumb: 21)
        left_hand_box = get_bbox_from_landmarks(lm, [15, 17, 19, 21], w, h, padding=25)
        if left_hand_box:
            cv2.rectangle(annotated_frame, (left_hand_box[0], left_hand_box[1]), 
                          (left_hand_box[2], left_hand_box[3]), (255, 0, 255), 2)
            cv2.putText(annotated_frame, "LEFT HAND", (left_hand_box[0], max(14, left_hand_box[1] - 6)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)

        # Right Hand (Wrist: 16, Pinky: 18, Index: 20, Thumb: 22)
        right_hand_box = get_bbox_from_landmarks(lm, [16, 18, 20, 22], w, h, padding=25)
        if right_hand_box:
            cv2.rectangle(annotated_frame, (right_hand_box[0], right_hand_box[1]), 
                          (right_hand_box[2], right_hand_box[3]), (255, 0, 255), 2)
            cv2.putText(annotated_frame, "RIGHT HAND", (right_hand_box[0], max(14, right_hand_box[1] - 6)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)

        # Left Leg (Hip: 23, Knee: 25, Ankle: 27, Heel: 29, Foot: 31)
        left_leg_box = get_bbox_from_landmarks(lm, [23, 25, 27, 29, 31], w, h, padding=20)
        if left_leg_box:
            cv2.rectangle(annotated_frame, (left_leg_box[0], left_leg_box[1]), 
                          (left_leg_box[2], left_leg_box[3]), (0, 165, 255), 2)
            cv2.putText(annotated_frame, "LEFT LEG", (left_leg_box[0], max(14, left_leg_box[1] - 6)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)

        # Right Leg (Hip: 24, Knee: 26, Ankle: 28, Heel: 30, Foot: 32)
        right_leg_box = get_bbox_from_landmarks(lm, [24, 26, 28, 30, 32], w, h, padding=20)
        if right_leg_box:
            cv2.rectangle(annotated_frame, (right_leg_box[0], right_leg_box[1]), 
                          (right_leg_box[2], right_leg_box[3]), (0, 165, 255), 2)
            cv2.putText(annotated_frame, "RIGHT LEG", (right_leg_box[0], max(14, right_leg_box[1] - 6)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)

    # --- FPS DISPLAY ---
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (w - 140, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Aero-Vision Ultra-Reactive Tracker", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()