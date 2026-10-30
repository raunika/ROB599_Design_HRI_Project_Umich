import os
import time
import json
from collections import deque
import threading
import cv2
import numpy as np
from deepface import DeepFace

# ================= CONFIGURATION =================
DB_PATH = "./database"
FRAME_WIDTH = 640           # Video data stream frame width
FRAME_HEIGHT = 480          # Video data stream frame height
TARGET_FPS = 10             # Video data stream fps
OUTPUT_HZ = 2.0             # Json output frequency

PREFERRED_CAMERA_INDEX = 1  # For systems with multiple cameras
PRIMARY_USER = "User"       # Primary user interacting with the system

MODEL_NAME = 'SFace'        # Model used for face recognition
DETECTOR_BACKEND = 'opencv' # Detector used for face detection and recognition

DEADZONE_WIDTH = 0.5        # Center frame dead zone
SMOOTHING_ALPHA = 0.3       # Bounding box smoothing coefficient
RECOGNITION_INTERVAL = 0.5  # Interval used for recognotion to increase efficiency
EMOTION_INTERVAL = 0.5      # Interval used for emotion processing to increase efficiency
# =================================================

class AsyncVideoCapture:
    """Asynchronous video capture thread to eliminate camera buffer lag and input latency."""
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION if os.name == 'posix' else cv2.CAP_ANY)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Minimize internal buffer
        
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        
        # Automatically start the thread upon creation so it never freezes
        self.start()

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.stopped = True
        self.cap.release()


def init_camera(width, height, preferred_index=None):
    indices_to_try = [preferred_index] if preferred_index is not None else range(3)
    
    for idx in indices_to_try:
        if idx is None:
            continue
        print(f"[INFO] Attempting to open camera index {idx}...")
        cam = AsyncVideoCapture(idx, width, height)
        
        if cam.isOpened():
            ret, frame = cam.read()
            if ret and frame is not None:
                print(f"[INFO] Successfully connected to camera at index {idx}")
                return cam
            cam.release()
            
    if preferred_index is not None:
        print(f"[WARNING] Preferred camera index {preferred_index} failed. Falling back to auto-scan...")
        return init_camera(width, height, preferred_index=None)
        
    return None

def compute_normalized_yaw(box_center_x, frame_width, deadzone):
    rel_x = (box_center_x - (frame_width / 2.0)) / (frame_width / 2.0)
    max_effective_rel = 0.7
    scaled_rel_x = np.clip(rel_x / max_effective_rel, -1.0, 1.0)
    
    if abs(scaled_rel_x) <= (deadzone / 2.0):
        return 0.5  
    
    sign = np.sign(scaled_rel_x)
    adjusted_mag = (abs(scaled_rel_x) - (deadzone / 2.0)) / (1.0 - (deadzone / 2.0))
    
    if sign > 0:
        return round(float(0.5 + (adjusted_mag * 0.5)), 4)
    else:
        return round(float(0.5 - (adjusted_mag * 0.5)), 4)

def compute_normalized_pitch(box_center_y, frame_height, deadzone):
    rel_y = (box_center_y - (frame_height / 2.0)) / (frame_height / 2.0)
    inverted_rel_y = -rel_y
    
    max_effective_rel = 0.7
    scaled_rel_y = np.clip(inverted_rel_y / max_effective_rel, -1.0, 1.0)
    
    if abs(scaled_rel_y) <= (deadzone / 2.0):
        return 0.5  
    
    sign = np.sign(scaled_rel_y)
    adjusted_mag = (abs(scaled_rel_y) - (deadzone / 2.0)) / (1.0 - (deadzone / 2.0))
    
    if sign > 0:
        return round(float(0.5 + (adjusted_mag * 0.5)), 4)
    else:
        return round(float(0.5 - (adjusted_mag * 0.5)), 4)

def main():
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
        print(f"[WARNING] Created empty '{DB_PATH}' directory. Add subfolders with images per person.")

    cap = init_camera(FRAME_WIDTH, FRAME_HEIGHT, preferred_index=PREFERRED_CAMERA_INDEX)
    if cap is None:
        print("[ERROR] Could not open any video device.")
        return

    tracked_box = None
    tracked_name = "Unknown"
    tracked_emotion = False         
    emotion_buffer = deque(maxlen=TARGET_FPS)  
    
    smoothed_yaw = 0.5
    smoothed_pitch = 0.5

    last_recognition_time = 0.0
    last_emotion_time = 0.0
    last_output_time = time.time()

    print(f"[INFO] Starting optimized DeepFace loop at ~{TARGET_FPS} FPS. Primary User: {PRIMARY_USER}. Press 'q' to exit.")

    try:
        while cap.isOpened():
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Failed to grab frame from camera.")
                time.sleep(0.01)
                continue

            h, w, _ = frame.shape

            # Detect faces in current frame
            try:
                detected_faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                    align=False
                )
            except Exception:
                detected_faces = []

            current_best_box = None
            max_face_area = 0

            if len(detected_faces) > 0:
                for face_obj in detected_faces:
                    facial_area = face_obj.get("facial_area", {})
                    fx = facial_area.get("x", 0)
                    fy = facial_area.get("y", 0)
                    fw = facial_area.get("w", 0)
                    fh = facial_area.get("h", 0)
                    
                    area = fw * fh
                    if area > max_face_area:
                        max_face_area = area
                        current_best_box = [fx, fy, fw, fh]

            if current_best_box is not None:
                fx, fy, f_w, f_h = current_best_box
                pad_x = int(f_w * 0.1)
                pad_y = int(f_h * 0.1)
                crop_x1 = max(0, fx - pad_x)
                crop_y1 = max(0, fy - pad_y)
                crop_x2 = min(w, fx + f_w + pad_x)
                crop_y2 = min(h, fy + f_h + pad_y)
                face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                # Periodic Identity Matching (Database)
                current_time = time.time()
                if current_time - last_recognition_time >= RECOGNITION_INTERVAL:
                    last_recognition_time = current_time
                    if face_crop.size > 0:
                        try:
                            dfs = DeepFace.find(
                                img_path=face_crop, 
                                db_path=DB_PATH, 
                                model_name=MODEL_NAME,
                                detector_backend=DETECTOR_BACKEND, 
                                enforce_detection=False,
                                silent=True
                            )
                            if len(dfs) > 0 and not dfs[0].empty:
                                top_match_path = dfs[0].iloc[0]['identity']
                                tracked_name = os.path.basename(os.path.dirname(top_match_path))
                            else:
                                tracked_name = "Unknown"
                        except Exception:
                            tracked_name = "Unknown"

                # Enforce primary user filter
                if PRIMARY_USER and PRIMARY_USER.strip() != "" and tracked_name != PRIMARY_USER:
                    current_best_box = None

            # Update tracked box & motion coordinates
            if current_best_box is not None:
                bx, by, bw, bh = current_best_box

                if tracked_box is None:
                    tracked_box = current_best_box
                else:
                    tracked_box[0] = int(tracked_box[0] * (1 - SMOOTHING_ALPHA) + bx * SMOOTHING_ALPHA)
                    tracked_box[1] = int(tracked_box[1] * (1 - SMOOTHING_ALPHA) + by * SMOOTHING_ALPHA)
                    tracked_box[2] = int(tracked_box[2] * (1 - SMOOTHING_ALPHA) + bw * SMOOTHING_ALPHA)
                    tracked_box[3] = int(tracked_box[3] * (1 - SMOOTHING_ALPHA) + bh * SMOOTHING_ALPHA)

                box_center_x = tracked_box[0] + (tracked_box[2] / 2.0)
                box_center_y = tracked_box[1] + (tracked_box[3] / 2.0)
                
                raw_yaw = compute_normalized_yaw(box_center_x, w, DEADZONE_WIDTH)
                raw_pitch = compute_normalized_pitch(box_center_y, h, DEADZONE_WIDTH)
                
                smoothed_yaw = round(float(smoothed_yaw * (1 - SMOOTHING_ALPHA) + raw_yaw * SMOOTHING_ALPHA), 4)
                smoothed_pitch = round(float(smoothed_pitch * (1 - SMOOTHING_ALPHA) + raw_pitch * SMOOTHING_ALPHA), 4)

                # Throttled Emotion Confidence Analysis
                current_time = time.time()
                if current_time - last_emotion_time >= EMOTION_INTERVAL:
                    last_emotion_time = current_time
                    try:
                        analysis = DeepFace.analyze(
                            img_path=frame,
                            actions=['emotion'],
                            detector_backend=DETECTOR_BACKEND,
                            enforce_detection=False,
                            silent=True
                        )
                        
                        emotions_dict = {}
                        if isinstance(analysis, list) and len(analysis) > 0:
                            emotions_dict = analysis[0].get('emotion', {})
                        elif isinstance(analysis, dict):
                            emotions_dict = analysis.get('emotion', {})
                        
                        happy_score = emotions_dict.get('happy', 0.0)
                        surprise_score = emotions_dict.get('surprise', 0.0)

                        is_frame_positive = (happy_score + surprise_score) > 20.0
                        emotion_buffer.append(is_frame_positive)
                    except Exception:
                        emotion_buffer.append(False)
                else:
                    # Maintain buffer state smoothly between interval ticks
                    if len(emotion_buffer) > 0:
                        emotion_buffer.append(emotion_buffer[-1])
                    else:
                        emotion_buffer.append(False)
            else:
                tracked_box = None
                tracked_name = "No Face" if not PRIMARY_USER else f"Waiting for {PRIMARY_USER}"
                emotion_buffer.append(False)
                smoothed_yaw = round(float(smoothed_yaw * (1 - SMOOTHING_ALPHA) + 0.5 * SMOOTHING_ALPHA), 4)
                smoothed_pitch = round(float(smoothed_pitch * (1 - SMOOTHING_ALPHA) + 0.5 * SMOOTHING_ALPHA), 4)

            if len(emotion_buffer) > 0:
                tracked_emotion = bool(sum(emotion_buffer) > (TARGET_FPS / 4.0))
            else:
                tracked_emotion = False

            # Output JSON at 2 Hz
            current_time = time.time()
            if current_time - last_output_time >= (1.0 / OUTPUT_HZ):
                last_output_time = current_time
                output_data = {
                    "timestamp": round(current_time, 2),
                    "name": tracked_name,
                    "emotion": bool(tracked_emotion),
                    "yaw": smoothed_yaw,
                    "pitch": smoothed_pitch
                }
                print(json.dumps(output_data))

            # Preview UI
            if tracked_box is not None:
                bx, by, bw, bh = tracked_box
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(frame, f"{tracked_name} (Positive: {tracked_emotion})", (bx, max(20, by - 25)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Y: {smoothed_yaw} | P: {smoothed_pitch}", (bx, max(20, by - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("DeepFace Edge Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # FPS regulation
            elapsed = time.time() - loop_start
            if elapsed < (1.0 / TARGET_FPS):
                time.sleep((1.0 / TARGET_FPS) - elapsed)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()