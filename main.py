"""
Smart Fencing System - Main Application
Real-Time Detection Based on Smart Fencing System for Farm Security
Author: Rohan Nandanwar
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
import threading
import time
import csv
import os
from datetime import datetime
from PIL import Image, ImageTk
import numpy as np

# Try importing optional dependencies
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not installed. Run: pip install ultralytics")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("[WARNING] deepface not installed. Run: pip install deepface")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARNING] tensorflow not installed. Run: pip install tensorflow")

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARNING] pygame not installed. Run: pip install pygame")

try:
    import pywhatkit
    WHATSAPP_AVAILABLE = True
except ImportError:
    WHATSAPP_AVAILABLE = False
    print("[WARNING] pywhatkit not installed. Run: pip install pywhatkit")

# ─────────────────────────────────────────────
#  SHOCK RULES TABLE
# ─────────────────────────────────────────────
SHOCK_RULES = {
    # Humans
    "adult_male":    {"shock": True,  "current_uA": 4000,  "action": "4000 µA Shock Deterrence"},
    "adult_female":  {"shock": True,  "current_uA": 2500,  "action": "2500 µA Shock Deterrence"},
    "child":         {"shock": False, "current_uA": 0,     "action": "No Shock (Safety Override)"},
    "unknown_human": {"shock": False, "current_uA": 0,     "action": "Alert Only"},
    # Animals
    "cow":       {"shock": True,  "current_uA": 2500, "action": "2500 µA Shock Deterrence"},
    "dog":       {"shock": True,  "current_uA": 1800, "action": "1800 µA Shock Deterrence"},
    "horse":     {"shock": True,  "current_uA": 2500, "action": "2500 µA Shock Deterrence"},
    "sheep":     {"shock": True,  "current_uA": 2000, "action": "2000 µA Shock Deterrence"},
    "elephant":  {"shock": True,  "current_uA": 4000, "action": "4000 µA Shock Deterrence"},
    "bear":      {"shock": True,  "current_uA": 4000, "action": "4000 µA Shock Deterrence"},
    "zebra":     {"shock": True,  "current_uA": 2500, "action": "2500 µA Shock Deterrence"},
    "giraffe":   {"shock": True,  "current_uA": 4000, "action": "4000 µA Shock Deterrence"},
    "cat":       {"shock": False, "current_uA": 1500, "action": "1500 µA Shock Deterrence"},
    "bird":      {"shock": False, "current_uA": 0,    "action": "Ultrasonic / Buzzer"},
    "chicken":   {"shock": False, "current_uA": 0,    "action": "Ultrasonic / Buzzer"},
    "squirrel":  {"shock": False, "current_uA": 0,    "action": "Ultrasonic / Buzzer"},
    "butterfly": {"shock": False, "current_uA": 0,    "action": "No Action"},
    "spider":    {"shock": False, "current_uA": 0,    "action": "No Action"},
}

# YOLO class names → our category keys
ANIMAL_MAP = {
    "bird": "bird", "cat": "cat", "dog": "dog", "horse": "horse",
    "sheep": "sheep", "cow": "cow", "elephant": "elephant", "bear": "bear",
    "zebra": "zebra", "giraffe": "giraffe"
}

# ─────────────────────────────────────────────
#  DETECTION ENGINE
# ─────────────────────────────────────────────
class DetectionEngine:
    # Ordered list of DeepFace detector backends to try (best → worst)
    FACE_BACKENDS = ["retinaface", "mtcnn", "opencv", "ssd"]

    def __init__(self, model_name="yolov8m.pt"):
        self.model = None
        self.animal_cnn = None
        self.model_name = model_name
        self.frame_count = 0
        self.deepface_skip = 4
        self._last_human_info = {}
        self._best_backend = "retinaface"  # will auto-select the best working one
        self._load_models()

    def _load_models(self):
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(self.model_name)  # type: ignore[reportPossiblyUnboundVariable]
                print(f"[OK] YOLO model loaded: {self.model_name}")
            except Exception as e:
                print(f"[ERROR] YOLO load failed: {e}")
        if TF_AVAILABLE and os.path.exists("animal10.h5"):
            try:
                self.animal_cnn = tf.keras.models.load_model("animal10.h5")  # type: ignore[reportPossiblyUnboundVariable]
                print("[OK] Animal-10 CNN loaded.")
            except Exception as e:
                print(f"[WARN] Animal CNN load failed: {e}")
        # Probe which face-detection backend works
        self._best_backend = self._probe_backend()

    def change_model(self, model_name):
        self.model_name = model_name
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_name)  # type: ignore[reportPossiblyUnboundVariable]
                print(f"[OK] Switched to: {model_name}")
            except Exception as e:
                print(f"[ERROR] {e}")

    def analyze_frame(self, frame):
        """Returns (annotated_frame, human_detections, animal_detections, object_detections)"""
        if self.model is None:
            return frame, [], [], []

        self.frame_count += 1
        annotated = frame.copy()
        humans, animals, objects = [], [], []

        results = self.model(frame, verbose=False, conf=0.35)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id].lower()
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_name == "person":
                age, gender = self._analyze_human(frame, x1, y1, x2, y2)
                rule_key = self._get_human_rule(age, gender)
                rule = SHOCK_RULES.get(rule_key, SHOCK_RULES["unknown_human"])
                category = self._human_category_label(age, gender)
                info = {
                    "label": f"{category} | Age: {age}",
                    "action": rule["action"],
                    "shock": rule["shock"],
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2)
                }
                humans.append(info)
                color = (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                overlay_text = f"{category}, Age:{age} ({conf:.2f})"
                cv2.putText(annotated, overlay_text, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            elif cls_name in ANIMAL_MAP:
                species = ANIMAL_MAP[cls_name]
                rule = SHOCK_RULES.get(species, {"action": "Alert", "shock": False})
                info = {
                    "label": f"{species.capitalize()} | Conf:{conf:.2f}",
                    "action": rule["action"],
                    "shock": rule["shock"],
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2)
                }
                animals.append(info)
                color = (255, 165, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{species} {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            else:
                info = {
                    "label": f"{cls_name.capitalize()} | Conf:{conf:.2f}",
                    "action": "Alert",
                    "shock": False,
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2)
                }
                objects.append(info)
                color = (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{cls_name} {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        return annotated, humans, animals, objects

    def _analyze_human(self, frame, x1, y1, x2, y2):
        """Detect age and gender using DeepFace + visual feature correction."""
        if not DEEPFACE_AVAILABLE:
            # Even without DeepFace, try visual features
            visual_gender = self._visual_gender_features(frame, x1, y1, x2, y2)
            return "?", visual_gender
        if self.frame_count % self.deepface_skip != 0:
            cached = self._last_human_info
            return cached.get("age", "?"), cached.get("gender", "Unknown")
        try:
            h = y2 - y1
            roi = frame[max(0, y1):y2, max(0, x1):x2]
            if roi.size == 0:
                return "?", "Unknown"

            # Prepare crops: upper body is most likely to contain face
            upper_h = max(h // 2, 60)
            upper = frame[max(0, y1):min(frame.shape[0], max(0, y1) + upper_h),
                          max(0, x1):min(frame.shape[1], x2)]

            # Upscale small crops for better face-detection accuracy
            crops = []
            for c in (upper, roi):
                if c is not None and c.size > 0:
                    ch, cw = c.shape[:2]
                    if ch < 160 or cw < 160:
                        scale = max(160 / ch, 160 / cw, 1.0)
                        c = cv2.resize(c, (int(cw * scale), int(ch * scale)),
                                       interpolation=cv2.INTER_LANCZOS4)
                    crops.append(c)

            deepface_age = None
            deepface_gender = None
            deepface_conf = 0.0
            gender_scores = {}

            # Try the best backend first, then fallbacks
            backends_to_try = [self._best_backend] + [
                b for b in self.FACE_BACKENDS if b != self._best_backend
            ] + ["skip"]

            for crop in crops:
                for backend in backends_to_try:
                    try:
                        result = DeepFace.analyze(  # type: ignore[reportPossiblyUnboundVariable]
                            crop,
                            actions=["age", "gender"],
                            enforce_detection=False,
                            silent=True,
                            detector_backend=backend
                        )
                        if isinstance(result, list):
                            result = result[0]
                        age = result.get("age", None)  # type: ignore[reportAttributeAccessIssue]
                        gender = result.get("dominant_gender", None)  # type: ignore[reportAttributeAccessIssue]
                        face_conf = result.get("face_confidence", 0)  # type: ignore[reportAttributeAccessIssue]
                        gender_scores = result.get("gender", {})  # type: ignore[reportAttributeAccessIssue]
                        if age is not None and gender is not None:
                            deepface_age = age
                            deepface_gender = gender
                            deepface_conf = float(face_conf) if face_conf else 0.0
                            if deepface_conf > 0.8:
                                # High-confidence face detection — trust it
                                break
                    except Exception:
                        continue
                if deepface_age is not None and deepface_conf > 0.8:
                    break

            # Get visual feature-based gender prediction
            visual_gender = self._visual_gender_features(frame, x1, y1, x2, y2)

            # Combine DeepFace + visual features for final gender
            final_gender = self._fuse_gender(
                deepface_gender, deepface_conf, gender_scores, visual_gender
            )
            final_age = deepface_age if deepface_age is not None else "?"

            self._last_human_info = {"age": final_age, "gender": final_gender}
            print(f"[FACE] DeepFace={deepface_gender}(conf={deepface_conf:.2f}), "
                  f"Visual={visual_gender}, Final={final_gender}, Age={final_age}")
            return final_age, final_gender

        except Exception as e:
            print(f"[FACE ERROR] {e}")
            return "?", "Unknown"

    def _visual_gender_features(self, frame, x1, y1, x2, y2):
        """
        Analyze visual features to estimate gender:
        - Long hair detection (dark region below head extending past shoulders)
        - Body shape (hip-to-shoulder width ratio)
        - Upper body curvature
        Returns: 'Woman', 'Man', or 'Unknown'
        """
        try:
            h = y2 - y1
            w = x2 - x1
            roi = frame[max(0, y1):y2, max(0, x1):x2]
            if roi.size == 0:
                return "Unknown"

            score = 0.0  # positive = Woman, negative = Man

            # ── 1) HAIR LENGTH DETECTION ──────────────────
            # Analyze the upper 40% for dark regions (hair)
            head_region = roi[0:int(h * 0.4), :]
            if head_region.size > 0:
                gray_head = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
                # Hair is typically dark (< 80 intensity)
                hair_mask = (gray_head < 80).astype(np.uint8)
                head_h, head_w = gray_head.shape[:2]

                # Check if dark pixels extend to shoulder level (long hair indicator)
                if head_h > 10 and head_w > 10:
                    # Divide head region into top half and bottom half
                    top_hair = hair_mask[:head_h // 2, :]
                    bottom_hair = hair_mask[head_h // 2:, :]
                    top_ratio = np.sum(top_hair) / max(top_hair.size, 1)
                    bottom_ratio = np.sum(bottom_hair) / max(bottom_hair.size, 1)

                    # Long hair: significant dark pixels in both top AND bottom of head region
                    if top_ratio > 0.15 and bottom_ratio > 0.12:
                        # Hair extends down — likely long hair
                        hair_extent = bottom_ratio / max(top_ratio, 0.01)
                        if hair_extent > 0.4:
                            score += 1.5  # Strong long-hair signal → Woman
                            print(f"  [VISUAL] Long hair detected (top={top_ratio:.2f}, bottom={bottom_ratio:.2f})")

                    # Check hair width spread (long hair is usually wider)
                    hair_cols = np.sum(hair_mask, axis=0)
                    hair_spread = np.sum(hair_cols > 0) / max(head_w, 1)
                    if hair_spread > 0.6:
                        score += 0.8  # Wide hair spread → Woman
                        print(f"  [VISUAL] Wide hair spread: {hair_spread:.2f}")

            # ── 2) BODY SHAPE (HIP-TO-SHOULDER RATIO) ────
            # Wider hips relative to shoulders → more likely Woman
            if h > 40 and w > 20:
                shoulder_strip = roi[int(h * 0.2):int(h * 0.3), :]
                hip_strip = roi[int(h * 0.55):int(h * 0.7), :]

                if shoulder_strip.size > 0 and hip_strip.size > 0:
                    # Use edge detection to find body width at shoulders vs hips
                    gray_s = cv2.cvtColor(shoulder_strip, cv2.COLOR_BGR2GRAY)
                    gray_h = cv2.cvtColor(hip_strip, cv2.COLOR_BGR2GRAY)
                    edges_s = cv2.Canny(gray_s, 50, 150)
                    edges_h = cv2.Canny(gray_h, 50, 150)

                    s_cols = np.sum(edges_s, axis=0)
                    h_cols = np.sum(edges_h, axis=0)

                    s_width = np.sum(s_cols > 0)
                    h_width = np.sum(h_cols > 0)

                    if s_width > 5:
                        hip_shoulder_ratio = h_width / max(s_width, 1)
                        if hip_shoulder_ratio > 1.15:
                            score += 1.0  # Wider hips → Woman
                            print(f"  [VISUAL] Hip/shoulder ratio: {hip_shoulder_ratio:.2f} → Woman")
                        elif hip_shoulder_ratio < 0.85:
                            score -= 0.8  # Wider shoulders → Man
                            print(f"  [VISUAL] Hip/shoulder ratio: {hip_shoulder_ratio:.2f} → Man")

            # ── 3) COLOR ANALYSIS ─────────────────────────
            # Women's clothing often has warmer/brighter colors
            if roi.size > 0:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                # Check for warm colors (reds, pinks, purples) in clothing area
                clothing = hsv[int(h * 0.25):int(h * 0.7), :]
                if clothing.size > 0:
                    # Pink/red hue range (roughly 0-10, 160-180)
                    warm_mask = ((clothing[:, :, 0] < 15) | (clothing[:, :, 0] > 155)) & \
                                (clothing[:, :, 1] > 50)
                    warm_ratio = np.sum(warm_mask) / max(warm_mask.size, 1)
                    if warm_ratio > 0.1:
                        score += 0.5
                        print(f"  [VISUAL] Warm clothing colors: {warm_ratio:.2f}")

            # ── 4) SKIN AREA SMOOTHNESS ───────────────────
            # Women typically have smoother skin texture
            upper_body = roi[0:int(h * 0.4), :]
            if upper_body.size > 0:
                gray_ub = cv2.cvtColor(upper_body, cv2.COLOR_BGR2GRAY)
                # Detect skin-colored pixels in HSV
                hsv_ub = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
                skin_mask = (hsv_ub[:, :, 0] >= 0) & (hsv_ub[:, :, 0] <= 25) & \
                            (hsv_ub[:, :, 1] >= 30) & (hsv_ub[:, :, 2] >= 60)
                skin_pixels = np.sum(skin_mask)
                if skin_pixels > 100:
                    # Check texture roughness via Laplacian variance on skin areas
                    skin_gray = gray_ub.copy()
                    skin_gray[~skin_mask] = 0
                    laplacian_var = cv2.Laplacian(skin_gray, cv2.CV_64F).var()
                    if laplacian_var < 500:
                        score += 0.4  # Smoother skin → slight Woman bias
                    elif laplacian_var > 1500:
                        score -= 0.4  # Rougher texture → slight Man bias

            # Final decision
            if score >= 1.5:
                return "Woman"
            elif score <= -1.0:
                return "Man"
            return "Unknown"

        except Exception as e:
            print(f"  [VISUAL ERROR] {e}")
            return "Unknown"

    def _fuse_gender(self, deepface_gender, deepface_conf, gender_scores, visual_gender):
        """
        Combine DeepFace prediction with visual features for final gender.
        - High-confidence DeepFace with agreeing visual → trust it
        - Low-confidence or disagreement → use visual features
        """
        if deepface_gender is None and visual_gender == "Unknown":
            return "Unknown"
        if deepface_gender is None:
            return visual_gender

        df_str = str(deepface_gender).lower()
        df_is_woman = "woman" in df_str or "female" in df_str

        # Get DeepFace gender probability if available
        df_gender_prob = 0.5
        if isinstance(gender_scores, dict):
            woman_prob = gender_scores.get("Woman", gender_scores.get("Female", 50))
            df_gender_prob = float(woman_prob) / 100.0  # 0-1 scale, higher = more Woman

        # High-confidence face detection AND strong gender probability → trust DeepFace
        if deepface_conf > 0.8 and (df_gender_prob > 0.75 or df_gender_prob < 0.25):
            print(f"  [FUSE] High-conf DeepFace ({df_gender_prob:.2f}) → trusting it")
            return "Woman" if df_is_woman else "Man"

        # If visual features have a strong opinion, weight them heavily
        if visual_gender == "Woman" and (df_gender_prob > 0.35 or deepface_conf < 0.5):
            print(f"  [FUSE] Visual=Woman + DeepFace uncertain → Woman")
            return "Woman"
        if visual_gender == "Man" and (df_gender_prob < 0.65 or deepface_conf < 0.5):
            print(f"  [FUSE] Visual=Man + DeepFace uncertain → Man")
            return "Man"

        # Visual disagrees with high-conf DeepFace → trust visual when face conf is low
        if visual_gender != "Unknown" and deepface_conf < 0.6:
            print(f"  [FUSE] Low face-conf, using visual: {visual_gender}")
            return visual_gender

        # Default: trust DeepFace
        return "Woman" if df_is_woman else "Man"

    def _probe_backend(self):
        """Test which DeepFace detector backend is available and working."""
        if not DEEPFACE_AVAILABLE:
            return "skip"
        # Create a small test image (blank face-sized)
        test_img = np.zeros((160, 160, 3), dtype=np.uint8) + 128
        for backend in self.FACE_BACKENDS:
            try:
                DeepFace.analyze(  # type: ignore[reportPossiblyUnboundVariable]
                    test_img,
                    actions=["gender"],
                    enforce_detection=False,
                    silent=True,
                    detector_backend=backend
                )
                print(f"[OK] Face-detection backend: {backend}")
                return backend
            except Exception:
                print(f"[WARN] Backend '{backend}' not available, trying next...")
                continue
        print("[WARN] No face-detection backend available, using 'skip'")
        return "skip"

    def _human_category_label(self, age, gender):
        """Return a clean display label: 'Man', 'Woman', or 'Child'."""
        try:
            age_int = int(age)
        except (ValueError, TypeError):
            age_int = None
        if age_int is not None and age_int < 18:
            return "Child"
        gender_str = str(gender).lower()
        if "woman" in gender_str or "female" in gender_str:
            return "Woman"
        elif "man" in gender_str or "male" in gender_str:
            return "Man"
        return "Person"

    def _get_human_rule(self, age, gender):
        try:
            age = int(age)
        except (ValueError, TypeError):
            return "unknown_human"
        if age < 18:
            return "child"
        gender_str = str(gender).lower()
        if "man" in gender_str or "male" in gender_str:
            return "adult_male"
        elif "woman" in gender_str or "female" in gender_str:
            return "adult_female"
        return "unknown_human"


# ─────────────────────────────────────────────
#  LOGGER
# ─────────────────────────────────────────────
class Logger:
    def __init__(self, filepath="detection_log.csv"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        if not os.path.exists(filepath):
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Type", "Label", "Action", "Confidence"])

    def log(self, det_type, label, action, conf):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, det_type, label, action, f"{conf:.2f}"])


# ─────────────────────────────────────────────
#  ALERT SYSTEM
# ─────────────────────────────────────────────
class AlertSystem:
    def __init__(self, whatsapp_number=""):
        self.whatsapp_number = whatsapp_number
        self.buzzer_active = False
        self._buzzer_thread = None

    def trigger_shock_alert(self, action_label):
        """Simulate shock deterrence (prints + popup in real system)"""
        print(f"[SHOCK TRIGGERED] {action_label}")

    def trigger_ultrasonic(self):
        """Play buzzer sound"""
        if not self.buzzer_active:
            self.buzzer_active = True
            self._buzzer_thread = threading.Thread(target=self._play_buzz, daemon=True)
            self._buzzer_thread.start()

    def stop_buzzer(self):
        self.buzzer_active = False

    def _play_buzz(self):
        played = False
        # Try pygame with audio file first
        if PYGAME_AVAILABLE:
            for audio_file in ("preview.mp3", "preview.wav", "buzzer.mp3", "buzzer.wav"):
                if os.path.exists(audio_file):
                    try:
                        pygame.mixer.music.load(audio_file)  # type: ignore[reportPossiblyUnboundVariable]
                        pygame.mixer.music.play()  # type: ignore[reportPossiblyUnboundVariable]
                        time.sleep(5)
                        played = True
                        break
                    except Exception as e:
                        print(f"[BUZZER] Audio file error: {e}")
        # Fallback: Windows system beep (always works)
        if not played:
            try:
                import winsound
                print("[BUZZER] Playing alert tone...")
                for freq in (1400, 1800, 2200, 1800, 1400):
                    winsound.Beep(freq, 400)
            except Exception:
                print("[BUZZER] Ultrasonic alert triggered (no audio device)")
                time.sleep(3)
        self.buzzer_active = False

    def send_whatsapp(self, message, image_path=None):
        if not self.whatsapp_number or not WHATSAPP_AVAILABLE:
            print(f"[WHATSAPP] {message}")
            return
        try:
            now = datetime.now()
            hour = now.hour
            minute = now.minute + 2
            if minute >= 60:
                minute -= 60
                hour = (hour + 1) % 24
            if image_path and os.path.exists(image_path):
                pywhatkit.sendwhats_image(self.whatsapp_number, image_path, message,  # type: ignore[reportPossiblyUnboundVariable, reportPrivateImportUsage]
                                          wait_time=15, tab_close=True)
            else:
                pywhatkit.sendwhatmsg(self.whatsapp_number, message,  # type: ignore[reportPossiblyUnboundVariable, reportPrivateImportUsage]
                                      hour, minute, tab_close=True)
        except Exception as e:
            print(f"[WHATSAPP ERROR] {e}")


# ─────────────────────────────────────────────
#  MAIN GUI APPLICATION
# ─────────────────────────────────────────────
class SmartFencingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SmartVision – Smart Fencing System")
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("1280x780")
        self.root.minsize(1100, 700)

        self.engine = DetectionEngine()
        self.logger = Logger("logs/detection_log.csv")
        self.alert = AlertSystem()

        self.cap = None
        self.running = False
        self.last_frame = None
        self.crop_mode = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI CONSTRUCTION ──────────────────────
    def _build_ui(self):
        # ── Top bar
        topbar = tk.Frame(self.root, bg="#16213e", height=50)
        topbar.pack(fill="x", side="top")

        tk.Label(topbar, text="🛡 SmartVision – Farm Security",
                 font=("Helvetica", 16, "bold"), bg="#16213e", fg="#e94560").pack(side="left", padx=15, pady=8)

        # Model selector
        tk.Label(topbar, text="YOLO Model:", bg="#16213e", fg="#aaa",
                 font=("Helvetica", 10)).pack(side="left", padx=(20,4))
        self.model_var = tk.StringVar(value="yolov8m.pt")
        model_cb = ttk.Combobox(topbar, textvariable=self.model_var, width=14,
                                values=["yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt","yolov8x.pt"])
        model_cb.pack(side="left", padx=4)
        model_cb.bind("<<ComboboxSelected>>", lambda e: self.engine.change_model(self.model_var.get()))

        self.log_indicator = tk.Label(topbar, text="● Logs Active", bg="#16213e",
                                      fg="#00ff88", font=("Helvetica", 10, "bold"))
        self.log_indicator.pack(side="right", padx=15)

        # ── Button bar
        btnbar = tk.Frame(self.root, bg="#0f3460", pady=6)
        btnbar.pack(fill="x")

        btn_style = {"bg": "#e94560", "fg": "white", "font": ("Helvetica", 10, "bold"),
                     "relief": "flat", "padx": 12, "pady": 5, "cursor": "hand2"}

        tk.Button(btnbar, text="📷 Laptop Camera",  command=self.start_webcam,  **btn_style).pack(side="left", padx=6)
        tk.Button(btnbar, text="📱 DroidCam",       command=self.start_droidcam, **btn_style).pack(side="left", padx=6)
        tk.Button(btnbar, text="🎬 Load Video",     command=self.load_video,     **btn_style).pack(side="left", padx=6)
        tk.Button(btnbar, text="🖼 Load Image",     command=self.load_image,     **btn_style).pack(side="left", padx=6)
        tk.Button(btnbar, text="■ Stop",            command=self.stop,
                  bg="#555", fg="white", font=("Helvetica", 10, "bold"),
                  relief="flat", padx=12, pady=5, cursor="hand2").pack(side="left", padx=6)
        tk.Button(btnbar, text="🔇 Stop Buzzer",    command=self._stop_buzzer,
                  bg="#f39c12", fg="white", font=("Helvetica", 10, "bold"),
                  relief="flat", padx=12, pady=5, cursor="hand2").pack(side="left", padx=6)
        tk.Button(btnbar, text="📋 Open Logs",      command=self._open_logs,
                  bg="#27ae60", fg="white", font=("Helvetica", 10, "bold"),
                  relief="flat", padx=12, pady=5, cursor="hand2").pack(side="left", padx=6)

        # ── Main content area
        content = tk.Frame(self.root, bg="#1a1a2e")
        content.pack(fill="both", expand=True, padx=10, pady=8)

        # Video canvas
        self.canvas = tk.Label(content, bg="#000", width=780, height=520,
                               text="No Video Source\nClick a button above to start",
                               font=("Helvetica", 14), fg="#555")
        self.canvas.pack(side="left", fill="both", expand=True)

        # ── Right info panels
        right = tk.Frame(content, bg="#1a1a2e", width=300)
        right.pack(side="right", fill="y", padx=(10,0))
        right.pack_propagate(False)

        self.human_box  = self._info_panel(right, "👤 HUMAN INFO",  "#27ae60")
        self.animal_box = self._info_panel(right, "🐾 ANIMAL INFO", "#e67e22")
        self.object_box = self._info_panel(right, "📦 OBJECT INFO", "#e94560")

        # ── Bottom bar (WhatsApp + Crop)
        bottom = tk.Frame(self.root, bg="#16213e", pady=6)
        bottom.pack(fill="x", side="bottom")

        tk.Label(bottom, text="WhatsApp #:", bg="#16213e", fg="#aaa",
                 font=("Helvetica", 10)).pack(side="left", padx=(10,4))
        self.wa_entry = tk.Entry(bottom, width=18, font=("Helvetica", 10),
                                 bg="#0f3460", fg="white", insertbackground="white")
        self.wa_entry.insert(0, "+91XXXXXXXXXX")
        self.wa_entry.pack(side="left", padx=4)

        tk.Button(bottom, text="✂ Crop & Send", command=self._crop_and_send,
                  bg="#8e44ad", fg="white", font=("Helvetica", 10, "bold"),
                  relief="flat", padx=12, pady=4, cursor="hand2").pack(side="left", padx=8)

        self.status_var = tk.StringVar(value="Ready – Select a source to begin.")
        tk.Label(bottom, textvariable=self.status_var, bg="#16213e",
                 fg="#aaa", font=("Helvetica", 9)).pack(side="right", padx=15)

    def _info_panel(self, parent, title, color):
        frame = tk.LabelFrame(parent, text=title, font=("Helvetica", 10, "bold"),
                              bg="#16213e", fg=color, bd=1, relief="solid",
                              labelanchor="n", padx=8, pady=6)
        frame.pack(fill="x", pady=6)
        label = tk.Label(frame, text="No detections", font=("Courier", 9),
                         bg="#16213e", fg="#ccc", justify="left", wraplength=270)
        label.pack(anchor="w")
        return label

    # ── VIDEO SOURCES ─────────────────────────
    def start_webcam(self):
        self._start_capture(0)

    def start_droidcam(self):
        url = simpledialog.askstring("DroidCam", "Enter DroidCam IP URL\n(e.g. http://192.168.1.5:4747/video)",
                                     parent=self.root)
        if url:
            self._start_capture(url)

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self._start_capture(path)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            frame = cv2.imread(path)
            if frame is not None:
                frame = cv2.resize(frame, (640, 480))
                self._process_single_frame(frame)

    def _start_capture(self, source):
        self.stop()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open source: {source}")
            return
        self.running = True
        self._set_status(f"▶ Running – {source}")
        t = threading.Thread(target=self._video_loop, daemon=True)
        t.start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self._set_status("⏹ Stopped.")

    # ── VIDEO LOOP ────────────────────────────
    def _video_loop(self):
        while self.running:
            if self.cap is None:
                break
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            frame = cv2.resize(frame, (640, 480))
            self._process_single_frame(frame)
            time.sleep(0.03)

    def _process_single_frame(self, frame):
        annotated, humans, animals, objects = self.engine.analyze_frame(frame)
        self.last_frame = annotated.copy()

        # Update GUI in main thread
        self.root.after(0, self._update_ui, annotated, humans, animals, objects)

        # Logging + alerts (background)
        threading.Thread(target=self._handle_detections,
                         args=(humans, animals, objects), daemon=True).start()

    def _update_ui(self, frame, humans, animals, objects):
        # Convert frame to Tkinter image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.configure(image=img, text="")
        self.canvas.image = img  # type: ignore[reportAttributeAccessIssue]

        # Human panel
        if humans:
            txt = "\n".join([f"• {h['label']}\n  ➤ {h['action']}" for h in humans])
        else:
            txt = "No humans detected"
        self.human_box.configure(text=txt)

        # Animal panel
        if animals:
            txt = "\n".join([f"• {a['label']}\n  ➤ {a['action']}" for a in animals])
        else:
            txt = "No animals detected"
        self.animal_box.configure(text=txt)

        # Object panel
        if objects:
            txt = "\n".join([f"• {o['label']}" for o in objects[:5]])
        else:
            txt = "No objects detected"
        self.object_box.configure(text=txt)

    def _handle_detections(self, humans, animals, objects):
        for h in humans:
            self.logger.log("Human", h["label"], h["action"], h["conf"])
            if h["shock"]:
                self.alert.trigger_shock_alert(h["action"])

        for a in animals:
            self.logger.log("Animal", a["label"], a["action"], a["conf"])
            if "Ultrasonic" in a["action"]:
                self.alert.trigger_ultrasonic()
            elif a["shock"]:
                self.alert.trigger_shock_alert(a["action"])

        for o in objects:
            self.logger.log("Object", o["label"], o["action"], o["conf"])

    # ── CROP & SEND ───────────────────────────
    def _crop_and_send(self):
        if self.last_frame is None:
            messagebox.showinfo("Info", "No frame available yet.")
            return
        CropWindow(self.root, self.last_frame, self._send_cropped)

    def _send_cropped(self, cropped_img):
        save_path = f"logs/alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        os.makedirs("logs", exist_ok=True)
        cv2.imwrite(save_path, cropped_img)
        self.alert.whatsapp_number = self.wa_entry.get().strip()
        msg = f"🚨 Detection Alert! ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n" \
              f"Check your farm – intruder detected."
        self.alert.send_whatsapp(msg, save_path)
        self._set_status(f"✅ Alert sent! Image saved: {save_path}")

    # ── HELPERS ───────────────────────────────
    def _stop_buzzer(self):
        self.alert.stop_buzzer()
        self._set_status("🔇 Buzzer stopped.")

    def _open_logs(self):
        log_path = os.path.abspath("logs/detection_log.csv")
        if os.path.exists(log_path):
            LogViewer(self.root, log_path)
        else:
            messagebox.showinfo("Logs", "No log file found yet.")

    def _set_status(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))

    def _on_close(self):
        self.stop()
        self.root.destroy()


# ─────────────────────────────────────────────
#  CROP WINDOW
# ─────────────────────────────────────────────
class CropWindow:
    def __init__(self, parent, frame, callback):
        self.callback = callback
        self.frame = frame
        self.start = None
        self.end = None

        self.win = tk.Toplevel(parent)
        self.win.title("Crop Frame – Click & Drag, then press Enter")
        self.win.configure(bg="#000")

        h, w = frame.shape[:2]
        self.canvas = tk.Canvas(self.win, width=w, height=h, cursor="crosshair")
        self.canvas.pack()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(0, 0, anchor="nw", image=self.img)

        self.rect_id = None
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.win.bind("<Return>", self._on_confirm)

        tk.Label(self.win, text="Draw selection → press Enter to send",
                 bg="#000", fg="#aaa", font=("Helvetica", 9)).pack(pady=4)

    def _on_press(self, e):
        self.start = (e.x, e.y)

    def _on_drag(self, e):
        if self.start is None:
            return
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start[0], self.start[1], e.x, e.y,
            outline="#e94560", width=2)

    def _on_release(self, e):
        self.end = (e.x, e.y)

    def _on_confirm(self, e):
        if self.start and self.end and self.frame is not None:
            x1, y1 = min(self.start[0], self.end[0]), min(self.start[1], self.end[1])
            x2, y2 = max(self.start[0], self.end[0]), max(self.start[1], self.end[1])
            cropped = self.frame[y1:y2, x1:x2]
            if cropped.size > 0:
                self.callback(cropped)
        self.win.destroy()


# ─────────────────────────────────────────────
#  LOG VIEWER
# ─────────────────────────────────────────────
class LogViewer:
    def __init__(self, parent, log_path):
        win = tk.Toplevel(parent)
        win.title("Detection Logs")
        win.geometry("860x500")
        win.configure(bg="#16213e")

        tk.Label(win, text="Detection Log", font=("Helvetica", 13, "bold"),
                 bg="#16213e", fg="#e94560").pack(pady=8)

        frame = tk.Frame(win, bg="#16213e")
        frame.pack(fill="both", expand=True, padx=10, pady=6)

        cols = ("Timestamp", "Type", "Label", "Action", "Confidence")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=20)
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=150)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True)

        with open(log_path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                tree.insert("", "end", values=row)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import tkinter.simpledialog
    root = tk.Tk()
    app = SmartFencingApp(root)
    root.mainloop()
