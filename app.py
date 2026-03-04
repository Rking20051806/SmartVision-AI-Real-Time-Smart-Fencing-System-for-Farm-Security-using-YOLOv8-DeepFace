"""
Smart Fencing System – Web Application (Gradio)
Real-Time Detection Based on Smart Fencing System for Farm Security
Author: Rohan Nandanwar
Deployable on Hugging Face Spaces / Render / Any cloud platform
"""

import gradio as gr  # pyright: ignore
import cv2
import csv
import os
import sys
import traceback
import numpy as np
from datetime import datetime
from PIL import Image

# ─── Torch Compatibility Fix (PyTorch 2.6+ weights_only) ──
try:
    import torch
    if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
        # PyTorch 2.6+ defaults weights_only=True which blocks YOLO loading
        try:
            from ultralytics.nn.tasks import DetectionModel, SegmentationModel, ClassificationModel
            torch.serialization.add_safe_globals([DetectionModel, SegmentationModel, ClassificationModel])
            print("[INIT] Patched torch safe_globals for ultralytics")
        except Exception:
            # Fallback: set default to weights_only=False
            _original_load = torch.load
            def _patched_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return _original_load(*args, **kwargs)
            torch.load = _patched_load  # type: ignore
            print("[INIT] Patched torch.load with weights_only=False")
except Exception as e:
    print(f"[WARN] Torch compatibility patch failed: {e}")

# ─── ML Imports ───────────────────────────────
LOAD_ERRORS = []

try:
    from ultralytics import YOLO  # pyright: ignore
    YOLO_AVAILABLE = True
except Exception as e:
    YOLO_AVAILABLE = False
    LOAD_ERRORS.append(f"ultralytics: {e}")

try:
    from deepface import DeepFace  # pyright: ignore
    DEEPFACE_AVAILABLE = True
except Exception as e:
    DEEPFACE_AVAILABLE = False
    LOAD_ERRORS.append(f"deepface: {e}")

try:
    import tensorflow as tf  # pyright: ignore
    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False
    LOAD_ERRORS.append(f"tensorflow: {e}")

print(f"[INIT] YOLO={YOLO_AVAILABLE}, DeepFace={DEEPFACE_AVAILABLE}, TF={TF_AVAILABLE}")
if LOAD_ERRORS:
    print(f"[INIT] Load errors: {LOAD_ERRORS}")

# ─────────────────────────────────────────────
#  SHOCK RULES TABLE
# ─────────────────────────────────────────────
SHOCK_RULES = {
    "adult_male":    {"shock": True,  "current_uA": 4000, "action": "4000 uA Shock Deterrence",   "buzzer": True},
    "adult_female":  {"shock": True,  "current_uA": 2500, "action": "2500 uA Shock Deterrence",   "buzzer": True},
    "child":         {"shock": False, "current_uA": 0,    "action": "No Shock (Safety Override)",  "buzzer": True},
    "unknown_human": {"shock": False, "current_uA": 0,    "action": "Alert Only",                  "buzzer": True},
    "cow":       {"shock": True,  "current_uA": 2500, "action": "2500 uA Shock Deterrence", "buzzer": True},
    "dog":       {"shock": True,  "current_uA": 1800, "action": "1800 uA Shock Deterrence", "buzzer": True},
    "horse":     {"shock": True,  "current_uA": 2500, "action": "2500 uA Shock Deterrence", "buzzer": True},
    "sheep":     {"shock": True,  "current_uA": 2000, "action": "2000 uA Shock Deterrence", "buzzer": True},
    "elephant":  {"shock": True,  "current_uA": 4000, "action": "4000 uA Shock Deterrence", "buzzer": True},
    "bear":      {"shock": True,  "current_uA": 4000, "action": "4000 uA Shock Deterrence", "buzzer": True},
    "zebra":     {"shock": True,  "current_uA": 2500, "action": "2500 uA Shock Deterrence", "buzzer": True},
    "giraffe":   {"shock": True,  "current_uA": 4000, "action": "4000 uA Shock Deterrence", "buzzer": True},
    "cat":       {"shock": False, "current_uA": 1500, "action": "1500 uA Shock Deterrence", "buzzer": True},
    "bird":      {"shock": False, "current_uA": 0,    "action": "Ultrasonic / Buzzer",      "buzzer": True},
    "chicken":   {"shock": False, "current_uA": 0,    "action": "Ultrasonic / Buzzer",      "buzzer": True},
    "squirrel":  {"shock": False, "current_uA": 0,    "action": "Ultrasonic / Buzzer",      "buzzer": True},
    "butterfly": {"shock": False, "current_uA": 0,    "action": "No Action",                "buzzer": False},
    "spider":    {"shock": False, "current_uA": 0,    "action": "No Action",                "buzzer": False},
}

ANIMAL_MAP = {
    "bird": "bird", "cat": "cat", "dog": "dog", "horse": "horse",
    "sheep": "sheep", "cow": "cow", "elephant": "elephant", "bear": "bear",
    "zebra": "zebra", "giraffe": "giraffe"
}


# ─────────────────────────────────────────────
#  DETECTION ENGINE
# ─────────────────────────────────────────────
class DetectionEngine:
    FACE_BACKENDS = ["retinaface", "mtcnn", "opencv", "ssd"]

    def __init__(self, model_name="yolov8n.pt"):
        self.model = None
        self.animal_cnn = None
        self.model_name = model_name
        self.frame_count = 0
        self._last_human_info = {}
        self._best_backend = "skip"
        self.status_messages = []
        self._load_models()

    def _load_models(self):
        self.status_messages = []
        # --- YOLO ---
        if YOLO_AVAILABLE:
            model_path = self.model_name
            if os.path.exists(model_path):
                self.status_messages.append(f"YOLO model file found: {model_path}")
            else:
                self.status_messages.append(f"YOLO model not found locally: {model_path} (will auto-download)")
            try:
                self.model = YOLO(model_path)  # pyright: ignore
                self.status_messages.append(f"YOLO model loaded OK: {model_path}")
            except Exception as e:
                self.status_messages.append(f"YOLO load FAILED: {e}")
                self.model = None
        else:
            self.status_messages.append("ultralytics not installed - YOLO unavailable")

        # --- Animal CNN ---
        if TF_AVAILABLE and os.path.exists("animal10.h5"):
            try:
                self.animal_cnn = tf.keras.models.load_model("animal10.h5")  # pyright: ignore
                self.status_messages.append("Animal-10 CNN loaded")
            except Exception as e:
                self.status_messages.append(f"Animal CNN load failed: {e}")
        else:
            self.status_messages.append("Animal CNN (animal10.h5) not available - using YOLO only")

        # --- DeepFace backend ---
        if DEEPFACE_AVAILABLE:
            self._best_backend = self._probe_backend()
            self.status_messages.append(f"DeepFace available, backend: {self._best_backend}")
        else:
            self.status_messages.append("DeepFace not installed - gender/age unavailable")
            self._best_backend = "skip"

    def change_model(self, model_name):
        self.model_name = model_name
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_name)  # pyright: ignore
                return f"Switched to: {model_name}"
            except Exception as e:
                return f"Failed to switch: {e}"
        return "YOLO not available"

    def get_status(self):
        return "\n".join(self.status_messages) if self.status_messages else "No status info"

    def analyze_frame(self, frame):
        """Returns (annotated_frame, human_detections, animal_detections, object_detections, alerts)"""
        if self.model is None:
            return frame, [], [], [], ["YOLO model not loaded. Detection unavailable."]

        self.frame_count += 1
        annotated = frame.copy()
        humans, animals, objects, alerts = [], [], [], []

        try:
            results = self.model(frame, verbose=False, conf=0.25)[0]
        except Exception as e:
            return frame, [], [], [], [f"YOLO inference error: {e}"]

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
                    "current_uA": rule.get("current_uA", 0),
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2),
                    "type": "Human"
                }
                humans.append(info)

                # Draw on frame
                color = (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                overlay = f"{category}, Age:{age} ({conf:.2f})"
                cv2.putText(annotated, overlay, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Alerts
                if rule.get("buzzer"):
                    alerts.append(f"BUZZER TRIGGERED: {category} detected - {rule['action']}")
                if rule["shock"]:
                    alerts.append(f"SHOCK ARMED: {rule['current_uA']} uA for {category}")
                alerts.append(f"WhatsApp Alert Sent: {category} (Age:{age}) at fence perimeter")

            elif cls_name in ANIMAL_MAP:
                species = ANIMAL_MAP[cls_name]
                rule = SHOCK_RULES.get(species, {"action": "Alert", "shock": False, "buzzer": True})
                info = {
                    "label": f"{species.capitalize()} | Conf:{conf:.2f}",
                    "action": rule["action"],
                    "shock": rule["shock"],
                    "current_uA": rule.get("current_uA", 0),
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2),
                    "type": "Animal"
                }
                animals.append(info)

                color = (0, 165, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{species} {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Alerts
                if rule.get("buzzer"):
                    alerts.append(f"BUZZER TRIGGERED: {species.capitalize()} detected - {rule['action']}")
                if rule.get("shock"):
                    alerts.append(f"SHOCK ARMED: {rule.get('current_uA', 0)} uA for {species.capitalize()}")
                alerts.append(f"WhatsApp Alert Sent: {species.capitalize()} detected near fence")

            else:
                info = {
                    "label": f"{cls_name.capitalize()} | Conf:{conf:.2f}",
                    "action": "Logged",
                    "shock": False,
                    "current_uA": 0,
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2),
                    "type": "Object"
                }
                objects.append(info)
                color = (255, 0, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{cls_name} {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if not humans and not animals and not objects:
            alerts.append("No objects detected in this frame. Try a clearer image.")

        return annotated, humans, animals, objects, alerts

    # ── Human Analysis ────────────────────────
    def _analyze_human(self, frame, x1, y1, x2, y2):
        if not DEEPFACE_AVAILABLE:
            visual_gender = self._visual_gender_features(frame, x1, y1, x2, y2)
            return "?", visual_gender

        try:
            h = y2 - y1
            w = x2 - x1
            roi = frame[max(0, y1):y2, max(0, x1):x2]
            if roi.size == 0:
                return "?", "Unknown"

            upper_h = max(h // 2, 60)
            upper = frame[max(0, y1):min(frame.shape[0], max(0, y1) + upper_h),
                          max(0, x1):min(frame.shape[1], x2)]

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

            backends_to_try = [self._best_backend] + [
                b for b in self.FACE_BACKENDS if b != self._best_backend
            ] + ["skip"]

            for crop in crops:
                for backend in backends_to_try:
                    try:
                        result = DeepFace.analyze(  # pyright: ignore
                            crop, actions=["age", "gender"],
                            enforce_detection=False, silent=True,
                            detector_backend=backend
                        )
                        if isinstance(result, list):
                            result = result[0]
                        result_dict: dict = dict(result) if not isinstance(result, dict) else result  # pyright: ignore
                        age_val = result_dict.get("age", None)
                        gender_val = result_dict.get("dominant_gender", None)
                        face_conf = result_dict.get("face_confidence", 0)
                        g_scores = result_dict.get("gender", {})
                        if age_val is not None and gender_val is not None:
                            deepface_age = age_val
                            deepface_gender = gender_val
                            deepface_conf = float(face_conf) if face_conf else 0.0
                            gender_scores = g_scores
                            if deepface_conf > 0.8:
                                break
                    except Exception:
                        continue
                if deepface_age is not None and deepface_conf > 0.8:
                    break

            visual_gender = self._visual_gender_features(frame, x1, y1, x2, y2)
            final_gender = self._fuse_gender(
                deepface_gender, deepface_conf, gender_scores, visual_gender
            )
            final_age = deepface_age if deepface_age is not None else "?"
            self._last_human_info = {"age": final_age, "gender": final_gender}
            return final_age, final_gender

        except Exception as e:
            print(f"[WARN] Human analysis error: {e}")
            return "?", "Unknown"

    # ── Visual Gender Features ────────────────
    def _visual_gender_features(self, frame, x1, y1, x2, y2):
        try:
            h = y2 - y1
            w = x2 - x1
            roi = frame[max(0, y1):y2, max(0, x1):x2]
            if roi.size == 0:
                return "Unknown"

            score = 0.0

            head_region = roi[0:int(h * 0.4), :]
            if head_region.size > 0:
                gray_head = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
                hair_mask = (gray_head < 80).astype(np.uint8)
                head_h, head_w = gray_head.shape[:2]
                if head_h > 10 and head_w > 10:
                    top_hair = hair_mask[:head_h // 2, :]
                    bottom_hair = hair_mask[head_h // 2:, :]
                    top_ratio = np.sum(top_hair) / max(top_hair.size, 1)
                    bottom_ratio = np.sum(bottom_hair) / max(bottom_hair.size, 1)
                    if top_ratio > 0.15 and bottom_ratio > 0.12:
                        hair_extent = bottom_ratio / max(top_ratio, 0.01)
                        if hair_extent > 0.4:
                            score += 1.5
                    hair_cols = np.sum(hair_mask, axis=0)
                    hair_spread = np.sum(hair_cols > 0) / max(head_w, 1)
                    if hair_spread > 0.6:
                        score += 0.8

            if h > 40 and w > 20:
                shoulder_strip = roi[int(h * 0.2):int(h * 0.3), :]
                hip_strip = roi[int(h * 0.55):int(h * 0.7), :]
                if shoulder_strip.size > 0 and hip_strip.size > 0:
                    gray_s = cv2.cvtColor(shoulder_strip, cv2.COLOR_BGR2GRAY)
                    gray_h = cv2.cvtColor(hip_strip, cv2.COLOR_BGR2GRAY)
                    edges_s = cv2.Canny(gray_s, 50, 150)
                    edges_h = cv2.Canny(gray_h, 50, 150)
                    s_width = np.sum(np.sum(edges_s, axis=0) > 0)
                    h_width = np.sum(np.sum(edges_h, axis=0) > 0)
                    if s_width > 5:
                        ratio = h_width / max(s_width, 1)
                        if ratio > 1.15:
                            score += 1.0
                        elif ratio < 0.85:
                            score -= 0.8

            if roi.size > 0:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                clothing = hsv[int(h * 0.25):int(h * 0.7), :]
                if clothing.size > 0:
                    warm_mask = ((clothing[:, :, 0] < 15) | (clothing[:, :, 0] > 155)) & \
                                (clothing[:, :, 1] > 50)
                    warm_ratio = np.sum(warm_mask) / max(warm_mask.size, 1)
                    if warm_ratio > 0.1:
                        score += 0.5

            upper_body = roi[0:int(h * 0.4), :]
            if upper_body.size > 0:
                gray_ub = cv2.cvtColor(upper_body, cv2.COLOR_BGR2GRAY)
                hsv_ub = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
                skin_mask = (hsv_ub[:, :, 0] >= 0) & (hsv_ub[:, :, 0] <= 25) & \
                            (hsv_ub[:, :, 1] >= 30) & (hsv_ub[:, :, 2] >= 60)
                if np.sum(skin_mask) > 100:
                    skin_gray = gray_ub.copy()
                    skin_gray[~skin_mask] = 0
                    lap_var = cv2.Laplacian(skin_gray, cv2.CV_64F).var()
                    if lap_var < 500:
                        score += 0.4
                    elif lap_var > 1500:
                        score -= 0.4

            if score >= 1.5:
                return "Woman"
            elif score <= -1.0:
                return "Man"
            return "Unknown"
        except Exception:
            return "Unknown"

    def _fuse_gender(self, deepface_gender, deepface_conf, gender_scores, visual_gender):
        if deepface_gender is None and visual_gender == "Unknown":
            return "Unknown"
        if deepface_gender is None:
            return visual_gender

        df_str = str(deepface_gender).lower()
        df_is_woman = "woman" in df_str or "female" in df_str
        df_gender_prob = 0.5
        if isinstance(gender_scores, dict):
            woman_prob = gender_scores.get("Woman", gender_scores.get("Female", 50))
            df_gender_prob = float(woman_prob) / 100.0

        if deepface_conf > 0.8 and (df_gender_prob > 0.75 or df_gender_prob < 0.25):
            return "Woman" if df_is_woman else "Man"
        if visual_gender == "Woman" and (df_gender_prob > 0.35 or deepface_conf < 0.5):
            return "Woman"
        if visual_gender == "Man" and (df_gender_prob < 0.65 or deepface_conf < 0.5):
            return "Man"
        if visual_gender != "Unknown" and deepface_conf < 0.6:
            return visual_gender
        return "Woman" if df_is_woman else "Man"

    def _probe_backend(self):
        if not DEEPFACE_AVAILABLE:
            return "skip"
        test_img = np.zeros((160, 160, 3), dtype=np.uint8) + 128
        for backend in self.FACE_BACKENDS:
            try:
                DeepFace.analyze(test_img, actions=["gender"],  # pyright: ignore
                                 enforce_detection=False, silent=True,
                                 detector_backend=backend)
                print(f"[OK] Face-detection backend: {backend}")
                return backend
            except Exception as e:
                print(f"[WARN] Backend {backend} failed: {e}")
                continue
        print("[WARN] All face backends failed, using 'skip'")
        return "skip"

    def _human_category_label(self, age, gender):
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
        d = os.path.dirname(filepath)
        if d:
            os.makedirs(d, exist_ok=True)
        if not os.path.exists(filepath):
            with open(filepath, "w", newline="") as f:
                csv.writer(f).writerow(["Timestamp", "Type", "Label", "Action", "Confidence"])

    def log(self, det_type, label, action, conf):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.filepath, "a", newline="") as f:
                csv.writer(f).writerow([ts, det_type, label, action, f"{conf:.2f}"])
        except Exception:
            pass


# ─────────────────────────────────────────────
#  GLOBAL ENGINE & LOGGER
# ─────────────────────────────────────────────
AVAILABLE_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
engine = DetectionEngine("yolov8n.pt")
logger = Logger("detection_log.csv")

print(f"[INIT] Engine status:\n{engine.get_status()}")


# ─────────────────────────────────────────────
#  PROCESSING FUNCTIONS
# ─────────────────────────────────────────────
def format_detections(humans, animals, objects, alerts):
    """Build a formatted Markdown report of all detections."""
    lines = []

    # Alerts Panel
    if alerts:
        lines.append("### Alerts & Notifications\n")
        for a in alerts:
            lines.append(f"- {a}")
        lines.append("")

    # Detections
    if humans:
        lines.append("### Human Detections\n")
        for h in humans:
            shock_text = "SHOCK" if h["shock"] else "SAFE"
            lines.append(f"- **{h['label']}** ({h['conf']:.0%})")
            lines.append(f"  - [{shock_text}] {h['action']}")
            if h.get("current_uA"):
                lines.append(f"  - Current: {h['current_uA']} uA")
        lines.append("")

    if animals:
        lines.append("### Animal Detections\n")
        for a in animals:
            shock_text = "SHOCK" if a["shock"] else "ALERT"
            lines.append(f"- **{a['label']}**")
            lines.append(f"  - [{shock_text}] {a['action']}")
            if a.get("current_uA"):
                lines.append(f"  - Current: {a['current_uA']} uA")
        lines.append("")

    if objects:
        lines.append("### Other Objects\n")
        for o in objects[:10]:
            lines.append(f"- {o['label']}")
        lines.append("")

    if not humans and not animals and not objects:
        lines.append("*No detections in this frame.*\n")

    # Summary
    total = len(humans) + len(animals) + len(objects)
    shock_count = sum(1 for d in humans + animals if d.get("shock"))
    buzzer_count = len([a for a in alerts if "BUZZER" in a])
    whatsapp_count = len([a for a in alerts if "WhatsApp" in a])
    lines.append(f"\n---\n**Total: {total}** detections | **{shock_count}** shock triggers | "
                 f"**{buzzer_count}** buzzer alerts | **{whatsapp_count}** notifications")

    return "\n".join(lines)


def process_image(image, model_name):
    """Process a single uploaded image."""
    if image is None:
        return None, "**Upload an image to begin detection.**"

    if model_name and model_name != engine.model_name:
        msg = engine.change_model(model_name)
        print(f"[MODEL] {msg}")

    try:
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            frame = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            frame = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        h, w = frame.shape[:2]
        scale = min(640 / w, 640 / h, 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        annotated, humans, animals, objects, alerts = engine.analyze_frame(frame)

        for det in humans:
            logger.log("Human", det["label"], det["action"], det["conf"])
        for det in animals:
            logger.log("Animal", det["label"], det["action"], det["conf"])
        for det in objects:
            logger.log("Object", det["label"], det["action"], det["conf"])

        result_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        report = format_detections(humans, animals, objects, alerts)

        return Image.fromarray(result_img), report

    except Exception as e:
        error_msg = f"**Error processing image:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return None, error_msg


def process_webcam(frame, model_name):
    """Process a webcam frame (streaming)."""
    if frame is None:
        return None, "*Waiting for webcam...*"

    if model_name and model_name != engine.model_name:
        engine.change_model(model_name)

    try:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        scale = min(640 / w, 640 / h, 1.0)
        if scale < 1.0:
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))

        annotated, humans, animals, objects, alerts = engine.analyze_frame(bgr)

        for det in humans:
            logger.log("Human", det["label"], det["action"], det["conf"])
        for det in animals:
            logger.log("Animal", det["label"], det["action"], det["conf"])

        result_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        report = format_detections(humans, animals, objects, alerts)

        return result_img, report

    except Exception as e:
        return frame, f"Error: {e}"


def process_video(video_path, model_name):
    """Process an uploaded video file."""
    if video_path is None:
        return None, "**Upload a video to begin.**"

    if model_name and model_name != engine.model_name:
        engine.change_model(model_name)

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "**Error:** Could not open video file."

        all_humans, all_animals, all_objects, all_alerts = [], [], [], []
        result_frame = None
        frame_idx = 0
        max_frames = 90

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            scale = min(640 / w, 640 / h, 1.0)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            if frame_idx % 3 == 0:
                annotated, humans, animals, objects, alerts = engine.analyze_frame(frame)
                all_humans.extend(humans)
                all_animals.extend(animals)
                all_objects.extend(objects)
                all_alerts.extend(alerts)
                if result_frame is None and (humans or animals):
                    result_frame = annotated.copy()
                elif result_frame is None:
                    result_frame = annotated.copy()
                for det in humans:
                    logger.log("Human", det["label"], det["action"], det["conf"])
                for det in animals:
                    logger.log("Animal", det["label"], det["action"], det["conf"])

            frame_idx += 1

        cap.release()

        if result_frame is not None:
            result_img = Image.fromarray(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
        else:
            result_img = None

        unique_alerts = list(dict.fromkeys(all_alerts))[:20]
        report = format_detections(all_humans, all_animals, all_objects, unique_alerts)
        report = f"**Analyzed {frame_idx} frames ({frame_idx // 3} processed)**\n\n" + report
        return result_img, report

    except Exception as e:
        return None, f"**Error:** {e}\n\n```\n{traceback.format_exc()}\n```"


def get_log_file():
    """Return the CSV log file for download."""
    if os.path.exists("detection_log.csv"):
        return "detection_log.csv"
    return None


def get_shock_rules_table():
    """Show shock rules as Markdown table."""
    lines = ["| Entity | Shock? | Current (uA) | Buzzer? | Action |",
             "|--------|--------|-------------|---------|--------|"]
    for key, rule in SHOCK_RULES.items():
        name = key.replace("_", " ").title()
        shock = "Yes" if rule.get("shock") else "No"
        buzzer = "Yes" if rule.get("buzzer") else "No"
        current = rule.get("current_uA", 0)
        lines.append(f"| {name} | {shock} | {current} | {buzzer} | {rule['action']} |")
    return "\n".join(lines)


def get_system_status():
    """Return system diagnostic info."""
    lines = ["### System Status\n"]
    lines.append(f"- **Python:** {sys.version.split()[0]}")
    lines.append(f"- **YOLO (ultralytics):** {'Available' if YOLO_AVAILABLE else 'Not installed'}")
    lines.append(f"- **DeepFace:** {'Available' if DEEPFACE_AVAILABLE else 'Not installed'}")
    lines.append(f"- **TensorFlow:** {'Available' if TF_AVAILABLE else 'Not installed'}")
    lines.append(f"- **Current Model:** {engine.model_name}")
    lines.append(f"- **Face Backend:** {engine._best_backend}")
    lines.append("")
    lines.append("### Model Load Log\n")
    for msg in engine.status_messages:
        lines.append(f"- {msg}")
    if LOAD_ERRORS:
        lines.append("\n### Import Errors\n")
        for e in LOAD_ERRORS:
            lines.append(f"- {e}")

    lines.append("\n### Model Files\n")
    for m in AVAILABLE_MODELS:
        exists = "Found" if os.path.exists(m) else "Missing"
        lines.append(f"- {m}: {exists}")
    lines.append(f"- animal10.h5: {'Found' if os.path.exists('animal10.h5') else 'Not available'}")

    lines.append(f"\n### Working Directory: `{os.getcwd()}`")
    try:
        files = os.listdir(".")
        lines.append(f"- Files: {', '.join(sorted(files)[:20])}")
    except Exception:
        pass

    return "\n".join(lines)


# ─────────────────────────────────────────────
#  GRADIO UI
# ─────────────────────────────────────────────
TITLE = "Smart Fencing System"
DESCRIPTION = """
**Real-Time Detection Based on Smart Fencing System for Farm Security**

Upload an image, video, or use your webcam to detect humans (with age and gender) and animals.
The system determines the appropriate shock deterrence and triggers buzzer / WhatsApp alerts.

*By Rohan Nandanwar - B.Tech Minor Project, DMIHER (DU)*
"""

with gr.Blocks(
    title="Smart Fencing System",
    theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue"),  # pyright: ignore
    css="""
        .main-header { text-align: center; margin-bottom: 10px; }
        .detection-report { min-height: 200px; }
    """
) as demo:

    gr.Markdown(f"# {TITLE}", elem_classes="main-header")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        model_selector = gr.Dropdown(
            choices=AVAILABLE_MODELS,
            value="yolov8n.pt",
            label="YOLO Model",
            info="Larger models are more accurate but slower on CPU"
        )

    with gr.Tabs():
        # Tab 1: Image Upload
        with gr.TabItem("Image Upload"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="pil", label="Upload Image")
                    img_btn = gr.Button("Detect", variant="primary", size="lg")
                with gr.Column(scale=1):
                    img_output = gr.Image(label="Detection Result")
            img_report = gr.Markdown(
                value="*Upload an image and click Detect to start.*",
                label="Detection Report",
                elem_classes="detection-report"
            )

            img_btn.click(
                fn=process_image,
                inputs=[img_input, model_selector],
                outputs=[img_output, img_report]
            )

        # Tab 2: Webcam (Live)
        with gr.TabItem("Webcam (Live)"):
            gr.Markdown("*Click to start your webcam. Detections update per frame.*")
            with gr.Row():
                with gr.Column(scale=1):
                    webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Webcam Feed")
                with gr.Column(scale=1):
                    webcam_output = gr.Image(label="Detection Result")
            webcam_report = gr.Markdown(elem_classes="detection-report")

            webcam_input.stream(
                fn=process_webcam,
                inputs=[webcam_input, model_selector],
                outputs=[webcam_output, webcam_report]
            )

        # Tab 3: Video Upload
        with gr.TabItem("Video Upload"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid_input = gr.Video(label="Upload Video")
                    vid_btn = gr.Button("Analyze Video", variant="primary", size="lg")
                with gr.Column(scale=1):
                    vid_output = gr.Image(label="Detection Sample Frame")
            vid_report = gr.Markdown(elem_classes="detection-report")

            vid_btn.click(
                fn=process_video,
                inputs=[vid_input, model_selector],
                outputs=[vid_output, vid_report]
            )

        # Tab 4: Shock Rules
        with gr.TabItem("Shock Rules"):
            gr.Markdown("### Smart Fencing Shock Rules Table")
            gr.Markdown(get_shock_rules_table())
            gr.Markdown("""
> **Note:** In the web demo, buzzer alerts and WhatsApp notifications are **simulated** 
> (shown as text alerts in the report). In the desktop app (main.py), these trigger actual 
> hardware buzzer sounds and WhatsApp messages via the connected device.
            """)

        # Tab 5: Detection Log
        with gr.TabItem("Detection Log"):
            gr.Markdown("### Detection History")
            log_btn = gr.Button("Download Detection Log", variant="secondary")
            log_file = gr.File(label="Log File")
            log_btn.click(fn=get_log_file, outputs=[log_file])

        # Tab 6: System Status
        with gr.TabItem("System Status"):
            gr.Markdown("### Diagnostics and Model Status")
            status_btn = gr.Button("Refresh Status", variant="secondary")
            status_display = gr.Markdown(value=get_system_status())
            status_btn.click(fn=get_system_status, outputs=[status_display])

    gr.Markdown("---\n*Smart Fencing System v2.0 - Farm Security Through Intelligent Detection*")


# ─────────────────────────────────────────────
#  LAUNCH
# ─────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
