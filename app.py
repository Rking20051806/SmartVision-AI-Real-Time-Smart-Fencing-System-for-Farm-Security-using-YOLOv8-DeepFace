"""
Smart Fencing System – Web Application (Gradio)
Real-Time Detection Based on Smart Fencing System for Farm Security
Author: Rohan Nandanwar
Deployable on Hugging Face Spaces / Render / Any cloud platform
"""

import gradio as gr
import cv2
import csv
import os
import numpy as np
from datetime import datetime
from PIL import Image

# ─── ML Imports ───────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not installed")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("[WARNING] deepface not installed")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ─────────────────────────────────────────────
#  SHOCK RULES TABLE
# ─────────────────────────────────────────────
SHOCK_RULES = {
    "adult_male":    {"shock": True,  "current_uA": 4000,  "action": "4000 µA Shock Deterrence"},
    "adult_female":  {"shock": True,  "current_uA": 2500,  "action": "2500 µA Shock Deterrence"},
    "child":         {"shock": False, "current_uA": 0,     "action": "No Shock (Safety Override)"},
    "unknown_human": {"shock": False, "current_uA": 0,     "action": "Alert Only"},
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

ANIMAL_MAP = {
    "bird": "bird", "cat": "cat", "dog": "dog", "horse": "horse",
    "sheep": "sheep", "cow": "cow", "elephant": "elephant", "bear": "bear",
    "zebra": "zebra", "giraffe": "giraffe"
}


# ─────────────────────────────────────────────
#  DETECTION ENGINE (same logic as main.py)
# ─────────────────────────────────────────────
class DetectionEngine:
    FACE_BACKENDS = ["retinaface", "mtcnn", "opencv", "ssd"]

    def __init__(self, model_name="yolov8n.pt"):
        self.model = None
        self.animal_cnn = None
        self.model_name = model_name
        self.frame_count = 0
        self.deepface_skip = 1  # analyze every frame for image mode
        self._last_human_info = {}
        self._best_backend = "retinaface"
        self._load_models()

    def _load_models(self):
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(self.model_name)  # pyright: ignore
                print(f"[OK] YOLO model loaded: {self.model_name}")
            except Exception as e:
                print(f"[ERROR] YOLO load failed: {e}")
        if TF_AVAILABLE and os.path.exists("animal10.h5"):
            try:
                self.animal_cnn = tf.keras.models.load_model("animal10.h5")  # pyright: ignore
                print("[OK] Animal-10 CNN loaded.")
            except Exception as e:
                print(f"[WARN] Animal CNN load failed: {e}")
        self._best_backend = self._probe_backend()

    def change_model(self, model_name):
        self.model_name = model_name
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_name)  # pyright: ignore
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
                    "current_uA": rule.get("current_uA", 0),
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2)
                }
                humans.append(info)
                color = (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                overlay = f"{category}, Age:{age} ({conf:.2f})"
                cv2.putText(annotated, overlay, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            elif cls_name in ANIMAL_MAP:
                species = ANIMAL_MAP[cls_name]
                rule = SHOCK_RULES.get(species, {"action": "Alert", "shock": False})
                info = {
                    "label": f"{species.capitalize()} | Conf:{conf:.2f}",
                    "action": rule["action"],
                    "shock": rule["shock"],
                    "current_uA": rule.get("current_uA", 0),
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
                    "current_uA": 0,
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2)
                }
                objects.append(info)
                color = (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{cls_name} {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        return annotated, humans, animals, objects

    # ── Human Analysis ────────────────────────
    def _analyze_human(self, frame, x1, y1, x2, y2):
        if not DEEPFACE_AVAILABLE:
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
                        age = result.get("age", None)  # pyright: ignore
                        gender = result.get("dominant_gender", None)  # pyright: ignore
                        face_conf = result.get("face_confidence", 0)  # pyright: ignore
                        gender_scores = result.get("gender", {})  # pyright: ignore
                        if age is not None and gender is not None:
                            deepface_age = age
                            deepface_gender = gender
                            deepface_conf = float(face_conf) if face_conf else 0.0
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

        except Exception:
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

            # 1) Hair length detection
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

            # 2) Body shape
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

            # 3) Clothing color
            if roi.size > 0:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                clothing = hsv[int(h * 0.25):int(h * 0.7), :]
                if clothing.size > 0:
                    warm_mask = ((clothing[:, :, 0] < 15) | (clothing[:, :, 0] > 155)) & \
                                (clothing[:, :, 1] > 50)
                    warm_ratio = np.sum(warm_mask) / max(warm_mask.size, 1)
                    if warm_ratio > 0.1:
                        score += 0.5

            # 4) Skin smoothness
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
            except Exception:
                continue
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
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        if not os.path.exists(filepath):
            with open(filepath, "w", newline="") as f:
                csv.writer(f).writerow(["Timestamp", "Type", "Label", "Action", "Confidence"])

    def log(self, det_type, label, action, conf):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.filepath, "a", newline="") as f:
            csv.writer(f).writerow([ts, det_type, label, action, f"{conf:.2f}"])


# ─────────────────────────────────────────────
#  GLOBAL ENGINE & LOGGER
# ─────────────────────────────────────────────
AVAILABLE_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
engine = DetectionEngine("yolov8n.pt")
logger = Logger("detection_log.csv")


# ─────────────────────────────────────────────
#  PROCESSING FUNCTIONS
# ─────────────────────────────────────────────
def format_detections(humans, animals, objects):
    """Build a formatted Markdown report of all detections."""
    lines = []

    if humans:
        lines.append("### 🧑 Human Detections\n")
        for h in humans:
            shock_icon = "⚡" if h["shock"] else "✅"
            lines.append(f"- **{h['label']}** ({h['conf']:.0%})")
            lines.append(f"  - {shock_icon} {h['action']}")
            if h.get("current_uA"):
                lines.append(f"  - Current: {h['current_uA']} µA")
        lines.append("")

    if animals:
        lines.append("### 🐾 Animal Detections\n")
        for a in animals:
            shock_icon = "⚡" if a["shock"] else "🔔"
            lines.append(f"- **{a['label']}**")
            lines.append(f"  - {shock_icon} {a['action']}")
        lines.append("")

    if objects:
        lines.append("### 📦 Other Objects\n")
        for o in objects[:8]:
            lines.append(f"- {o['label']}")
        lines.append("")

    if not humans and not animals and not objects:
        lines.append("*No detections in this frame.*")

    # Summary stats
    total = len(humans) + len(animals) + len(objects)
    shock_count = sum(1 for d in humans + animals if d.get("shock"))
    lines.append(f"\n---\n**Total: {total}** detections | **{shock_count}** shock triggers")

    return "\n".join(lines)


def process_image(image, model_name):
    """Process a single uploaded image."""
    if image is None:
        return None, "*Upload an image to begin detection.*"

    # Switch model if changed
    if model_name and model_name != engine.model_name:
        engine.change_model(model_name)

    # Convert PIL → OpenCV BGR
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (640, 480))

    # Run detection
    annotated, humans, animals, objects = engine.analyze_frame(frame)

    # Log detections
    for h in humans:
        logger.log("Human", h["label"], h["action"], h["conf"])
    for a in animals:
        logger.log("Animal", a["label"], a["action"], a["conf"])
    for o in objects:
        logger.log("Object", o["label"], o["action"], o["conf"])

    # Convert back to RGB for display
    result_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    report = format_detections(humans, animals, objects)

    return Image.fromarray(result_img), report


def process_webcam(frame, model_name):
    """Process a webcam frame (streaming)."""
    if frame is None:
        return None, "*Waiting for webcam...*"

    if model_name and model_name != engine.model_name:
        engine.change_model(model_name)

    # Gradio webcam gives numpy RGB
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (640, 480))

    annotated, humans, animals, objects = engine.analyze_frame(bgr)

    for h in humans:
        logger.log("Human", h["label"], h["action"], h["conf"])
    for a in animals:
        logger.log("Animal", a["label"], a["action"], a["conf"])

    result_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    report = format_detections(humans, animals, objects)

    return result_img, report


def process_video(video_path, model_name):
    """Process an uploaded video file, return annotated first frame + full report."""
    if video_path is None:
        return None, "*Upload a video to begin.*"

    if model_name and model_name != engine.model_name:
        engine.change_model(model_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "**Error:** Could not open video file."

    all_humans, all_animals, all_objects = [], [], []
    result_frame = None
    frame_idx = 0
    max_frames = 60  # process up to 60 frames

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))

        # Process every 3rd frame for speed
        if frame_idx % 3 == 0:
            annotated, humans, animals, objects = engine.analyze_frame(frame)
            all_humans.extend(humans)
            all_animals.extend(animals)
            all_objects.extend(objects)
            if result_frame is None:
                result_frame = annotated.copy()
            for h in humans:
                logger.log("Human", h["label"], h["action"], h["conf"])
            for a in animals:
                logger.log("Animal", a["label"], a["action"], a["conf"])

        frame_idx += 1

    cap.release()

    if result_frame is not None:
        result_img = Image.fromarray(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
    else:
        result_img = None

    report = format_detections(all_humans, all_animals, all_objects)
    report = f"**Analyzed {frame_idx} frames**\n\n" + report
    return result_img, report


def get_log_file():
    """Return the CSV log file for download."""
    if os.path.exists("detection_log.csv"):
        return "detection_log.csv"
    return None


def get_shock_rules_table():
    """Show shock rules as Markdown table."""
    lines = ["| Entity | Shock? | Current (µA) | Action |",
             "|--------|--------|-------------|--------|"]
    for key, rule in SHOCK_RULES.items():
        name = key.replace("_", " ").title()
        shock = "⚡ Yes" if rule.get("shock") else "❌ No"
        current = rule.get("current_uA", 0)
        lines.append(f"| {name} | {shock} | {current} | {rule['action']} |")
    return "\n".join(lines)


# ─────────────────────────────────────────────
#  GRADIO UI
# ─────────────────────────────────────────────
TITLE = "⚡ Smart Fencing System"
DESCRIPTION = """
**Real-Time Detection Based on Smart Fencing System for Farm Security**

Upload an image, video, or use your webcam to detect humans (with age & gender) and animals.
The system determines the appropriate shock deterrence based on detection type.

*By Rohan Nandanwar*
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
            label="🔧 YOLO Model",
            info="Larger models (m/l/x) are more accurate but slower"
        )

    with gr.Tabs():
        # ── Tab 1: Image Upload ──
        with gr.TabItem("📷 Image Upload"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="pil", label="Upload Image")
                    img_btn = gr.Button("🔍 Detect", variant="primary", size="lg")
                with gr.Column(scale=1):
                    img_output = gr.Image(label="Detection Result")
            img_report = gr.Markdown(label="Detection Report", elem_classes="detection-report")

            img_btn.click(
                fn=process_image,
                inputs=[img_input, model_selector],
                outputs=[img_output, img_report]
            )

        # ── Tab 2: Webcam (Live) ──
        with gr.TabItem("📹 Webcam (Live)"):
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

        # ── Tab 3: Video Upload ──
        with gr.TabItem("🎬 Video Upload"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid_input = gr.Video(label="Upload Video")
                    vid_btn = gr.Button("🔍 Analyze Video", variant="primary", size="lg")
                with gr.Column(scale=1):
                    vid_output = gr.Image(label="Detection Sample Frame")
            vid_report = gr.Markdown(elem_classes="detection-report")

            vid_btn.click(
                fn=process_video,
                inputs=[vid_input, model_selector],
                outputs=[vid_output, vid_report]
            )

        # ── Tab 4: Shock Rules ──
        with gr.TabItem("📋 Shock Rules"):
            gr.Markdown("### ⚡ Smart Fencing Shock Rules Table")
            gr.Markdown(get_shock_rules_table())

        # ── Tab 5: Detection Log ──
        with gr.TabItem("📊 Detection Log"):
            gr.Markdown("### Detection History")
            log_btn = gr.Button("📥 Download Detection Log", variant="secondary")
            log_file = gr.File(label="Log File")
            log_btn.click(fn=get_log_file, outputs=[log_file])

    gr.Markdown("---\n*Smart Fencing System v2.0 — Farm Security Through Intelligent Detection*")


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
