"""
SmartVision AI – Streamlit Web Application
Real-Time Smart Fencing System for Farm Security using YOLOv8 & DeepFace
Author: Rohan Nandanwar | B.Tech Minor Project, DMIHER (DU)
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import csv
import pandas as pd
from datetime import datetime

# ─── Page Config ──────────────────────────────
st.set_page_config(
    page_title="SmartVision AI – Smart Fencing System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for dark dashboard look ───────
st.markdown("""
<style>
/* ── Global dark theme ── */
[data-testid="stAppViewContainer"] { background-color: #0e1117; }
[data-testid="stSidebar"] { background-color: #16213e; }
header[data-testid="stHeader"] { background-color: #0e1117; }

/* ── Header banner ── */
.hero-banner {
    background: linear-gradient(135deg, #16213e 0%, #1a5276 50%, #1e8449 100%);
    border-radius: 14px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,.4);
}
.hero-banner h1 { color: #e94560; margin: 0; font-size: 2.2rem; }
.hero-banner h3 { color: #ecf0f1; margin: .3rem 0 0; font-weight: 400; font-size: 1.1rem; }
.hero-banner p  { color: #aab7c4; margin: .5rem 0 0; font-size: .85rem; }

/* ── Metric cards ── */
.metric-card {
    background: #16213e;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    border: 1px solid #1a5276;
    box-shadow: 0 2px 10px rgba(0,0,0,.3);
}
.metric-card .num  { font-size: 2.4rem; font-weight: 700; margin: 0; }
.metric-card .lbl  { color: #aab7c4; font-size: .85rem; margin: 0; }

/* ── Detection panel ── */
.det-panel {
    background: #16213e;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: .8rem;
    border-left: 4px solid;
    box-shadow: 0 2px 8px rgba(0,0,0,.25);
}
.det-panel.human  { border-color: #27ae60; }
.det-panel.animal { border-color: #e67e22; }
.det-panel.object { border-color: #e94560; }
.det-panel .title { font-weight: 700; font-size: 1rem; margin: 0 0 .3rem; }
.det-panel .info  { color: #bdc3c7; font-size: .88rem; line-height: 1.6; }
.det-panel .shock { color: #e74c3c; font-weight: 700; }
.det-panel .safe  { color: #2ecc71; font-weight: 700; }

/* ── Table styling ── */
.rules-table { width: 100%; border-collapse: collapse; margin: .5rem 0; }
.rules-table th {
    background: #1a5276; color: #ecf0f1; padding: .6rem .8rem;
    text-align: left; font-size: .9rem;
}
.rules-table td {
    background: #16213e; color: #bdc3c7; padding: .5rem .8rem;
    border-bottom: 1px solid #0f3460; font-size: .88rem;
}
.rules-table tr:hover td { background: #1a3a5c; }

/* ── Sidebar status badges ── */
.status-ok  { color: #2ecc71; font-weight: 600; }
.status-warn { color: #f39c12; font-weight: 600; }

/* Hide default Streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Shock Rules ──────────────────────────────
SHOCK_RULES = {
    "adult_male":    {"shock": True,  "current_uA": 4000, "action": "4000 µA Shock Deterrence"},
    "adult_female":  {"shock": True,  "current_uA": 2500, "action": "2500 µA Shock Deterrence"},
    "child":         {"shock": False, "current_uA": 0,    "action": "No Shock (Safety Override)"},
    "unknown_human": {"shock": False, "current_uA": 0,    "action": "Alert Only"},
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
    "zebra": "zebra", "giraffe": "giraffe",
}

# ─── Optional dependency imports ──────────────
YOLO_AVAILABLE = False
YOLO = None
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass

DEEPFACE_AVAILABLE = False
DeepFace = None
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DETECTION HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_resource
def load_yolo_model(model_name: str):
    """Load and cache a YOLO model (downloaded automatically if needed)."""
    if YOLO_AVAILABLE and YOLO is not None:
        return YOLO(model_name)
    return None


def _analyze_face(frame, x1, y1, x2, y2):
    """Use DeepFace on a face-cropped ROI for age/gender."""
    if not DEEPFACE_AVAILABLE or DeepFace is None:
        return "?", "Unknown"
    try:
        roi = frame[max(0, y1):y2, max(0, x1):x2]
        if roi.size == 0:
            return "?", "Unknown"
        face_roi = roi
        try:
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                pad = int(0.2 * max(fw, fh))
                face_roi = roi[
                    max(0, fy - pad): min(roi.shape[0], fy + fh + pad),
                    max(0, fx - pad): min(roi.shape[1], fx + fw + pad),
                ]
                if face_roi.size == 0:
                    face_roi = roi
        except Exception:
            pass
        res = DeepFace.analyze(face_roi, actions=["age", "gender"],
                               enforce_detection=False, silent=True)
        if isinstance(res, list):
            res = res[0]
        return res.get("age", "?"), res.get("dominant_gender", "Unknown")
    except Exception:
        return "?", "Unknown"


def _human_rule(age, gender):
    try:
        age = int(age)
    except (ValueError, TypeError):
        return "unknown_human"
    if age < 18:
        return "child"
    g = str(gender).lower()
    if "man" in g or "male" in g:
        return "adult_male"
    if "woman" in g or "female" in g:
        return "adult_female"
    return "unknown_human"


# ── Haar cascade fallback (always available) ──
def detect_haar(frame):
    annotated = frame.copy()
    humans, animals, objects_list = [], [], []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    body_cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

    faces = face_cas.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    for x, y, w, h in faces:
        age, gender = _analyze_face(frame, x, y, x + w, y + h)
        rk = _human_rule(age, gender)
        rule = SHOCK_RULES.get(rk, SHOCK_RULES["unknown_human"])
        humans.append(dict(label=f"Person | Age:{age} Gender:{gender}",
                           action=rule["action"], shock=rule["shock"],
                           current_uA=rule["current_uA"], conf=0.85))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, "Human 0.85", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    bodies = body_cas.detectMultiScale(gray, 1.1, 3, minSize=(50, 100))
    for x, y, w, h in bodies:
        overlap = any(abs(x - fx) < fw and abs(y - fy) < fh for fx, fy, fw, fh in faces)
        if not overlap:
            rule = SHOCK_RULES["unknown_human"]
            humans.append(dict(label="Person (Body)", action=rule["action"],
                               shock=False, current_uA=0, conf=0.70))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(annotated, "Human (body)", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

    return annotated, humans, animals, objects_list


# ── YOLO detection ────────────────────────────
def detect_yolo(frame, model, conf_thr=0.35):
    if model is None:
        return detect_haar(frame)
    annotated = frame.copy()
    humans, animals, objects_list = [], [], []

    results = model(frame, verbose=False, conf=conf_thr)[0]
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id].lower()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_name == "person":
            age, gender = _analyze_face(frame, x1, y1, x2, y2)
            rk = _human_rule(age, gender)
            rule = SHOCK_RULES.get(rk, SHOCK_RULES["unknown_human"])
            humans.append(dict(label=f"Person | Age:{age} Gender:{gender}",
                               action=rule["action"], shock=rule["shock"],
                               current_uA=rule["current_uA"], conf=conf))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"Human {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        elif cls_name in ANIMAL_MAP:
            species = ANIMAL_MAP[cls_name]
            rule = SHOCK_RULES.get(species, {"action": "Alert", "shock": False, "current_uA": 0})
            animals.append(dict(label=f"{species.capitalize()} | Conf:{conf:.2f}",
                                action=rule["action"], shock=rule["shock"],
                                current_uA=rule.get("current_uA", 0), conf=conf))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(annotated, f"{species} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 2)
        else:
            objects_list.append(dict(label=f"{cls_name.capitalize()} | Conf:{conf:.2f}",
                                     action="Alert", shock=False, current_uA=0, conf=conf))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, f"{cls_name} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    return annotated, humans, animals, objects_list


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UI RENDERING HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_metric(label, value, color="#e94560"):
    return f"""
    <div class="metric-card">
        <p class="num" style="color:{color}">{value}</p>
        <p class="lbl">{label}</p>
    </div>"""


def render_detection_panel(det, category="human"):
    status_cls = "shock" if det["shock"] else "safe"
    status_txt = "⚡ SHOCK" if det["shock"] else "✔ SAFE"
    return f"""
    <div class="det-panel {category}">
        <p class="title">{det['label']}</p>
        <p class="info">
            Status: <span class="{status_cls}">{status_txt}</span><br>
            Action: {det['action']}<br>
            Current: {det.get('current_uA', 0)} µA &nbsp;|&nbsp; Confidence: {det['conf']:.0%}
        </p>
    </div>"""


def render_detections(humans, animals, objects_list):
    """Three-column detection dashboard."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 👤 Human Detections")
        if humans:
            for h in humans:
                st.markdown(render_detection_panel(h, "human"), unsafe_allow_html=True)
        else:
            st.info("No humans detected")
    with c2:
        st.markdown("#### 🐾 Animal Detections")
        if animals:
            for a in animals:
                st.markdown(render_detection_panel(a, "animal"), unsafe_allow_html=True)
        else:
            st.info("No animals detected")
    with c3:
        st.markdown("#### 📦 Other Objects")
        if objects_list:
            for o in objects_list:
                st.markdown(render_detection_panel(o, "object"), unsafe_allow_html=True)
        else:
            st.info("No other objects")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN APPLICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    # ── Hero Banner ──
    st.markdown("""
    <div class="hero-banner">
        <h1>🛡️ SmartVision AI</h1>
        <h3>Real-Time Smart Fencing System for Farm Security</h3>
        <p>Powered by YOLOv8 &amp; DeepFace &nbsp;|&nbsp; Author: Rohan Nandanwar &nbsp;|&nbsp; DMIHER (DU)</p>
    </div>""", unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
        st.markdown("## ⚙️ Control Panel")
        st.markdown("---")

        # Model selector
        model = None
        det_mode = "Haar Cascade (fallback)"
        if YOLO_AVAILABLE:
            model_choice = st.selectbox(
                "🔧 YOLO Model",
                ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                index=2,  # default yolov8m.pt
                help="Larger models are more accurate but slower",
            )
            det_mode = f"YOLOv8 ({model_choice})"
            with st.spinner(f"Loading {model_choice}…"):
                model = load_yolo_model(model_choice)
        else:
            st.warning("YOLO not available – using Haar Cascade fallback")

        conf_threshold = st.slider("🎯 Confidence Threshold", 0.10, 1.00, 0.35, 0.05)

        st.markdown("---")
        st.markdown("### 📊 System Status")
        yolo_status = '<span class="status-ok">✅ Loaded</span>' if YOLO_AVAILABLE else '<span class="status-warn">⚠️ Haar fallback</span>'
        df_status = '<span class="status-ok">✅ Loaded</span>' if DEEPFACE_AVAILABLE else '<span class="status-warn">⚠️ Not available</span>'
        st.markdown(f"**Detection:** {det_mode}", unsafe_allow_html=True)
        st.markdown(f"**YOLO:** {yolo_status}", unsafe_allow_html=True)
        st.markdown(f"**DeepFace:** {df_status}", unsafe_allow_html=True)
        st.markdown(f"**OpenCV:** <span class='status-ok'>✅ {cv2.__version__}</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            "<small style='color:#7f8c8d'>© 2025 Rohan Nandanwar<br>"
            "B.Tech Minor Project – DMIHER (DU)</small>",
            unsafe_allow_html=True,
        )

    # ── Main tabs ──
    tab_img, tab_vid, tab_rules, tab_logs, tab_about = st.tabs([
        "📷 Image Detection",
        "🎥 Video Detection",
        "⚡ Shock Rules",
        "📋 Detection Logs",
        "ℹ️ About",
    ])

    # ━━━━━━ TAB 1 — IMAGE ━━━━━━
    with tab_img:
        st.markdown("### 📷 Upload an Image for Detection")
        uploaded = st.file_uploader("Choose an image…", type=["jpg", "jpeg", "png", "bmp"], key="img_up")

        if uploaded is not None:
            image = Image.open(uploaded)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            with st.spinner("🔍 Analyzing image…"):
                if YOLO_AVAILABLE and model:
                    annotated, humans, animals, objs = detect_yolo(frame, model, conf_threshold)
                else:
                    annotated, humans, animals, objs = detect_haar(frame)

            # Side-by-side images
            left, right = st.columns(2)
            with left:
                st.markdown("**📸 Original Image**")
                st.image(image, use_container_width=True)
            with right:
                st.markdown("**🔍 Detection Results**")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Metric cards
            total = len(humans) + len(animals) + len(objs)
            shocks = sum(1 for d in humans + animals if d["shock"])
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(render_metric("Total Detections", total, "#3498db"), unsafe_allow_html=True)
            m2.markdown(render_metric("Humans", len(humans), "#27ae60"), unsafe_allow_html=True)
            m3.markdown(render_metric("Animals", len(animals), "#e67e22"), unsafe_allow_html=True)
            m4.markdown(render_metric("⚡ Shock Alerts", shocks, "#e74c3c"), unsafe_allow_html=True)

            st.markdown("---")
            render_detections(humans, animals, objs)

            # Save log
            _save_log(humans, animals, objs)

        else:
            st.markdown(
                "<div style='text-align:center;padding:4rem 0;color:#7f8c8d'>"
                "<h2>📷</h2><p>Upload an image above to start detection</p></div>",
                unsafe_allow_html=True,
            )

    # ━━━━━━ TAB 2 — VIDEO ━━━━━━
    with tab_vid:
        st.markdown("### 🎥 Upload a Video for Detection")
        vid_file = st.file_uploader("Choose a video…", type=["mp4", "avi", "mov", "mkv"], key="vid_up")

        if vid_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(vid_file.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            st.markdown("**🎬 Processing Video…**")
            progress = st.progress(0)
            frame_display = st.empty()
            stats_display = st.empty()

            frame_idx = 0
            sum_humans = 0
            sum_animals = 0
            all_humans, all_animals, all_objs = [], [], []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                # Process every 5th frame for performance
                if frame_idx % 5 == 0:
                    if YOLO_AVAILABLE and model:
                        ann, h, a, o = detect_yolo(frame, model, conf_threshold)
                    else:
                        ann, h, a, o = detect_haar(frame)
                    sum_humans += len(h)
                    sum_animals += len(a)
                    all_humans.extend(h)
                    all_animals.extend(a)
                    all_objs.extend(o)
                    frame_display.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                                        use_container_width=True)
                    stats_display.markdown(
                        f"**Frame** {frame_idx}/{total_frames} &nbsp;|&nbsp; "
                        f"**Humans:** {len(h)} &nbsp;|&nbsp; **Animals:** {len(a)}"
                    )
                if total_frames > 0:
                    progress.progress(min(frame_idx / total_frames, 1.0))

            cap.release()
            os.unlink(tfile.name)
            progress.progress(1.0)

            st.success(f"✅ Processed {frame_idx} frames  |  Humans: {sum_humans}  |  Animals: {sum_animals}")

            # Summary metrics
            m1, m2, m3 = st.columns(3)
            m1.markdown(render_metric("Frames Processed", frame_idx, "#3498db"), unsafe_allow_html=True)
            m2.markdown(render_metric("Total Humans", sum_humans, "#27ae60"), unsafe_allow_html=True)
            m3.markdown(render_metric("Total Animals", sum_animals, "#e67e22"), unsafe_allow_html=True)

            _save_log(all_humans, all_animals, all_objs)
        else:
            st.markdown(
                "<div style='text-align:center;padding:4rem 0;color:#7f8c8d'>"
                "<h2>🎥</h2><p>Upload a video above to start detection</p></div>",
                unsafe_allow_html=True,
            )

    # ━━━━━━ TAB 3 — SHOCK RULES ━━━━━━
    with tab_rules:
        st.markdown("### ⚡ Smart Shock Rules Engine")
        st.markdown("The system applies adaptive deterrence levels based on the detected entity type.")

        st.markdown("#### 👤 Human Rules")
        human_html = '<table class="rules-table"><tr><th>Entity</th><th>Shock</th><th>Current (µA)</th><th>Action</th></tr>'
        for key in ["adult_male", "adult_female", "child", "unknown_human"]:
            r = SHOCK_RULES[key]
            shock_txt = "✅ Yes" if r["shock"] else "❌ No"
            human_html += f'<tr><td>{key.replace("_"," ").title()}</td><td>{shock_txt}</td><td>{r["current_uA"]}</td><td>{r["action"]}</td></tr>'
        human_html += "</table>"
        st.markdown(human_html, unsafe_allow_html=True)

        st.markdown("#### 🐾 Animal Rules")
        animal_keys = sorted(ANIMAL_MAP.values()) + ["chicken", "squirrel", "butterfly", "spider"]
        seen = set()
        animal_html = '<table class="rules-table"><tr><th>Species</th><th>Shock</th><th>Current (µA)</th><th>Action</th></tr>'
        for key in animal_keys:
            if key in seen or key not in SHOCK_RULES:
                continue
            seen.add(key)
            r = SHOCK_RULES[key]
            shock_txt = "✅ Yes" if r["shock"] else "❌ No"
            animal_html += f'<tr><td>{key.capitalize()}</td><td>{shock_txt}</td><td>{r["current_uA"]}</td><td>{r["action"]}</td></tr>'
        animal_html += "</table>"
        st.markdown(animal_html, unsafe_allow_html=True)

        st.markdown(
            "<br><small style='color:#95a5a6'>⚠️ Currents are for simulation/reference only. "
            "Real hardware must comply with IEC 60479-1 & IEC 60335-2-76.</small>",
            unsafe_allow_html=True,
        )

    # ━━━━━━ TAB 4 — LOGS ━━━━━━
    with tab_logs:
        st.markdown("### 📋 Detection Logs")
        log_path = "logs/detection_log.csv"
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            st.dataframe(df, use_container_width=True, height=400)
            st.download_button("⬇️ Download CSV", df.to_csv(index=False),
                               "detection_log.csv", "text/csv")
        else:
            st.info("No detection logs yet. Run a detection to generate logs.")

    # ━━━━━━ TAB 5 — ABOUT ━━━━━━
    with tab_about:
        st.markdown("""
        ### 🛡️ About SmartVision AI

        **Smart Fencing System for Farm Security** — An AI-powered perimeter defense
        that detects, classifies, and deters intruders in real time.

        #### Key Features
        | Feature | Desktop (`main.py`) | Cloud (`app.py`) |
        |---------|:---:|:---:|
        | YOLOv8 Detection | ✅ | ✅ |
        | DeepFace Age/Gender | ✅ | ⚠️ Optional |
        | Haar Cascade Fallback | ✅ | ✅ |
        | Live Webcam | ✅ | ❌ |
        | Image Upload | ✅ | ✅ |
        | Video Upload | ✅ | ✅ |
        | Buzzer / Ultrasonic | ✅ | ❌ |
        | WhatsApp Alerts | ✅ | ❌ |
        | CSV Logging | ✅ | ✅ |

        #### Technology Stack
        - **YOLOv8** (Ultralytics) – Real-time object detection (80+ classes)
        - **DeepFace** – Face analysis for age & gender estimation
        - **OpenCV** – Image processing & Haar cascade fallback
        - **Streamlit** – Web application framework
        - **Tkinter** – Desktop GUI (main.py)

        #### Author
        **Rohan Nandanwar** — B.Tech, DMIHER (DU)

        ---
        *This is a demonstration/simulation system. Actual electric fencing
        requires safety certifications per IEC 60479-1 and IEC 60335-2-76.*
        """)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LOGGING HELPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _save_log(humans, animals, objects_list):
    """Append detection results to CSV log."""
    log_dir = "logs"
    log_path = os.path.join(log_dir, "detection_log.csv")
    os.makedirs(log_dir, exist_ok=True)

    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Timestamp", "Type", "Label", "Action", "Confidence"])
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for h in humans:
            writer.writerow([ts, "Human", h["label"], h["action"], f"{h['conf']:.2f}"])
        for a in animals:
            writer.writerow([ts, "Animal", a["label"], a["action"], f"{a['conf']:.2f}"])
        for o in objects_list:
            writer.writerow([ts, "Object", o["label"], o["action"], f"{o['conf']:.2f}"])


if __name__ == "__main__":
    main()
