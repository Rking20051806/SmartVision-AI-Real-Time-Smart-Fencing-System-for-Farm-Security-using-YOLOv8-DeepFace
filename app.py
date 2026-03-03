"""
SmartVision AI - Web Demo
Real-Time Smart Fencing System for Farm Security using YOLOv8 & DeepFace
Streamlit Web Application for Live Hosting
Author: Rohan Nandanwar
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="SmartVision AI - Smart Fencing System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  SHOCK RULES TABLE (matches main.py)
# ─────────────────────────────────────────────
SHOCK_RULES = {
    # Humans
    "adult_male":    {"shock": True,  "current_uA": 4000, "action": "4000 µA Shock Deterrence"},
    "adult_female":  {"shock": True,  "current_uA": 2500, "action": "2500 µA Shock Deterrence"},
    "child":         {"shock": False, "current_uA": 0,    "action": "No Shock (Safety Override)"},
    "unknown_human": {"shock": False, "current_uA": 0,    "action": "Alert Only"},
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

ANIMAL_MAP = {
    "bird": "bird", "cat": "cat", "dog": "dog", "horse": "horse",
    "sheep": "sheep", "cow": "cow", "elephant": "elephant", "bear": "bear",
    "zebra": "zebra", "giraffe": "giraffe"
}

# Try importing optional dependencies
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


# ─────────────────────────────────────────────
#  DETECTION ENGINES
# ─────────────────────────────────────────────
@st.cache_resource
def load_yolo_model(model_name):
    if YOLO_AVAILABLE and YOLO is not None:
        return YOLO(model_name)
    return None


def analyze_human_face(frame, x1, y1, x2, y2):
    """Analyze human face for age/gender using DeepFace with face crop."""
    if not DEEPFACE_AVAILABLE or DeepFace is None:
        return "?", "Unknown"
    try:
        roi = frame[max(0, y1):y2, max(0, x1):x2]
        if roi.size == 0:
            return "?", "Unknown"

        # Detect face within person ROI for better accuracy
        face_roi = roi
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                                   minNeighbors=4, minSize=(30, 30))
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                pad = int(0.2 * max(fw, fh))
                fx1 = max(0, fx - pad)
                fy1 = max(0, fy - pad)
                fx2 = min(roi.shape[1], fx + fw + pad)
                fy2 = min(roi.shape[0], fy + fh + pad)
                face_roi = roi[fy1:fy2, fx1:fx2]
                if face_roi.size == 0:
                    face_roi = roi
        except Exception:
            face_roi = roi

        result = DeepFace.analyze(face_roi, actions=["age", "gender"],
                                  enforce_detection=False, silent=True)
        if isinstance(result, list):
            result = result[0]
        age = result.get("age", "?")
        gender = result.get("dominant_gender", "Unknown")
        return age, gender
    except Exception:
        return "?", "Unknown"


def get_human_rule(age, gender):
    """Determine shock rule based on age/gender."""
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


def process_frame_haar(frame):
    """Detection using OpenCV Haar Cascades (always available, no extra deps)."""
    annotated = frame.copy()
    humans, animals, objects_list = [], [], []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    body_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_fullbody.xml"
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        age, gender = analyze_human_face(frame, x, y, x + w, y + h)
        rule_key = get_human_rule(age, gender)
        rule = SHOCK_RULES.get(rule_key, SHOCK_RULES["unknown_human"])
        info = {
            "label": f"Person | Age:{age} Gender:{gender}",
            "action": rule["action"],
            "shock": rule["shock"],
            "conf": 0.85,
            "current_uA": rule.get("current_uA", 0),
        }
        humans.append(info)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, f"Human 0.85", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Detect full bodies
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 100))
    for (x, y, w, h) in bodies:
        overlap = any(abs(x - fx) < fw and abs(y - fy) < fh for (fx, fy, fw, fh) in faces)
        if not overlap:
            rule = SHOCK_RULES["unknown_human"]
            info = {
                "label": "Person (Body Detected)",
                "action": rule["action"],
                "shock": rule["shock"],
                "conf": 0.70,
                "current_uA": rule.get("current_uA", 0),
            }
            humans.append(info)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(annotated, "Human (Body)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    return annotated, humans, animals, objects_list


def process_frame_yolo(frame, model, confidence_threshold=0.35):
    """Process frame using YOLO model with DeepFace age/gender analysis."""
    if model is None:
        return process_frame_haar(frame)

    annotated = frame.copy()
    humans, animals, objects_list = [], [], []

    results = model(frame, verbose=False, conf=confidence_threshold)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id].lower()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_name == "person":
            age, gender = analyze_human_face(frame, x1, y1, x2, y2)
            rule_key = get_human_rule(age, gender)
            rule = SHOCK_RULES.get(rule_key, SHOCK_RULES["unknown_human"])
            info = {
                "label": f"Person | Age:{age} Gender:{gender}",
                "action": rule["action"],
                "shock": rule["shock"],
                "conf": conf,
                "current_uA": rule.get("current_uA", 0),
            }
            humans.append(info)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"Human {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        elif cls_name in ANIMAL_MAP:
            species = ANIMAL_MAP[cls_name]
            rule = SHOCK_RULES.get(species, {"action": "Alert", "shock": False, "current_uA": 0})
            info = {
                "label": f"{species.capitalize()} | Conf:{conf:.2f}",
                "action": rule["action"],
                "shock": rule["shock"],
                "conf": conf,
                "current_uA": rule.get("current_uA", 0),
            }
            animals.append(info)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(annotated, f"{species} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 2)
        else:
            info = {
                "label": f"{cls_name.capitalize()} | Conf:{conf:.2f}",
                "action": "Alert",
                "shock": False,
                "conf": conf,
            }
            objects_list.append(info)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, f"{cls_name} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    return annotated, humans, animals, objects_list


# ─────────────────────────────────────────────
#  UI COMPONENTS
# ─────────────────────────────────────────────
def display_detections(humans, animals, objects_list):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Humans Detected")
        if humans:
            for h in humans:
                status = "🔴 SHOCK" if h["shock"] else "🟢 SAFE"
                st.markdown(
                    f"**{h['label']}**\n"
                    f"- Status: {status}\n"
                    f"- Action: {h['action']}\n"
                    f"- Current: {h['current_uA']} µA\n"
                    f"- Confidence: {h['conf']:.2%}\n---"
                )
        else:
            st.info("No humans detected")

    with col2:
        st.subheader("🐾 Animals Detected")
        if animals:
            for a in animals:
                status = "🔴 SHOCK" if a["shock"] else "🟢 SAFE"
                st.markdown(
                    f"**{a['label']}**\n"
                    f"- Status: {status}\n"
                    f"- Action: {a['action']}\n"
                    f"- Current: {a.get('current_uA', 0)} µA\n---"
                )
        else:
            st.info("No animals detected")

    with col3:
        st.subheader("📦 Other Objects")
        if objects_list:
            for o in objects_list:
                st.markdown(f"**{o['label']}**\n- Action: {o['action']}\n---")
        else:
            st.info("No other objects detected")


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
def main():
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #1a5276, #2ecc71);
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    </style>
    <div class="main-header">
        <h1>🛡️ SmartVision AI</h1>
        <h3>Real-Time Smart Fencing System for Farm Security</h3>
        <p><em>Powered by YOLOv8 & DeepFace | Author: Rohan Nandanwar</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    detection_mode = "OpenCV Haar Cascade (Cloud)"
    model = None

    if YOLO_AVAILABLE:
        model_options = ["yolov8m.pt", "yolov8n.pt", "yolov8s.pt", "yolov8l.pt", "yolov8x.pt"]
        selected_model = st.sidebar.selectbox("Select YOLO Model", model_options)
        detection_mode = f"YOLOv8 ({selected_model})"
        with st.spinner(f"Loading {selected_model}..."):
            model = load_yolo_model(selected_model)

    confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.35, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Status")
    st.sidebar.markdown(f"- Detection: **{detection_mode}**")
    st.sidebar.markdown(f"- YOLO: {'✅' if YOLO_AVAILABLE else '⚠️ Haar Cascade fallback'}")
    st.sidebar.markdown(f"- DeepFace: {'✅' if DEEPFACE_AVAILABLE else '⚠️ Not available'}")
    st.sidebar.markdown(f"- OpenCV: ✅ {cv2.__version__}")

    if not YOLO_AVAILABLE:
        st.sidebar.info("💡 For full YOLO detection, run locally:\npip install ultralytics")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📷 Image Detection", "🎥 Video Detection", "📊 Shock Rules", "ℹ️ About"
    ])

    with tab1:
        st.header("Upload Image for Detection")
        uploaded_image = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png", "bmp"], key="img"
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            with st.spinner("🔍 Analyzing image..."):
                if YOLO_AVAILABLE and model:
                    annotated, humans, animals, objects_list = process_frame_yolo(
                        frame, model, confidence
                    )
                else:
                    annotated, humans, animals, objects_list = process_frame_haar(frame)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📸 Original")
                st.image(image, use_container_width=True)
            with col2:
                st.subheader("🔍 Detection Results")
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)

            st.markdown("---")
            total = len(humans) + len(animals) + len(objects_list)
            shock_count = sum(1 for h in humans if h["shock"]) + \
                          sum(1 for a in animals if a["shock"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Detections", total)
            c2.metric("Humans", len(humans))
            c3.metric("Animals", len(animals))
            c4.metric("⚡ Shock Alerts", shock_count)

            st.markdown("---")
            display_detections(humans, animals, objects_list)

    with tab2:
        st.header("Upload Video for Detection")
        uploaded_video = st.file_uploader(
            "Choose a video...", type=["mp4", "avi", "mov"], key="vid"
        )

        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            stats_placeholder = st.empty()
            progress_bar = st.progress(0)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            total_humans = 0
            total_animals = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                if frame_count % 5 == 0:
                    if YOLO_AVAILABLE and model:
                        annotated, humans, animals, objs = process_frame_yolo(
                            frame, model, confidence
                        )
                    else:
                        annotated, humans, animals, objs = process_frame_haar(frame)
                    total_humans += len(humans)
                    total_animals += len(animals)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    stframe.image(annotated_rgb, use_container_width=True)
                    stats_placeholder.markdown(
                        f"**Frame:** {frame_count}/{total_frames} | "
                        f"**Humans:** {len(humans)} | **Animals:** {len(animals)}"
                    )
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            os.unlink(tfile.name)
            progress_bar.progress(1.0)
            st.success(
                f"✅ Done! {frame_count} frames | "
                f"Humans: {total_humans} | Animals: {total_animals}"
            )

    with tab3:
        st.header("⚡ Smart Shock Rules Engine")
        st.markdown(
            "The system applies different deterrence levels based on the detected entity:"
        )

        st.subheader("👤 Human Rules")
        human_data = []
        for key in ["adult_male", "adult_female", "child", "unknown_human"]:
            r = SHOCK_RULES[key]
            human_data.append({
                "Entity": key.replace("_", " ").title(),
                "Shock": "Yes" if r["shock"] else "No",
                "Current (µA)": r["current_uA"],
                "Action": r["action"],
            })
        st.table(human_data)

        st.subheader("🐾 Animal Rules")
        animal_data = []
        for key in sorted(ANIMAL_MAP.values()):
            if key in SHOCK_RULES:
                r = SHOCK_RULES[key]
                animal_data.append({
                    "Species": key.capitalize(),
                    "Shock": "Yes" if r["shock"] else "No",
                    "Current (µA)": r["current_uA"],
                    "Action": r["action"],
                })
        # Add extra animals not in ANIMAL_MAP
        for key in ["chicken", "squirrel", "butterfly", "spider"]:
            if key in SHOCK_RULES:
                r = SHOCK_RULES[key]
                animal_data.append({
                    "Species": key.capitalize(),
                    "Shock": "Yes" if r["shock"] else "No",
                    "Current (µA)": r["current_uA"],
                    "Action": r["action"],
                })
        st.table(animal_data)

    with tab4:
        st.header("About SmartVision AI")
        st.markdown("""
        ### 🛡️ Smart Fencing System for Farm Security

        This AI-powered system provides real-time detection and classification of humans
        and animals approaching farm perimeters, with automatic deterrence actions.

        #### Features:
        - **YOLOv8 Object Detection** – Real-time detection of 80+ object classes
        - **DeepFace Analysis** – Age & gender recognition for humans (face-crop based)
        - **Smart Shock Rules** – Adaptive deterrence per entity type
        - **Safety Overrides** – Children automatically protected
        - **Buzzer / Ultrasonic Alerts** – For birds and small animals
        - **WhatsApp Alerts** – Instant notification with intruder photo (desktop)
        - **CSV Logging** – Every detection saved to logs

        #### Desktop vs Cloud:
        | Feature | Desktop (main.py) | Cloud (app.py) |
        |---------|-------------------|----------------|
        | YOLOv8 | ✅ Full | ⚠️ If installed |
        | DeepFace | ✅ Full | ⚠️ If installed |
        | Haar Cascade | ✅ | ✅ |
        | Live Webcam | ✅ | ❌ |
        | Image Upload | ✅ | ✅ |
        | Video Upload | ✅ | ✅ |
        | Buzzer Sound | ✅ | ❌ |
        | WhatsApp Alert | ✅ | ❌ |

        #### Author
        **Rohan Nandanwar** – B.Tech, DMIHER (DU)

        ---
        *Demonstration system. Actual electric fences require safety certifications
        per IEC 60479-1 and IEC 60335-2-76.*
        """)


if __name__ == "__main__":
    main()
