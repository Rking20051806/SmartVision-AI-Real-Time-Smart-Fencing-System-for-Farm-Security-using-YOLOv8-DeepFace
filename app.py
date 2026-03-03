"""
SmartVision AI - Web Demo
Real-Time Smart Fencing System for Farm Security using YOLOv8 & DeepFace
Streamlit Web Application for Live Hosting
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

# Try importing ML dependencies
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DeepFace = None
    DEEPFACE_AVAILABLE = False

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


@st.cache_resource
def load_yolo_model(model_name):
    """Load and cache YOLO model"""
    if YOLO_AVAILABLE:
        return YOLO(model_name)
    return None


def get_human_rule(age, gender):
    """Determine shock rule based on age and gender"""
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


def analyze_human(frame, x1, y1, x2, y2):
    """Analyze human for age and gender using DeepFace"""
    if not DEEPFACE_AVAILABLE:
        return "?", "Unknown"
    try:
        roi = frame[max(0, y1):y2, max(0, x1):x2]
        if roi.size == 0:
            return "?", "Unknown"
        
        result = DeepFace.analyze(roi, actions=["age", "gender"],
                                  enforce_detection=False, silent=True)
        if isinstance(result, list):
            result = result[0]
        age = result.get("age", "?")
        gender = result.get("dominant_gender") or result.get("gender") or "Unknown"
        return age, gender
    except Exception:
        return "?", "Unknown"


def process_frame(frame, model, confidence_threshold=0.4):
    """Process a single frame and return detections"""
    if model is None:
        return frame, [], [], []
    
    annotated = frame.copy()
    humans, animals, objects = [], [], []
    
    results = model(frame, verbose=False, conf=confidence_threshold)[0]
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id].lower()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        if cls_name == "person":
            age, gender = analyze_human(frame, x1, y1, x2, y2)
            rule_key = get_human_rule(age, gender)
            rule = SHOCK_RULES.get(rule_key, SHOCK_RULES["unknown_human"])
            info = {
                "label": f"Person | Age: {age} | Gender: {gender}",
                "action": rule["action"],
                "shock": rule["shock"],
                "conf": conf,
                "current_uA": rule.get("current_uA", 0),
            }
            humans.append(info)
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"Human {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        elif cls_name in ANIMAL_MAP:
            species = ANIMAL_MAP[cls_name]
            rule = SHOCK_RULES.get(species, {"action": "Alert", "shock": False})
            info = {
                "label": f"{species.capitalize()} | Conf: {conf:.2f}",
                "action": rule["action"],
                "shock": rule["shock"],
                "conf": conf,
                "current_uA": rule.get("current_uA", 0),
            }
            animals.append(info)
            color = (255, 165, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{species} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        else:
            info = {
                "label": f"{cls_name.capitalize()} | Conf: {conf:.2f}",
                "action": "Alert",
                "shock": False,
                "conf": conf,
            }
            objects.append(info)
            color = (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{cls_name} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return annotated, humans, animals, objects


def display_detections(humans, animals, objects):
    """Display detection results in Streamlit"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Humans Detected")
        if humans:
            for h in humans:
                status = "🔴 SHOCK" if h["shock"] else "🟢 SAFE"
                st.markdown(f"""
                **{h['label']}**
                - Status: {status}
                - Action: {h['action']}
                - Current: {h['current_uA']} µA
                - Confidence: {h['conf']:.2%}
                ---
                """)
        else:
            st.info("No humans detected")
    
    with col2:
        st.subheader("🐾 Animals Detected")
        if animals:
            for a in animals:
                status = "🔴 SHOCK" if a["shock"] else "🟢 SAFE"
                st.markdown(f"""
                **{a['label']}**
                - Status: {status}
                - Action: {a['action']}
                - Current: {a.get('current_uA', 0)} µA
                ---
                """)
        else:
            st.info("No animals detected")
    
    with col3:
        st.subheader("📦 Other Objects")
        if objects:
            for o in objects:
                st.markdown(f"""
                **{o['label']}**
                - Action: {o['action']}
                ---
                """)
        else:
            st.info("No other objects detected")


def main():
    # Header
    st.title("🛡️ SmartVision AI")
    st.markdown("### Real-Time Smart Fencing System for Farm Security")
    st.markdown("*Powered by YOLOv8 & DeepFace*")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Model selection
    model_options = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
    selected_model = st.sidebar.selectbox(
        "Select YOLO Model",
        model_options,
        help="Larger models are more accurate but slower"
    )
    
    # Confidence threshold
    confidence = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.4,
        step=0.05
    )
    
    # Status indicators
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Status")
    st.sidebar.markdown(f"- YOLO: {'✅ Available' if YOLO_AVAILABLE else '❌ Not Available'}")
    st.sidebar.markdown(f"- DeepFace: {'✅ Available' if DEEPFACE_AVAILABLE else '❌ Not Available'}")
    
    # Load model
    model = None
    if YOLO_AVAILABLE:
        with st.spinner(f"Loading {selected_model}..."):
            model = load_yolo_model(selected_model)
        st.sidebar.success(f"Model loaded: {selected_model}")
    else:
        st.sidebar.error("YOLO not available. Install: pip install ultralytics")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📷 Image Detection", "🎥 Video Detection", "ℹ️ About"])
    
    with tab1:
        st.header("Upload Image for Detection")
        uploaded_image = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png", "bmp"],
            key="image_uploader"
        )
        
        if uploaded_image is not None:
            # Load image
            image = Image.open(uploaded_image)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Process
            with st.spinner("Analyzing image..."):
                annotated, humans, animals, objects = process_frame(frame, model, confidence)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            with col2:
                st.subheader("Detection Results")
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)
            
            # Detection details
            st.markdown("---")
            st.header("📋 Detection Details")
            display_detections(humans, animals, objects)
            
            # Summary stats
            total = len(humans) + len(animals) + len(objects)
            shock_count = sum(1 for h in humans if h["shock"]) + sum(1 for a in animals if a["shock"])
            
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Detections", total)
            col2.metric("Humans", len(humans))
            col3.metric("Animals", len(animals))
            col4.metric("Shock Alerts", shock_count, delta_color="inverse")
    
    with tab2:
        st.header("Upload Video for Detection")
        uploaded_video = st.file_uploader(
            "Choose a video...",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_uploader"
        )
        
        if uploaded_video is not None:
            # Save video temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            
            # Video processing
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            stats_placeholder = st.empty()
            
            stop_button = st.button("Stop Processing")
            
            frame_count = 0
            total_humans = 0
            total_animals = 0
            total_shocks = 0
            
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every 3rd frame for speed
                if frame_count % 3 == 0:
                    annotated, humans, animals, objects = process_frame(frame, model, confidence)
                    total_humans += len(humans)
                    total_animals += len(animals)
                    total_shocks += sum(1 for h in humans if h["shock"]) + sum(1 for a in animals if a["shock"])
                    
                    # Display frame
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    stframe.image(annotated_rgb, use_container_width=True)
                    
                    # Update stats
                    stats_placeholder.markdown(f"""
                    **Frame:** {frame_count} | **Humans:** {len(humans)} | **Animals:** {len(animals)} | **Shock Alerts:** {total_shocks}
                    """)
            
            cap.release()
            os.unlink(tfile.name)
            
            st.success(f"Video processing complete! Processed {frame_count} frames.")
    
    with tab3:
        st.header("About SmartVision AI")
        st.markdown("""
        ### 🛡️ Smart Fencing System for Farm Security
        
        This AI-powered system provides real-time detection and classification of humans and animals
        approaching farm perimeters, automatically determining appropriate deterrence actions.
        
        #### Features:
        - **YOLOv8 Object Detection**: State-of-the-art real-time detection
        - **DeepFace Analysis**: Age and gender recognition for humans
        - **Smart Shock Rules**: Adaptive deterrence based on detected entity
        - **Safety Overrides**: Children automatically protected from shock
        
        #### Shock Rules:
        | Entity | Current (µA) | Action |
        |--------|-------------|--------|
        | Adult Male | 4000 | Shock Deterrence |
        | Adult Female | 2500 | Shock Deterrence |
        | Child | 0 | Safety Override |
        | Large Animals | 2500-4000 | Shock Deterrence |
        | Small Animals | 0-1800 | Ultrasonic/Alert |
        
        #### Author
        **Rohan Nandanwar**
        
        ---
        *This is a demonstration system. Actual electric fence implementations require proper safety certifications and regulations compliance.*
        """)


if __name__ == "__main__":
    main()
