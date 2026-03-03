"""
Smart Fencing System - Main Application
Real-Time Detection Based on Smart Fencing System for Farm Security
Author: Rohan Nandanwar
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import threading
import time
import csv
import os
from datetime import datetime
from PIL import Image, ImageTk
import numpy as np
import math
import struct
import wave

ASSET_DIR = "assets"
DEFAULT_BUZZER_WAV = os.path.join(ASSET_DIR, "smartvision_buzzer.wav")


def ensure_default_buzzer_audio():
    """Create a bundled buzzer tone so alerts always have audio."""
    os.makedirs(ASSET_DIR, exist_ok=True)
    if os.path.exists(DEFAULT_BUZZER_WAV):
        return
    sample_rate = 22050
    duration_sec = 1.4
    total_samples = int(sample_rate * duration_sec)
    with wave.open(DEFAULT_BUZZER_WAV, "w") as wav_file:
        wav_file.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        for i in range(total_samples):
            t = i / sample_rate
            freq = 1500 if t < 0.7 else 2100
            value = int(32767 * math.sin(2 * math.pi * freq * t))
            wav_file.writeframes(struct.pack("<h", value))


ensure_default_buzzer_audio()

# Try importing optional dependencies
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not installed. Run: pip install ultralytics")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DeepFace = None
    DEEPFACE_AVAILABLE = False
    print("[WARNING] deepface not installed. Run: pip install deepface")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False
    print("[WARNING] tensorflow not installed. Run: pip install tensorflow")

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None
    PYGAME_AVAILABLE = False
    print("[WARNING] pygame not installed. Run: pip install pygame")
except Exception as e:
    pygame = None
    PYGAME_AVAILABLE = False
    print(f"[WARNING] pygame mixer init failed: {e}")

try:
    import pywhatkit
    WHATSAPP_AVAILABLE = True
except ImportError:
    pywhatkit = None
    WHATSAPP_AVAILABLE = False
    print("[WARNING] pywhatkit not installed. Run: pip install pywhatkit")

try:
    import winsound  # Windows-only; used for fallback buzzer tones
    WINSOUND_AVAILABLE = True
except ImportError:
    winsound = None
    WINSOUND_AVAILABLE = False

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
    def __init__(self, model_name="yolov8n.pt"):
        self.model = None
        self.animal_cnn = None
        self.model_name = model_name
        self.frame_count = 0
        self.deepface_skip = 8
        self._last_human_info = {}
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._load_models()

    def _load_models(self):
        if YOLO_AVAILABLE and YOLO is not None:
            try:
                self.model = YOLO(self.model_name)
                print(f"[OK] YOLO model loaded: {self.model_name}")
            except Exception as e:
                print(f"[ERROR] YOLO load failed: {e}")
        if TF_AVAILABLE and tf is not None and os.path.exists("animal10.h5"):
            try:
                self.animal_cnn = tf.keras.models.load_model("animal10.h5")
                print("[OK] Animal-10 CNN loaded.")
            except Exception as e:
                print(f"[WARN] Animal CNN load failed: {e}")

    def change_model(self, model_name):
        self.model_name = model_name
        if YOLO_AVAILABLE and YOLO is not None:
            try:
                self.model = YOLO(model_name)
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

        results = self.model(frame, verbose=False, conf=0.4)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id].lower()
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_name == "person":
                age, gender = self._analyze_human(frame, x1, y1, x2, y2)
                rule_key = self._get_human_rule(age, gender)
                rule = SHOCK_RULES.get(rule_key, SHOCK_RULES["unknown_human"])
                info = {
                    "label": f"Person | Age:{age} Gender:{gender}",
                    "action": rule["action"],
                    "shock": rule["shock"],
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2),
                    "age": age,
                    "gender": gender,
                    "rule_key": rule_key,
                    "current_uA": rule.get("current_uA"),
                }
                humans.append(info)
                color = (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"Human {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            elif cls_name in ANIMAL_MAP:
                species = ANIMAL_MAP[cls_name]
                rule = SHOCK_RULES.get(species, {"action": "Alert", "shock": False})
                info = {
                    "label": f"{species.capitalize()} | Conf:{conf:.2f}",
                    "action": rule["action"],
                    "shock": rule["shock"],
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2),
                    "species": species,
                    "rule_key": species,
                    "current_uA": rule.get("current_uA"),
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
                    "bbox": (x1, y1, x2, y2),
                    "class_name": cls_name,
                }
                objects.append(info)
                color = (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{cls_name} {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        return annotated, humans, animals, objects

    def _analyze_human(self, frame, x1, y1, x2, y2):
        if not DEEPFACE_AVAILABLE or DeepFace is None:
            return "?", "Unknown"
        if self.frame_count % self.deepface_skip != 0:
            cached = self._last_human_info
            return cached.get("age", "?"), cached.get("gender", "Unknown")
        try:
            roi = frame[max(0, y1):y2, max(0, x1):x2]
            if roi.size == 0:
                return "?", "Unknown"

            # Try to focus on the face within the person ROI for more reliable age/gender
            face_roi = roi
            if not self.face_cascade.empty():
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
                if len(faces) > 0:
                    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                    pad = int(0.15 * max(fw, fh))
                    fx1 = max(0, fx - pad)
                    fy1 = max(0, fy - pad)
                    fx2 = min(roi.shape[1], fx + fw + pad)
                    fy2 = min(roi.shape[0], fy + fh + pad)
                    face_roi = roi[fy1:fy2, fx1:fx2]

            result = DeepFace.analyze(face_roi, actions=["age", "gender"],
                                      enforce_detection=False, silent=True)
            if isinstance(result, list):
                result = result[0]
            age = result.get("age", "?")
            gender = result.get("dominant_gender") or result.get("gender") or "Unknown"
            self._last_human_info = {"age": age, "gender": gender}
            return age, gender
        except Exception as e:
            print(f"[WARN] DeepFace analyze failed: {e}")
            cached = self._last_human_info
            return cached.get("age", "?"), cached.get("gender", "Unknown")

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
import webbrowser
from urllib.parse import quote

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    pyautogui = None
    PYAUTOGUI_AVAILABLE = False
    print("[WARNING] pyautogui not installed. Run: pip install pyautogui")

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PILImage = None
    PIL_AVAILABLE = False

try:
    import win32clipboard
    WIN32CLIP_AVAILABLE = True
except ImportError:
    win32clipboard = None
    WIN32CLIP_AVAILABLE = False
    print("[WARNING] pywin32 not installed. Run: pip install pywin32")


class AlertSystem:
    def __init__(self, whatsapp_number="", notify_callback=None):
        self.whatsapp_number = whatsapp_number
        self.buzzer_active = False
        self._buzzer_thread = None
        self.notify_callback = notify_callback
        self.whatsapp_wait_time = 15

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
        media_path = self._get_buzzer_media()
        if media_path and PYGAME_AVAILABLE and pygame is not None:
            try:
                pygame.mixer.music.load(media_path)
                pygame.mixer.music.play()
                time.sleep(5)
                played = True
            except Exception as e:
                print(f"[BUZZER ERROR] {e}")
        if not played:
            self._play_fallback_tone()
        self.buzzer_active = False

    def _play_fallback_tone(self):
        if WINSOUND_AVAILABLE and winsound is not None:
            try:
                for freq in (1400, 1800, 2200, 1800):
                    winsound.Beep(freq, 300)
            except Exception as e:
                print(f"[BUZZER] Fallback tone failed: {e}")
        else:
            print("[BUZZER] Ultrasonic alert triggered (audio device unavailable)")
            time.sleep(3)

    def _get_buzzer_media(self):
        for candidate in ("preview.mp3", "preview.wav", DEFAULT_BUZZER_WAV):
            if candidate and os.path.exists(candidate):
                return candidate
        return None

    def _notify(self, message):
        if callable(self.notify_callback):
            try:
                self.notify_callback(message)
            except Exception:
                print(message)
        else:
            print(message)

    def send_whatsapp(self, message, image_path=None):
        threading.Thread(
            target=self._send_whatsapp_worker,
            args=(message, image_path),
            daemon=True,
        ).start()

    def _copy_image_to_clipboard(self, image_path):
        """Copy image to clipboard for pasting into WhatsApp"""
        if not WIN32CLIP_AVAILABLE or win32clipboard is None:
            self._notify("[CLIPBOARD] win32clipboard not available")
            return False
        if not PIL_AVAILABLE or PILImage is None:
            self._notify("[CLIPBOARD] PIL not available")
            return False
            
        try:
            from io import BytesIO
            
            img = PILImage.open(image_path)
            output = BytesIO()
            img.convert("RGB").save(output, "BMP")
            data = output.getvalue()[14:]  # Remove BMP header
            output.close()
            
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
            win32clipboard.CloseClipboard()
            return True
        except Exception as e:
            self._notify(f"[CLIPBOARD ERROR] {e}")
            return False

    def _send_whatsapp_worker(self, message, image_path):
        target = (self.whatsapp_number or "").strip()
        if not target:
            self._notify(f"[WHATSAPP] Skipped (number missing)")
            return

        # Clean up phone number
        phone = target.replace(" ", "").replace("-", "")
        if not phone.startswith("+"):
            phone = "+" + phone

        resolved_image = None
        if image_path:
            abs_path = os.path.abspath(image_path)
            if os.path.exists(abs_path):
                resolved_image = abs_path
                self._notify(f"[WHATSAPP] Image found: {os.path.basename(abs_path)}")
            else:
                self._notify(f"[WHATSAPP] Image missing: {abs_path}")

        # Open WhatsApp Web with message
        encoded_msg = quote(message)
        url = f"https://web.whatsapp.com/send?phone={phone}&text={encoded_msg}"
        
        self._notify(f"[WHATSAPP] Opening WhatsApp Web for {phone}...")
        webbrowser.open(url)
        
        # Wait for page to load
        self._notify("[WHATSAPP] Waiting for page to load (15s)...")
        time.sleep(self.whatsapp_wait_time)

        if not PYAUTOGUI_AVAILABLE or pyautogui is None:
            self._notify("[WHATSAPP] pyautogui unavailable - please send manually")
            return

        try:
            if resolved_image:
                # Copy image to clipboard and paste
                self._notify("[WHATSAPP] Copying image to clipboard...")
                if self._copy_image_to_clipboard(resolved_image):
                    time.sleep(1)
                    # Paste image with Ctrl+V
                    pyautogui.hotkey('ctrl', 'v')
                    self._notify("[WHATSAPP] Image pasted, waiting...")
                    time.sleep(3)
                    # Press Enter to send
                    pyautogui.press('enter')
                    time.sleep(1)
            
            # Send the message
            pyautogui.press('enter')
            self._notify(f"[WHATSAPP] Message sent to {phone}!")
            
        except Exception as e:
            self._notify(f"[WHATSAPP ERROR] {e}")


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
        self.alert = AlertSystem(notify_callback=self._log_dialog)

        self.cap = None
        self.source = None
        self.failed_reads = 0
        self.max_failed_reads = 30
        self.running = False
        self.last_frame = None
        self.crop_mode = False
        self.auto_whatsapp_var = tk.BooleanVar(value=False)
        self.alert_cooldown_sec = 60
        self._last_sent = {}
        self.shock_enabled = True
        self._last_manual_label = ""

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
        self.model_var = tk.StringVar(value="yolov8n.pt")
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
        tk.Button(btnbar, text="🛑 Manual Override", command=self._manual_override,
              bg="#c0392b", fg="white", font=("Helvetica", 10, "bold"),
              relief="flat", padx=12, pady=5, cursor="hand2").pack(side="left", padx=6)
        tk.Button(btnbar, text="🔇 Stop Buzzer",    command=self._stop_buzzer,
                  bg="#f39c12", fg="white", font=("Helvetica", 10, "bold"),
                  relief="flat", padx=12, pady=5, cursor="hand2").pack(side="left", padx=6)
        tk.Button(btnbar, text="📋 Open Logs",      command=self._open_logs,
                  bg="#27ae60", fg="white", font=("Helvetica", 10, "bold"),
                  relief="flat", padx=12, pady=5, cursor="hand2").pack(side="left", padx=6)
        tk.Button(btnbar, text="💾 Save Log",       command=self._save_log,
              bg="#16a085", fg="white", font=("Helvetica", 10, "bold"),
              relief="flat", padx=12, pady=5, cursor="hand2").pack(side="left", padx=6)

        # ── Main content area
        content = tk.Frame(self.root, bg="#1a1a2e")
        content.pack(fill="both", expand=True, padx=10, pady=8)

        # ── Left info panels
        left = tk.Frame(content, bg="#1a1a2e", width=340)
        left.pack(side="left", fill="y", padx=(0,10))
        left.pack_propagate(False)

        self.human_box  = self._info_panel(left, "👤 HUMAN INFO",  "#27ae60")
        self.animal_box = self._info_panel(left, "🐾 ANIMAL INFO", "#e67e22")
        self.object_box = self._info_panel(left, "📦 OBJECT INFO", "#e94560")
        self.shock_box  = self._info_panel(left, "⚡ SHOCK INFO",   "#e74c3c")

        # Dialog / console box
        self.dialog_box = tk.Text(left, height=6, bg="#000", fg="#00ff88",
                      insertbackground="#00ff88", font=("Courier", 9))
        self.dialog_box.pack(fill="x", pady=6)
        self.dialog_box.insert("end", "[System] Ready for detections...\n")
        self.dialog_box.configure(state="disabled")

        # Video canvas (right side)
        self.canvas = tk.Label(content, bg="#000", width=780, height=520,
                       text="No Video Source\nClick a button above to start",
                       font=("Helvetica", 14), fg="#555")
        self.canvas.pack(side="right", fill="both", expand=True)

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

        tk.Checkbutton(bottom, text="Auto WhatsApp", variable=self.auto_whatsapp_var,
                   bg="#16213e", fg="#aaa", selectcolor="#0f3460",
                   activebackground="#16213e", activeforeground="#fff",
                   font=("Helvetica", 9)).pack(side="left", padx=6)

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
        url = tk.simpledialog.askstring("DroidCam", "Enter DroidCam IP URL\n(e.g. http://192.168.1.5:4747/video)",
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
        self.source = source
        self.source_is_file = isinstance(source, str) and os.path.exists(source)
        self.cap = cv2.VideoCapture(source)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open source: {source}")
            return
        self.running = True
        self.failed_reads = 0
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
            ret, frame = self.cap.read()
            if not ret:
                if self.source_is_file:
                    # Loop video files for continuous detection
                    try:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.05)
                        continue
                    except Exception:
                        pass
                self.failed_reads += 1
                if self.failed_reads % 5 == 0:
                    self._set_status("⚠️ Camera read failed. Reconnecting...")
                    if self.cap:
                        self.cap.release()
                    self.cap = cv2.VideoCapture(self.source)
                if self.failed_reads >= self.max_failed_reads:
                    self._set_status("❌ Camera lost. Please restart source.")
                    self.running = False
                    break
                time.sleep(0.15)
                continue
            self.failed_reads = 0
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
        self.canvas.image = img

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

        # Shock panel
        if not self.shock_enabled:
            shock_txt = "Manual Override: SHOCK DISABLED"
        else:
            shock_events = [h for h in humans if h.get("shock")] + [a for a in animals if a.get("shock")]
            if shock_events:
                shock_txt = "\n".join([f"• {e['label']}\n  ➤ {e['action']}" for e in shock_events])
            else:
                shock_txt = "No shock events"
        self.shock_box.configure(text=shock_txt)

    def _handle_detections(self, humans, animals, objects):
        for h in humans:
            self.logger.log("Human", h["label"], h["action"], h["conf"])
            if self.shock_enabled and h["shock"]:
                self.alert.trigger_shock_alert(h["action"])
            self._auto_alert("Human", h)

        for a in animals:
            self.logger.log("Animal", a["label"], a["action"], a["conf"])
            if "Ultrasonic" in a["action"]:
                self.alert.trigger_ultrasonic()
            elif self.shock_enabled and a["shock"]:
                self.alert.trigger_shock_alert(a["action"])
            self._auto_alert("Animal", a)

        for o in objects:
            self.logger.log("Object", o["label"], o["action"], o["conf"])

    def _auto_alert(self, det_type, det_info):
        if not self.auto_whatsapp_var.get():
            return
        number = self.wa_entry.get().strip()
        if not number or "XXXX" in number:
            self._log_dialog("ℹ️ Auto WhatsApp disabled – add a valid number first.")
            return
        now = time.time()
        key = f"{det_type}:{det_info.get('label','')}"
        last = self._last_sent.get(key, 0)
        if now - last < self.alert_cooldown_sec:
            return

        self._last_sent[key] = now
        self.alert.whatsapp_number = number

        img_path = None
        bbox = det_info.get("bbox")
        if bbox and self.last_frame is not None:
            x1, y1, x2, y2 = bbox
            crop = self.last_frame[y1:y2, x1:x2]
            if crop.size > 0:
                os.makedirs("logs", exist_ok=True)
                img_path = self._save_detection_snapshot(det_type, det_info, crop)

        if img_path is None and self.last_frame is not None:
            full_frame = self.last_frame.copy()
            img_path = self._save_detection_snapshot(det_type, det_info, full_frame)

        msg = self._format_alert_message(det_type, det_info)
        self.alert.send_whatsapp(msg, img_path)
        self._log_dialog(f"📤 Auto WhatsApp sent: {det_info.get('label','')}")

    # ── CROP & SEND ───────────────────────────
    def _crop_and_send(self):
        if self.last_frame is None:
            messagebox.showinfo("Info", "No frame available yet.")
            return
        CropWindow(self.root, self.last_frame, self._send_cropped)

    def _send_cropped(self, cropped_img):
        detail = tk.simpledialog.askstring(
            "Alert Details",
            "Enter what was detected (e.g., Cow near gate):",
            initialvalue=self._last_manual_label or "",
            parent=self.root,
        )
        if detail is None:
            self._set_status("ℹ️ Crop canceled.")
            return
        detail = detail.strip() or "Manual capture"
        self._last_manual_label = detail
        save_path = f"logs/alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        os.makedirs("logs", exist_ok=True)
        annotated_path = self._save_annotated_crop(
            cropped_img,
            filename_prefix="alert",
            detail_text=detail,
            target_path=save_path,
        )
        self.alert.whatsapp_number = self.wa_entry.get().strip()
        msg = self._format_manual_message(detail)
        self.alert.send_whatsapp(msg, annotated_path)
        self._set_status(f"📨 WhatsApp alert queued. Image saved: {annotated_path}")
        self._log_dialog("📤 Manual WhatsApp alert queued – watch Chrome to confirm send.")

    # ── HELPERS ───────────────────────────────
    def _stop_buzzer(self):
        self.alert.stop_buzzer()
        self._set_status("🔇 Buzzer stopped.")

    def _save_log(self):
        log_path = os.path.abspath("logs/detection_log.csv")
        if not os.path.exists(log_path):
            messagebox.showinfo("Logs", "No log file found yet.")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="detection_log.csv"
        )
        if save_path:
            with open(log_path, "r") as src, open(save_path, "w", newline="") as dst:
                dst.write(src.read())
            self._log_dialog(f"✅ Log saved to: {save_path}")
            self._set_status("✅ Log saved.")

    def _manual_override(self):
        self.shock_enabled = False
        self._log_dialog("⚠️ Manual override enabled — shock disabled.")
        self._set_status("⚠️ Manual override active (shock disabled).")

    def _log_dialog(self, message):
        if not hasattr(self, "dialog_box"):
            print(message)
            return

        def append():
            self.dialog_box.configure(state="normal")
            self.dialog_box.insert("end", message + "\n")
            self.dialog_box.see("end")
            self.dialog_box.configure(state="disabled")

        self.root.after(0, append)

    def _save_annotated_crop(self, image, filename_prefix, detail_text, target_path=None):
        if target_path is None:
            target_path = f"logs/{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        annotated = image.copy()
        if detail_text:
            h, w = annotated.shape[:2]
            bar_h = max(30, int(0.15 * h))
            cv2.rectangle(annotated, (0, h - bar_h), (w, h), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                detail_text,
                (10, h - bar_h//2),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.5, min(1.0, w / 800)),
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )
        cv2.imwrite(target_path, annotated)
        return target_path

    def _save_detection_snapshot(self, det_type, det_info, image):
        os.makedirs("logs", exist_ok=True)
        detail_text = self._describe_detection(det_type, det_info)
        return self._save_annotated_crop(
            image,
            filename_prefix="auto_alert",
            detail_text=detail_text,
        )

    def _describe_detection(self, det_type, det_info):
        if det_type == "Human":
            rule_label = str(det_info.get("rule_key", "Human")).replace("_", " ").title()
            age = det_info.get("age", "?")
            action = det_info.get("action", "Alert")
            return f"{rule_label} | Age {age} | {action}"
        if det_type == "Animal":
            species = det_info.get("species") or det_info.get("label", "Animal")
            action = det_info.get("action", "Alert")
            return f"{species.title()} | {action}"
        return det_info.get("label", "Detection")

    def _format_alert_message(self, det_type, det_info):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = "Human Info:" if det_type == "Human" else "Animal Info:" if det_type == "Animal" else f"{det_type} Info:"
        lines = ["🔴 Cropped Detection Alert!", header]

        if det_type == "Human":
            profile = str(det_info.get("rule_key", "Human")).replace("_", " ").title()
            age = det_info.get("age", "?")
            gender = str(det_info.get("gender", "Unknown")).title()
            lines.append(f"Detected: {profile} ({gender}, Age {age})")
        elif det_type == "Animal":
            species = det_info.get("species", det_info.get("label", "Animal")).title()
            conf = det_info.get("conf")
            conf_txt = f" | Conf {conf:.2f}" if isinstance(conf, (int, float)) else ""
            lines.append(f"Detected: {species}{conf_txt}")
        else:
            lines.append(f"Detected: {det_info.get('label', 'Object')}")

        action = det_info.get("action")
        if action:
            lines.append(f"Action: {action}")

        current = det_info.get("current_uA")
        if current:
            energy = self._estimate_joules(current)
            if energy is not None:
                lines.append(f"Voltage: {int(current)}V, {energy:.2f} J")

        lines.append(f"Time: {timestamp}")
        lines.append("")
        lines.append(f"What to do: {self._build_instruction_text(det_info)}")
        return "\n".join(lines)

    def _build_instruction_text(self, det_info):
        if det_info.get("shock"):
            return "Apply shock as per voltage mentioned above."
        action = det_info.get("action", "Monitor the area.")
        if action and "ultrasonic" in action.lower():
            return "Trigger ultrasonic deterrent immediately."
        return action or "Monitor the area."

    def _format_manual_message(self, detail):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "🔴 Cropped Detection Alert!",
            "Manual Info:",
            f"Details: {detail}",
            f"Time: {timestamp}",
            "",
            "What to do: Act based on the operator note above.",
        ]
        return "\n".join(lines)

    def _estimate_joules(self, current_uA):
        try:
            return round(float(current_uA) * 0.00024, 2)
        except (TypeError, ValueError):
            return None

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
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start[0], self.start[1], e.x, e.y,
            outline="#e94560", width=2)

    def _on_release(self, e):
        self.end = (e.x, e.y)

    def _on_confirm(self, e):
        if self.start and self.end:
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
