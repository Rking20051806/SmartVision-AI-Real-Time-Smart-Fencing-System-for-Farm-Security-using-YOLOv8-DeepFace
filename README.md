# 🛡 Smart Fencing System – Complete Setup Guide

**Real-Time Detection Based on Smart Fencing System for Farm Security**  
*B.Tech Minor Project – Rohan Nandanwar, DMIHER (DU)*

[![Open in HuggingFace Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/rking2005/SmartVision-Farm-Security)

---

## 🌐 Live Web Demo

Try the web version: **[Launch SmartVision AI Web App](https://huggingface.co/spaces/rking2005/SmartVision-Farm-Security)**

### Deploy to HuggingFace Spaces (Free)
1. Fork this repository
2. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
3. Select **Gradio** as the SDK
4. Upload `app.py`, `requirements.txt`, and all `yolov8*.pt` model files
5. The Space will build and deploy automatically!

---

## 📋 What This System Does

| Feature | Detail |
|---|---|
| **Human Detection** | YOLOv8 detects people + DeepFace estimates age & gender |
| **Animal Detection** | YOLOv8 + custom Animal-10 CNN classifies 10 species |
| **Smart Deterrence** | Rule-based engine applies correct shock/buzzer per intruder |
| **GUI Dashboard** | Live video feed with detection info panels (Tkinter) |
| **Alerts** | WhatsApp notifications with cropped intruder photo |
| **Logging** | Every detection saved to `logs/detection_log.csv` |

---

## 🖥 System Requirements

- Python **3.9 – 3.11** (recommended: 3.10)
- Windows 10/11, Ubuntu 20.04+, or macOS 12+
- RAM: minimum 4 GB (8 GB recommended)
- **No GPU required** – runs on CPU

---

## ⚡ Quick Start (5 Steps)

### Step 1 – Clone / Download the project
```
smart_fencing/
├── main.py                  ← Main application
├── train_animal_cnn.py      ← CNN training script
├── requirements.txt         ← All dependencies
├── animal10.h5              ← (Generated after training)
├── preview.mp3              ← (Add your buzzer audio here)
└── logs/                    ← Auto-created, stores logs & alerts
```

### Step 2 – Create Python virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3 – Install dependencies
```bash
pip install -r requirements.txt
```
> ⏳ This may take 5–10 minutes (TensorFlow and OpenCV are large)

### Step 4 – Train the Animal CNN (optional but recommended)
1. Download the Animals-10 dataset from Kaggle:  
   👉 https://www.kaggle.com/datasets/alessiocorrado99/animals10
2. Extract it so the folder `raw-img/` is inside `smart_fencing/`
3. Run:
```bash
python train_animal_cnn.py
```
4. After training, `animal10.h5` is created automatically.

> **Skip this step** if you only want human detection (YOLOv8 works out of the box)

### Step 5 – Run the application
```bash
python main.py
```

---

## 🎮 Using the Application

### Video Sources
| Button | Use When |
|---|---|
| 📷 Laptop Camera | Testing with your built-in webcam |
| 📱 DroidCam | Using your phone as an IP camera (install DroidCam app) |
| 🎬 Load Video | Testing with a pre-recorded video file |
| 🖼 Load Image | Detecting in a single photo |

### DroidCam Setup
1. Install **DroidCam** app on your Android phone
2. Connect phone and PC to the same WiFi
3. Open the app – note the IP address shown (e.g. `192.168.1.5`)
4. In SmartVision, click **DroidCam** and enter: `http://192.168.1.5:4747/video`

### WhatsApp Alerts
1. Enter your number (with country code) in the WhatsApp field, e.g. `+919876543210`
2. After a detection, click **✂ Crop & Send**
3. Draw a selection box around the intruder
4. Press **Enter** → WhatsApp Web will open and send the alert automatically

---

## 🔬 Shock Rules Reference

| Category | Shock Allowed | Current | Action |
|---|---|---|---|
| Adult Male | ✅ | 4000 µA | Shock Deterrence |
| Adult Female | ✅ | 2500 µA | Shock Deterrence |
| Child | ❌ | — | Safety Override |
| Unknown Human | ❌ | — | Alert Only |
| Cow / Horse | ✅ | 2500 µA | Shock Deterrence |
| Dog | ✅ | 1800 µA | Shock Deterrence |
| Elephant / Bear / Giraffe | ✅ | 4000 µA | Shock Deterrence |
| Bird / Chicken / Squirrel | ❌ | — | Ultrasonic Buzzer |
| Butterfly / Spider | ❌ | — | No Action |

> ⚠️ **Note:** Currents listed are for reference/simulation only. In real hardware deployment, verify against IEC 60479-1 and IEC 60335-2-76 safety standards.

---

## 🐾 Animal CNN – Supported Species

`butterfly`, `cat`, `chicken`, `cow`, `dog`, `elephant`, `horse`, `sheep`, `spider`, `squirrel`

---

## 🛠 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: ultralytics` | Run `pip install ultralytics` |
| `ModuleNotFoundError: deepface` | Run `pip install deepface` |
| Camera not opening | Try changing `0` to `1` or `2` in webcam source |
| DeepFace is slow | Normal on CPU – it analyzes every 8th frame by default |
| WhatsApp not sending | Make sure you're logged into WhatsApp Web in Chrome |
| `animal10.h5` not found | Run `train_animal_cnn.py` first, or skip for human-only detection |

---

## 📁 Output Files

- `logs/detection_log.csv` – Complete log of all detection events
- `logs/alert_YYYYMMDD_HHMMSS.jpg` – Saved cropped alert images

---

## 🔮 Future Improvements (from report)

- [ ] Field-test in diverse weather conditions
- [ ] Improve CNN accuracy using Transfer Learning (MobileNetV2 / EfficientNet)
- [ ] Add bounding box visualization to GUI
- [ ] Mobile app + cloud dashboard
- [ ] Real hardware integration (relay module for actual fence control)

---

## 📚 Key References

- Delwar et al. (2025) – YOLOv8 + IoT for animal intrusion detection
- YOLOv8 by Ultralytics – https://github.com/ultralytics/ultralytics
- DeepFace – https://github.com/serengil/deepface
- Animals-10 Dataset – https://www.kaggle.com/datasets/alessiocorrado99/animals10
