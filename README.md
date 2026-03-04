<p align="center">
  <img src="https://img.shields.io/badge/YOLOv8-Object%20Detection-blue?style=for-the-badge&logo=yolo" alt="YOLOv8"/>
  <img src="https://img.shields.io/badge/DeepFace-Face%20Analysis-green?style=for-the-badge" alt="DeepFace"/>
  <img src="https://img.shields.io/badge/TensorFlow-CNN-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Gradio-Web%20App-yellow?style=for-the-badge&logo=gradio" alt="Gradio"/>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
</p>

<h1 align="center">🛡️ SmartVision AI — Real-Time Smart Fencing System<br/>for Farm Security</h1>

<p align="center">
  <strong>An intelligent perimeter defense system using YOLOv8 & DeepFace for real-time human and animal intrusion detection with automated, rule-based deterrence.</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/rking2005/SmartVision-Farm-Security">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Hugging%20Face%20Spaces-blue?style=for-the-badge" alt="Live Demo"/>
  </a>
  &nbsp;
  <a href="https://github.com/Rking20051806/SmartVision-AI-Real-Time-Smart-Fencing-System-for-Farm-Security-using-YOLOv8-DeepFace/issues">
    <img src="https://img.shields.io/github/issues/Rking20051806/SmartVision-AI-Real-Time-Smart-Fencing-System-for-Farm-Security-using-YOLOv8-DeepFace?style=for-the-badge" alt="Issues"/>
  </a>
  <a href="https://github.com/Rking20051806/SmartVision-AI-Real-Time-Smart-Fencing-System-for-Farm-Security-using-YOLOv8-DeepFace/stargazers">
    <img src="https://img.shields.io/github/stars/Rking20051806/SmartVision-AI-Real-Time-Smart-Fencing-System-for-Farm-Security-using-YOLOv8-DeepFace?style=for-the-badge" alt="Stars"/>
  </a>
  <a href="https://github.com/Rking20051806/SmartVision-AI-Real-Time-Smart-Fencing-System-for-Farm-Security-using-YOLOv8-DeepFace/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/Rking20051806/SmartVision-AI-Real-Time-Smart-Fencing-System-for-Farm-Security-using-YOLOv8-DeepFace?style=for-the-badge" alt="License"/>
  </a>
</p>

---

## 🌐 Live Demo

> **Try it now — no installation required!**

### 🔗 [Launch SmartVision AI Web App →](https://huggingface.co/spaces/rking2005/SmartVision-Farm-Security)

The web app supports **image upload**, **webcam streaming**, and **video analysis** — all running in-browser on Hugging Face Spaces.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Smart Shock Rules Engine](#-smart-shock-rules-engine)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Web App (Gradio)](#-web-app-gradio)
- [Desktop App (Tkinter)](#-desktop-app-tkinter)
- [Animal CNN Training](#-animal-cnn-training)
- [Demo Mode](#-demo-mode)
- [Screenshots](#-screenshots)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [References](#-references)
- [Author](#-author)

---

## 🧠 Overview

**SmartVision AI** is an end-to-end intelligent farm security system that combines state-of-the-art computer vision models to detect, classify, and respond to intrusions along farm perimeters in real time.

The system identifies **humans** (with age & gender estimation via DeepFace) and **animals** (via YOLOv8 + a custom Animal-10 CNN), then applies a **rule-based smart deterrence engine** that selects the appropriate response — from harmless ultrasonic buzzers for small creatures to calibrated electrical deterrents for large animals — while implementing **automatic safety overrides** for children and vulnerable entities.

### 🎯 Problem Statement

Traditional electric fences are indiscriminate — they deliver the same shock regardless of what touches them. This poses serious risks to children, small animals, and protected wildlife while being unnecessarily aggressive against harmless creatures.

### 💡 Solution

SmartVision AI replaces brute-force deterrence with **intelligence-driven, proportional response** by detecting and classifying intruders before deciding the action.

---

## ✨ Key Features

| Feature | Description |
|:--------|:------------|
| **🧑 Human Detection & Classification** | YOLOv8 detects people; DeepFace estimates age & gender to differentiate adults, children, males, and females |
| **🐾 Animal Detection (10+ Species)** | YOLOv8 + custom CNN classifies: cow, dog, horse, sheep, elephant, bear, cat, bird, chicken, squirrel, butterfly, spider |
| **⚡ Smart Shock Rules Engine** | Proportional current delivery (0–4000 µA) based on intruder type with child safety override |
| **🔊 Multi-Modal Alerts** | Buzzer/ultrasonic alerts, WhatsApp notifications with cropped intruder images |
| **🖥️ Desktop GUI Dashboard** | Full-featured Tkinter app with live video feed, detection panels, model selector, and log viewer |
| **🌐 Web Application** | Gradio-powered web app deployable on Hugging Face Spaces with image/video/webcam support |
| **📊 Detection Logging** | All events logged to CSV with timestamp, type, label, action, and confidence |
| **🔄 Multiple YOLOv8 Models** | Switch between YOLOv8n / s / m / l / x at runtime for speed vs. accuracy tradeoff |
| **📱 DroidCam Integration** | Use your smartphone as an IP camera for live monitoring |
| **🎮 Demo Mode** | Test the full GUI and rules engine without ML dependencies |

---

## 🏗 System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         INPUT SOURCES                            │
│   📷 Webcam  │  📱 DroidCam  │  🎬 Video File  │  🖼 Image     │
└──────────────┬───────────────┬────────────────┬─────────────────┘
               │               │                │
               ▼               ▼                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    YOLOv8 OBJECT DETECTION                       │
│              (n / s / m / l / x model variants)                  │
│         Detects: persons, animals, and 80 COCO classes           │
└──────────────┬───────────────────────────────┬───────────────────┘
               │                               │
     ┌─────────▼─────────┐          ┌──────────▼──────────┐
     │  HUMAN PIPELINE   │          │  ANIMAL PIPELINE    │
     │                   │          │                     │
     │  DeepFace:        │          │  YOLO class map +   │
     │  • Age estimation │          │  Animal-10 CNN      │
     │  • Gender detect  │          │  (10 species)       │
     │  • Multi-backend  │          │                     │
     │    (RetinaFace,   │          │                     │
     │     MTCNN, SSD)   │          │                     │
     └─────────┬─────────┘          └──────────┬──────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                  SMART SHOCK RULES ENGINE                        │
│                                                                  │
│   Input: intruder_type → Output: {shock, current_µA, action}    │
│   • Child safety override (0 µA)                                │
│   • Proportional current by body mass / threat level            │
│   • Ultrasonic/buzzer for small animals                         │
│   • No action for harmless insects                              │
└──────────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│                      RESPONSE LAYER                              │
│  🔊 Buzzer Alert  │  📱 WhatsApp  │  📊 CSV Log  │  🖥 GUI     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Component | Technology |
|:----------|:-----------|
| Object Detection | [YOLOv8](https://github.com/ultralytics/ultralytics) (Ultralytics) |
| Face / Age / Gender | [DeepFace](https://github.com/serengil/deepface) (RetinaFace, MTCNN, SSD backends) |
| Animal Classification | Custom CNN trained on [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) dataset |
| Deep Learning Framework | TensorFlow / Keras |
| Web Application | [Gradio](https://gradio.app/) |
| Desktop GUI | Tkinter + PIL |
| Alerts | PyWhatKit (WhatsApp), Pygame (audio buzzer) |
| Language | Python 3.9+ |
| Deployment | Hugging Face Spaces |

---

## ⚡ Smart Shock Rules Engine

The core intelligence of the system — a rule-based engine that maps each detected intruder type to a proportional, safe response:

### 👤 Humans

| Category | Shock | Current | Action |
|:---------|:-----:|--------:|:-------|
| Adult Male | ✅ | 4,000 µA | Shock Deterrence |
| Adult Female | ✅ | 2,500 µA | Shock Deterrence |
| Child | ❌ | 0 µA | **Safety Override** — No shock |
| Unknown Human | ❌ | 0 µA | Alert Only |

### 🐾 Animals

| Category | Shock | Current | Action |
|:---------|:-----:|--------:|:-------|
| Elephant / Bear / Giraffe | ✅ | 4,000 µA | Shock Deterrence |
| Cow / Horse / Zebra | ✅ | 2,500 µA | Shock Deterrence |
| Sheep | ✅ | 2,000 µA | Shock Deterrence |
| Dog | ✅ | 1,800 µA | Shock Deterrence |
| Cat | — | 1,500 µA | Shock Deterrence |
| Bird / Chicken / Squirrel | ❌ | 0 µA | Ultrasonic / Buzzer |
| Butterfly / Spider | ❌ | 0 µA | No Action |

> ⚠️ **Disclaimer:** All current values are for simulation/research purposes. Real hardware deployment must comply with **IEC 60479-1** and **IEC 60335-2-76** safety standards.

---

## 📁 Project Structure

```
SmartVision-AI/
│
├── app.py                   # Gradio web application (HF Spaces deployment)
├── main.py                  # Desktop GUI application (Tkinter)
├── demo_mode.py             # Demo mode — no ML dependencies required
├── train_animal_cnn.py      # Animal-10 CNN training script
│
├── requirements.txt         # Python dependencies
├── runtime.txt              # Python runtime version for deployment
│
├── yolov8n.pt               # YOLOv8 Nano   (fastest, least accurate)
├── yolov8s.pt               # YOLOv8 Small
├── yolov8m.pt               # YOLOv8 Medium  (balanced)
├── yolov8l.pt               # YOLOv8 Large
├── yolov8x.pt               # YOLOv8 XLarge  (most accurate, slowest)
│
├── assets/
│   └── smartvision_buzzer.wav   # Alert buzzer sound
│
├── logs/
│   └── detection_log.csv    # Auto-generated detection logs
│
└── README.md                # This file
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9 – 3.11** (recommended: 3.10)
- **OS:** Windows 10/11, Ubuntu 20.04+, or macOS 12+
- **RAM:** 4 GB minimum (8 GB recommended)
- **GPU:** Optional — runs fully on CPU

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Rking20051806/SmartVision-AI-Real-Time-Smart-Fencing-System-for-Farm-Security-using-YOLOv8-DeepFace.git
cd SmartVision-AI-Real-Time-Smart-Fencing-System-for-Farm-Security-using-YOLOv8-DeepFace

# 2. Create a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> ⏳ Installation may take 5–10 minutes due to TensorFlow and OpenCV.

---

## 🌐 Web App (Gradio)

The web version is deployed on **Hugging Face Spaces** and can also be run locally:

```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

**Features:**
- 📸 **Image Upload** — Upload any image for instant detection
- 📹 **Webcam Streaming** — Real-time detection from your browser
- 🎬 **Video Analysis** — Upload and analyze video files
- 📋 **Shock Rules Table** — View all deterrence rules
- 📊 **Detection Log** — Download detection history
- 🔧 **System Status** — Check model diagnostics
- 🔄 **Model Selector** — Switch between YOLOv8 variants

---

## 🖥 Desktop App (Tkinter)

The full-featured desktop application with hardware integration:

```bash
python main.py
```

**Desktop-Exclusive Features:**
- 📷 Live camera feed with bounding box overlays
- 📱 DroidCam IP camera support
- 🔊 Audio buzzer alerts via Pygame
- 📱 WhatsApp notifications with cropped intruder images (via PyWhatKit)
- ✂️ Crop & Send — draw a selection and send via WhatsApp
- 📋 Built-in log viewer

### 📱 DroidCam Setup

1. Install the **DroidCam** app on your Android phone
2. Connect phone and PC to the **same WiFi network**
3. Open the app and note the IP address (e.g., `192.168.1.5`)
4. In SmartVision, click **DroidCam** and enter: `http://192.168.1.5:4747/video`

---

## 🐾 Animal CNN Training

Optionally train a custom CNN for enhanced animal classification:

```bash
# 1. Download the Animals-10 dataset from Kaggle:
#    https://www.kaggle.com/datasets/alessiocorrado99/animals10

# 2. Extract dataset so 'raw-img/' folder is in the project root

# 3. Train the model
python train_animal_cnn.py
```

**Training Config:**
- Input size: 128 × 128
- Batch size: 32
- Epochs: 30 (with early stopping)
- Output: `animal10.h5`
- Classes: `butterfly`, `cat`, `chicken`, `cow`, `dog`, `elephant`, `horse`, `sheep`, `spider`, `squirrel`

> **Note:** Skip this step if you only need human detection — YOLOv8 works out of the box.

---

## 🎮 Demo Mode

Test the complete GUI and rules engine **without any ML dependencies**:

```bash
python demo_mode.py
```

Demo mode generates synthetic farm scenes with simulated detections, cycling through all intruder types to demonstrate the shock rules engine, logging, and UI.

---

## 📸 Screenshots

| Web App (Gradio) | Desktop App (Tkinter) |
|:-:|:-:|
| Image / Video / Webcam detection with full report | Live camera feed with detection panels |

> *Deploy the [live demo](https://huggingface.co/spaces/rking2005/SmartVision-Farm-Security) to see the web interface in action.*

---

## 🔧 Troubleshooting

| Problem | Solution |
|:--------|:---------|
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics` |
| `ModuleNotFoundError: deepface` | `pip install deepface` |
| Camera not opening | Change webcam index from `0` to `1` or `2` |
| DeepFace is slow | Normal on CPU — analyzes every Nth frame by default |
| WhatsApp not sending | Ensure you're logged into WhatsApp Web in Chrome |
| `animal10.h5` not found | Run `python train_animal_cnn.py` or skip for human-only detection |
| `torch.load` error on PyTorch 2.6+ | The app auto-patches `weights_only=False`; ensure PyTorch < 2.6 or use the patched `app.py` |
| Hugging Face build fails | Verify `requirements.txt` pins are compatible; check Space build logs |

---

## 🗺 Roadmap

- [ ] Field-test under diverse weather and lighting conditions
- [ ] Transfer Learning with MobileNetV2 / EfficientNet for improved animal accuracy
- [ ] Mobile companion app with push notifications
- [ ] Cloud dashboard for multi-farm monitoring
- [ ] Real hardware integration (relay module for actual fence control)
- [ ] Night vision / IR camera support
- [ ] Multi-camera zone management

---

## 📚 References

- Delwar et al. (2025) — *YOLOv8 + IoT for Animal Intrusion Detection*
- **YOLOv8** by Ultralytics — [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **DeepFace** — [github.com/serengil/deepface](https://github.com/serengil/deepface)
- **Animals-10 Dataset** — [kaggle.com/datasets/alessiocorrado99/animals10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
- **Gradio** — [gradio.app](https://gradio.app/)
- IEC 60479-1 — *Effects of Current on Human Beings and Livestock*
- IEC 60335-2-76 — *Safety of Electric Fence Energizers*

---

## 👨‍💻 Author

**Rohan Nandanwar**  
B.Tech Minor Project — DMIHER (DU)

---

<p align="center">
  <strong>⭐ Star this repository if you found it useful!</strong><br/><br/>
  <a href="https://huggingface.co/spaces/rking2005/SmartVision-Farm-Security">
    <img src="https://img.shields.io/badge/Try%20Live%20Demo-%F0%9F%9A%80-brightgreen?style=for-the-badge" alt="Try Live Demo"/>
  </a>
</p>
