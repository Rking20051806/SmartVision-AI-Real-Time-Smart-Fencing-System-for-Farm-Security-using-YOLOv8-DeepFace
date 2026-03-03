"""
SmartVision – Demo Mode (No ML libraries needed)
Tests the GUI, shock rules engine, and logging
without requiring YOLOv8 or TensorFlow.

Run: python demo_mode.py
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import csv
import os
import random
from datetime import datetime
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np

# ─────────────────────────────────────────────
#  SHOCK RULES (same as main.py)
# ─────────────────────────────────────────────
SHOCK_RULES = {
    "adult_male":    {"current_uA": 4000, "action": "4000 µA Shock Deterrence", "color": "#e74c3c"},
    "adult_female":  {"current_uA": 2500, "action": "2500 µA Shock Deterrence", "color": "#e67e22"},
    "child":         {"current_uA": 0,    "action": "No Shock – Safety Override", "color": "#27ae60"},
    "unknown_human": {"current_uA": 0,    "action": "Alert Only",                "color": "#3498db"},
    "cow":           {"current_uA": 2500, "action": "2500 µA Shock Deterrence", "color": "#e67e22"},
    "dog":           {"current_uA": 1800, "action": "1800 µA Shock Deterrence", "color": "#e67e22"},
    "elephant":      {"current_uA": 4000, "action": "4000 µA Shock Deterrence", "color": "#e74c3c"},
    "bird":          {"current_uA": 0,    "action": "Ultrasonic / Buzzer",       "color": "#9b59b6"},
    "chicken":       {"current_uA": 0,    "action": "Ultrasonic / Buzzer",       "color": "#9b59b6"},
    "cat":           {"current_uA": 1500, "action": "1500 µA Shock Deterrence", "color": "#e67e22"},
    "sheep":         {"current_uA": 2000, "action": "2000 µA Shock Deterrence", "color": "#e67e22"},
    "horse":         {"current_uA": 2500, "action": "2500 µA Shock Deterrence", "color": "#e67e22"},
}

# Demo detections to simulate
DEMO_EVENTS = [
    {"type": "Human",  "key": "adult_male",   "label": "Person | Age:32 Gender:Male"},
    {"type": "Human",  "key": "adult_female", "label": "Person | Age:28 Gender:Female"},
    {"type": "Human",  "key": "child",        "label": "Person | Age:10 Gender:Unknown"},
    {"type": "Animal", "key": "cow",          "label": "Cow | Conf:0.94"},
    {"type": "Animal", "key": "dog",          "label": "Dog | Conf:0.99"},
    {"type": "Animal", "key": "chicken",      "label": "Chicken | Conf:1.00"},
    {"type": "Animal", "key": "elephant",     "label": "Elephant | Conf:0.87"},
    {"type": "Animal", "key": "bird",         "label": "Bird | Conf:0.95"},
]

# ─────────────────────────────────────────────
#  DEMO FRAME GENERATOR
# ─────────────────────────────────────────────
def make_demo_frame(width=640, height=480, event=None):
    """Generate a synthetic farm-like demo frame."""
    # Green field background
    img = Image.new("RGB", (width, height), "#2d5a27")
    draw = ImageDraw.Draw(img)

    # Sky
    for y in range(height // 3):
        ratio = y / (height // 3)
        r = int(135 * ratio + 30 * (1 - ratio))
        g = int(206 * ratio + 100 * (1 - ratio))
        b = int(235 * ratio + 180 * (1 - ratio))
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Ground gradient
    for y in range(height // 3, height):
        ratio = (y - height // 3) / (height * 2 // 3)
        g_val = int(90 - 30 * ratio)
        draw.line([(0, y), (width, y)], fill=(34, g_val, 20))

    # Fence posts
    for x in range(0, width, 80):
        draw.rectangle([x, height // 2, x + 8, height - 20], fill="#8B4513")
    # Fence wire
    draw.line([(0, height // 2 + 20), (width, height // 2 + 20)], fill="#aaa", width=2)
    draw.line([(0, height // 2 + 50), (width, height // 2 + 50)], fill="#aaa", width=2)

    # Tree
    draw.rectangle([50, height // 3, 65, height // 2], fill="#5D4037")
    draw.ellipse([20, height // 4, 100, height // 2], fill="#2e7d32")

    # Detection overlay
    if event:
        rule = SHOCK_RULES.get(event["key"], {})
        color_hex = rule.get("color", "#ffffff")
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)

        # Bounding box simulation
        cx, cy = width // 2 + random.randint(-80, 80), height // 2 + random.randint(-20, 60)
        bw, bh = 120, 160
        x1, y1 = cx - bw // 2, cy - bh // 2
        x2, y2 = cx + bw // 2, cy + bh // 2

        # Draw box
        for i in range(3):
            draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=(r, g, b))

        # Label background
        draw.rectangle([x1, y1 - 22, x1 + 200, y1], fill=(r, g, b))
        draw.text((x1 + 4, y1 - 18), event["label"], fill="white")

        # Action text
        action = rule.get("action", "")
        draw.rectangle([x1, y2, x1 + 230, y2 + 20], fill=(0, 0, 0, 180))
        draw.text((x1 + 4, y2 + 2), f"➤ {action}", fill="white")

    # Timestamp
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    draw.rectangle([0, height - 22, 220, height], fill=(0, 0, 0))
    draw.text((4, height - 18), ts, fill="#00ff88")

    return np.array(img)


# ─────────────────────────────────────────────
#  DEMO APP
# ─────────────────────────────────────────────
class DemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SmartVision – DEMO MODE")
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("1200x720")

        self.running = False
        self.current_event = None
        self.log_path = "logs/detection_log.csv"
        os.makedirs("logs", exist_ok=True)

        # Init CSV
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(["Timestamp","Type","Label","Action","Confidence"])

        self._build_ui()
        self._start_demo()

    def _build_ui(self):
        # Top bar
        topbar = tk.Frame(self.root, bg="#16213e", height=50)
        topbar.pack(fill="x")
        tk.Label(topbar, text="🛡 SmartVision – DEMO MODE",
                 font=("Helvetica", 16, "bold"), bg="#16213e", fg="#e94560").pack(side="left", padx=15, pady=8)
        tk.Label(topbar, text="[Simulated detections – no camera required]",
                 font=("Helvetica", 10), bg="#16213e", fg="#888").pack(side="left")
        self.log_lbl = tk.Label(topbar, text="● Logs Active", bg="#16213e",
                                fg="#00ff88", font=("Helvetica", 10, "bold"))
        self.log_lbl.pack(side="right", padx=15)

        # Btn bar
        btnbar = tk.Frame(self.root, bg="#0f3460", pady=6)
        btnbar.pack(fill="x")
        s = {"bg":"#e94560","fg":"white","font":("Helvetica",10,"bold"),"relief":"flat","padx":12,"pady":5,"cursor":"hand2"}
        tk.Button(btnbar, text="▶ Start Simulation", command=self._start_demo, **s).pack(side="left", padx=6)
        tk.Button(btnbar, text="■ Stop",             command=self._stop_demo,
                  bg="#555", fg="white", font=("Helvetica",10,"bold"), relief="flat", padx=12, pady=5).pack(side="left", padx=6)
        tk.Button(btnbar, text="📋 View Logs",       command=self._view_logs,
                  bg="#27ae60", fg="white", font=("Helvetica",10,"bold"), relief="flat", padx=12, pady=5).pack(side="left", padx=6)

        # Demo event buttons
        tk.Label(btnbar, text="  Trigger:", bg="#0f3460", fg="#aaa",
                 font=("Helvetica", 10)).pack(side="left", padx=(20, 4))
        for ev in ["cow", "dog", "chicken", "adult_male", "child", "elephant"]:
            tk.Button(btnbar, text=ev.replace("_", " ").title(),
                      command=lambda e=ev: self._trigger_event(e),
                      bg="#8e44ad", fg="white", font=("Helvetica", 9, "bold"),
                      relief="flat", padx=8, pady=5, cursor="hand2").pack(side="left", padx=3)

        # Content
        content = tk.Frame(self.root, bg="#1a1a2e")
        content.pack(fill="both", expand=True, padx=10, pady=8)

        self.canvas = tk.Label(content, bg="#000", width=640, height=480)
        self.canvas.pack(side="left")

        right = tk.Frame(content, bg="#1a1a2e", width=310)
        right.pack(side="right", fill="y", padx=(12,0))
        right.pack_propagate(False)

        self.human_box  = self._panel(right, "👤 HUMAN INFO",  "#27ae60")
        self.animal_box = self._panel(right, "🐾 ANIMAL INFO", "#e67e22")
        self.event_box  = self._panel(right, "📋 LAST EVENT",  "#3498db")

        # Stats
        stats = tk.LabelFrame(right, text="📊 Detection Stats", font=("Helvetica",10,"bold"),
                               bg="#16213e", fg="#e94560", bd=1, relief="solid")
        stats.pack(fill="x", pady=6)
        self.stat_human  = self._stat_row(stats, "Humans Detected:",  "#27ae60")
        self.stat_animal = self._stat_row(stats, "Animals Detected:", "#e67e22")
        self.stat_shock  = self._stat_row(stats, "Shocks Triggered:", "#e74c3c")
        self.stat_buzz   = self._stat_row(stats, "Buzzers Triggered:","#9b59b6")
        self.counts = {"human": 0, "animal": 0, "shock": 0, "buzz": 0}

        # Bottom
        bottom = tk.Frame(self.root, bg="#16213e", pady=5)
        bottom.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="Ready – Press Start Simulation")
        tk.Label(bottom, textvariable=self.status_var, bg="#16213e",
                 fg="#aaa", font=("Helvetica", 9)).pack(side="left", padx=12)

    def _panel(self, parent, title, color):
        f = tk.LabelFrame(parent, text=title, font=("Helvetica",10,"bold"),
                          bg="#16213e", fg=color, bd=1, relief="solid", padx=8, pady=5)
        f.pack(fill="x", pady=5)
        lbl = tk.Label(f, text="No detections", font=("Courier",9),
                       bg="#16213e", fg="#ccc", justify="left", wraplength=280)
        lbl.pack(anchor="w")
        return lbl

    def _stat_row(self, parent, label, color):
        row = tk.Frame(parent, bg="#16213e")
        row.pack(fill="x", padx=8, pady=2)
        tk.Label(row, text=label, bg="#16213e", fg="#aaa", font=("Helvetica",9)).pack(side="left")
        val = tk.Label(row, text="0", bg="#16213e", fg=color, font=("Helvetica",11,"bold"))
        val.pack(side="right")
        return val

    def _start_demo(self):
        if self.running:
            return
        self.running = True
        self.status_var.set("▶ Simulation running...")
        threading.Thread(target=self._demo_loop, daemon=True).start()

    def _stop_demo(self):
        self.running = False
        self.status_var.set("⏹ Simulation stopped.")

    def _demo_loop(self):
        frame_num = 0
        while self.running:
            # Every 3 seconds simulate a new detection
            event = None
            if frame_num % 90 == 0 and frame_num > 0:
                event = random.choice(DEMO_EVENTS)
                self.current_event = event
                self.root.after(0, self._show_detection, event)
            elif frame_num % 90 > 30:
                event = self.current_event  # Keep showing for a few seconds

            frame = make_demo_frame(event=event)
            self.root.after(0, self._show_frame, frame)
            frame_num += 1
            time.sleep(0.033)

    def _trigger_event(self, key):
        """Manually trigger a specific detection."""
        for ev in DEMO_EVENTS:
            if ev["key"] == key:
                self.root.after(0, self._show_detection, ev)
                break
        else:
            # For keys not in DEMO_EVENTS, build on the fly
            ev = {"type": "Animal" if key not in ("adult_male","adult_female","child") else "Human",
                  "key": key,
                  "label": f"{key.replace('_',' ').title()} | Demo"}
            self.root.after(0, self._show_detection, ev)

    def _show_frame(self, frame):
        from PIL import Image, ImageTk
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        self.canvas.configure(image=img)
        self.canvas.image = img

    def _show_detection(self, event):
        rule = SHOCK_RULES.get(event["key"], {"action": "Alert", "current_uA": 0, "color": "#fff"})
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if event["type"] == "Human":
            self.human_box.configure(
                text=f"• {event['label']}\n  ➤ {rule['action']}")
            self.animal_box.configure(text="No animals detected")
            self.counts["human"] += 1
            self.stat_human.configure(text=str(self.counts["human"]))
        else:
            self.animal_box.configure(
                text=f"• {event['label']}\n  ➤ {rule['action']}")
            self.human_box.configure(text="No humans detected")
            self.counts["animal"] += 1
            self.stat_animal.configure(text=str(self.counts["animal"]))

        if rule["current_uA"] > 0:
            self.counts["shock"] += 1
            self.stat_shock.configure(text=str(self.counts["shock"]))
        if "Ultrasonic" in rule["action"]:
            self.counts["buzz"] += 1
            self.stat_buzz.configure(text=str(self.counts["buzz"]))

        self.event_box.configure(
            text=f"Time: {ts}\nType: {event['type']}\nLabel: {event['label']}\nAction: {rule['action']}")

        self.status_var.set(f"⚡ Detection: {event['label']} → {rule['action']}")

        # Log it
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([ts, event["type"], event["label"],
                                    rule["action"], f"{random.uniform(0.85,0.99):.2f}"])

    def _view_logs(self):
        win = tk.Toplevel(self.root)
        win.title("Detection Logs")
        win.geometry("820x450")
        win.configure(bg="#16213e")
        tk.Label(win, text="Detection Log", font=("Helvetica",13,"bold"),
                 bg="#16213e", fg="#e94560").pack(pady=8)
        cols = ("Timestamp","Type","Label","Action","Confidence")
        tree = ttk.Treeview(win, columns=cols, show="headings", height=18)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=145)
        sb = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True, padx=10)
        if os.path.exists(self.log_path):
            with open(self.log_path) as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    tree.insert("", "end", values=row)


if __name__ == "__main__":
    root = tk.Tk()
    app = DemoApp(root)
    root.mainloop()
