# 🚗 AI-Parking-Analytics-Vehicle-Tracking

A full AI-powered parking analytics system using YOLO detection, aerial ROI detection, anti‑flicker tracking, heatmaps, and a 3D minimap — all integrated inside one script: **park.py**.

---

# 📁 Project Structure

project includes **two model folders**, each having its own purpose:

---

## 🟩 **Model 1 — Parking Detection Model**
Folder:
```
model1/
```
Purpose:
- Detect **cars**  
- Detect **free parking slots**  
- Provide bounding boxes for parking occupancy logic  
- Used by the main YOLO detection pipeline in `park.py`  

Expected file inside:
```
model1/weights/last.pt
```

---

## 🟦 **Model 2 — Aerial Vehicle Detection Model**
Folder:
```
multi-vehicles/Exp_Sample/
```
Purpose:
- Detect vehicles from **aerial / top‑view perspective**
- Works **only inside user‑drawn ROIs**
- Provides `"Vehiculo"` class detections for guiding dotted-line paths
- Supports classes `{0, 1, 2, 3}` depending on training

Expected file inside:
```
multi-vehicles/Exp_Sample/weights/last.pt
```

---

# 🔧 Requirements

Python version: **3.9 – 3.12**

Install dependencies:
```bash
pip install ultralytics opencv-python numpy
```

Optional (for GUI support):
```bash
pip install opencv-contrib-python
```

---

# ▶️ How to Run

1. Place your model files exactly like this:
```
model1/weights/last.pt  
multi-vehicles/Exp_Sample/weights/last.pt
```

2. Update paths inside the top of `park.py`:
```python
MODEL_PATH = r"model1/weights/last.pt"
AERIAL_MODEL_PATH = r"multi-vehicles/Exp_Sample/weights/last.pt"
```

3. Run the script:
```bash
python park.py
```

---

# 🖱️ ROI Selection Controls

When the first frame appears:
- **Left Mouse Drag** → Draw ROI
- **Z** → Undo last ROI
- **R** → Reset all ROIs
- **ENTER** → Confirm
- **Q / ESC** → Cancel

---

# 🎯 System Features

### 🚙 Parking Model (Model 1)
- Detects `car` and `free` classes
- Stable tracking with anti‑flicker smoothing
- Car timers (seconds parked)
- Free space counters

---

### 🛩️ Aerial Model (Model 2)
- Detections allowed only **inside ROIs**
- Used to guide vehicles towards nearest free parking slot
- Tracking performed separately from main tracker

---

### 🔥 Heatmap Generation
- Movement‑based heat accumulation
- Smooth decay over time
- Rendered inside minimap

---

### 🕹️ 3D Minimap
- Perspective projected parking‑lot visualization
- Pillars showing activity intensity
- Historical trails for each tracked car
- Integrated inside right‑side panel

---

# 🎨 Output

The output video contains:
- Left: Original processed video with all overlays  
- Right: Stats panel (counts, timers, minimap, heatmap)  

Saved automatically to the path in:
```python
VIDEO_OUT = "..."
```

---

# 📜 License
MIT License

---

# 📞 Support
If you want:
- Cleaner modular version  
- Folder auto‑creation  
- Multiple video input support  
- Web dashboard version  

Just ask!
