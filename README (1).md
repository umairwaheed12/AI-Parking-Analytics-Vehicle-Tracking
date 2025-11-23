# 🚗 AI-Parking-Analytics-Vehicle-Tracking
A complete AI-powered parking-lot analytics system using YOLO, ROI-based aerial detection, anti-flicker tracking, heatmaps, and a 3D minimap — all in a single Python script (`park.py`).

This system processes parking-lot videos and generates an output video with:
- ✔ Car & free-slot detection
- ✔ Aerial vehicle detection inside selected ROIs
- ✔ Stable anti-flicker object tracking
- ✔ Mouse-based interactive ROI selection
- ✔ Heatmap of motion activity
- ✔ 3D minimap with trails and perspective projection
- ✔ Right-side stats panel (counts, timers, minimap)
- ✔ Merged output video preview window

## 📁 Repository Structure
```
project_root/
│
├── park.py                     # Main script (detection + tracking + UI panel)
├── README.md                   # Project documentation
├── models/                     
│   ├── parking_model.pt
│   └── aerial_model.pt
└── examples/
    └── input_video.mp4
```

## 🔧 Requirements
Python 3.9 – 3.12

Install dependencies:
```
pip install ultralytics opencv-python numpy
```

Optional:
```
pip install opencv-contrib-python
```

## ▶️ How to Run
1. Place your YOLO weight files in `models/`.
2. Update paths inside `park.py`.
3. Run:
```
python park.py
```

## 🖱️ ROI Selection
- LMB drag: draw ROI
- Z: undo
- R: reset
- Enter: confirm
- Q/ESC: cancel

## 🎯 Features
- Parking detection
- Aerial ROI-restricted detection
- Anti-flicker tracking
- Heatmap
- 3D minimap
- Dashboard panel

## 📦 Output
- Combined video (original + panel)
- Real-time preview
- Final output file

## 📝 License
MIT License
