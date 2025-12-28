# 🧍‍♂️ Multi-Person Pose Detection & Re-Identification System

This project explores **robust multi-human pose detection and identity tracking in video footage**, with a long-term goal of **multi-camera human position estimation**.

The repository documents an **iterative engineering journey** — starting from single-person pose estimation, moving through multi-person detection challenges, and finally stabilizing on a YOLOv8-based pipeline with identity tracking.

---

## 🎯 Problem Statement

Real-world video analysis systems must be able to:

- Detect **multiple people** in a single frame  
- Estimate **full-body pose** for each person  
- **Track identities** across frames  
- Re-identify individuals after **temporary occlusion or missed detections**  
- Operate under **real-time constraints**  

This project experiments with multiple approaches to understand the trade-offs involved in solving these challenges.

---

## 🧠 Project Evolution & Approach

Each script in this repository represents a **distinct stage of experimentation** and learning.

---

## 1️⃣ Single-Person Pose Detection (Baseline)

### `single_mediapipe.py`

**Approach**
- Uses **MediaPipe Pose**
- Webcam or video input
- Draws a pose skeleton (wireframe)

**Outcome**
- Accurate and lightweight
- Limited to **single-person detection**

**Conclusion**
> Useful as a baseline, but unsuitable for crowded scenes.

---

## 2️⃣ Multi-Person Pose via TensorFlow / PoseNet

### `multi_tensor.py`

**Approach**
- Uses **PoseNet (TensorFlow Lite)**
- Frame skipping and resolution reduction to reduce latency
- Attempts multi-person pose estimation

**Trade-offs**
- Improved speed with TFLite
- Noticeable accuracy drop
- Still laggy for real-time use

**Conclusion**
> Demonstrates feasibility, but not stable enough for deployment.

---

## 3️⃣ Stable Multi-Person Pose Detection (YOLOv8)

### `multi_yolo_stable.py`

**Approach**
- Uses **YOLOv8n-pose**
- Detects multiple people and their keypoints in a single forward pass
- Skeletons rendered using OpenCV

**Why this worked**
- Unified detection and pose estimation
- Robust handling of crowded scenes
- Real-time capable

**Conclusion**
> This became the **stable backbone** of the system.

---

## 4️⃣ Identity Tracking & Re-Identification (WIP)

### `id-system.py`

**Approach**
- Uses YOLOv8 pose keypoints
- Constructs **tight bounding boxes from keypoints**
- Integrates **DeepSORT** for tracking
- Assigns persistent IDs to detected people

**Key Design Choice**
- Bounding boxes derived from **pose geometry**, not raw detector boxes  
  → improves ID stability during motion and pose changes

**Status**
- Prototype / Work in progress
- Architecturally sound direction for re-identification

---

## 🔗 System Capabilities (Current)

- ✅ Multi-person detection in video  
- ✅ Full-body pose estimation per person  
- ✅ Stable skeleton rendering  
- ✅ Identity tracking across frames  
- ⚠️ Partial re-identification after occlusion  
- 🚧 Multi-camera stitching (future work)  

---

## 🧩 Why This Project Is Non-Trivial

- Explores **multiple pose estimation pipelines**
- Balances **latency vs accuracy**
- Uses **pose-based tracking**, not only bounding boxes
- Integrates detection, pose estimation, and tracking
- Designed with **real-time constraints** in mind

This is an **engineering exploration project**, not a one-script demo.

---

## 🛠️ Tech Stack

- **Detection & Pose:** YOLOv8 (Ultralytics)
- **Tracking:** DeepSORT
- **Vision:** OpenCV
- **Pose Frameworks:** MediaPipe, PoseNet (TFLite)
- **Language:** Python

---

## 📁 Repository Structure

```
├── single_mediapipe.py      # Single-person pose baseline
├── multi_tensor.py          # PoseNet / TFLite multi-person experiment
├── multi_yolo_stable.py     # Stable YOLOv8 multi-person pose detection
├── id-system.py             # Pose-based identity tracking (WIP)
├── yolov8n-pose.pt          # YOLOv8 pose model
├── README.md
└── .gitattributes
```

---

## 🚀 How to Run

### Install dependencies
```bash
pip install ultralytics opencv-python deep-sort-realtime mediapipe tensorflow
```

### Run YOLOv8 multi-person pose detection
```bash
python multi_yolo_stable.py
```

### Run pose-based ID tracking
```bash
python id-system.py
```

---

## 🔮 Future Work

- Improve re-identification robustness
- Handle long-term occlusions
- Integrate **multi-camera stitching**
- Estimate global human positions across cameras
- Optimize for edge / low-power devices

---

## 📄 License

This project is intended for **educational and experimental use**.
