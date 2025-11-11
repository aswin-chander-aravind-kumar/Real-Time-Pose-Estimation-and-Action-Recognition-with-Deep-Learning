# 🕵️ Motion Detection System

A **real-time motion detection system** that integrates **OpenPose** and **MoveNet** for advanced human pose estimation and action recognition.  
Developed as part of **CS5330 – Pattern Recognition and Computer Vision** at **Northeastern University**.

---

## 📘 Overview

This project combines the strengths of **OpenPose** and **MoveNet** to detect, track, and analyze human motion.  
- **OpenPose** provides detailed, multi-person keypoint detection (body, face, hands, feet).  
- **MoveNet** offers fast, lightweight real-time inference suitable for mobile and embedded systems.  
- An **LSTM model** is used to classify human actions based on temporal sequences of pose keypoints.

---

## ⚙️ Methodology

### 1. Input
- Input: pre-recorded test video.  
- Each frame is extracted and preprocessed for pose estimation.

### 2. Pose Estimation Models
**OpenPose**
- Detects body, face, hands, and feet keypoints.  
- Draws skeletal lines for visualization.  
- Average confidence score: **0.42**

**MoveNet (Lightning / Thunder)**
- Single-pose and multi-pose modes.  
- Optimized for real-time performance.  
- Average confidence score: **0.25**

### 3. Tracking & Visualization
- Detected keypoints are connected to form human skeletons.  
- Frame-by-frame confidence plots visualize detection stability.  

### 4. Action Classification (LSTM)
- Sequences of pose keypoints are passed to an LSTM network.  
- LSTM captures temporal motion dynamics for **action recognition**.

---

## 🧩 Extensions
- Combine multiple models for ensemble predictions.  
- Extend to **multi-person action classification**.  
- Optimize for edge and mobile devices.  
- Integrate **MediaPipe** for preprocessing and pose refinement.  

---

## 👥 Team

| Name | Role |
|------|------|
| **Aswin Chander Aravind Kumar** | Pose Estimation & Camera Calibration |
| **Shruti Pasumarti** | 3D Overlay & AR Integration |

---

## 💭 Reflections

**Aswin Chander Aravind Kumar**  
> “This project deepened my understanding of computer vision, OpenCV, and camera calibration. It showed me the importance of careful parameter tuning for accurate detection.”

**Shruti Pasumarti**  
> “Depth perception and AR overlay alignment were challenging but rewarding. Debugging 3D overlays improved my understanding of integrating virtual and real-world coordinates.”

---

## 📚 Resources
- Professor’s Notes  
- Reference Codes  

---

## 🧰 Technologies Used
- Python  
- OpenCV  
- TensorFlow / PyTorch  
- OpenPose  
- MoveNet  
- LSTM Networks  

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install opencv-python tensorflow numpy matplotlib
