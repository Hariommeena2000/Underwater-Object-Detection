# Underwater Object Detection (UOD)

This project aims to detect various underwater objects like fish, coral reefs, rocks, and marine debris using state-of-the-art deep learning models. We compare and deploy two major object detection architectures: **YOLOv8** and **Faster R-CNN**, trained specifically for challenging underwater conditions such as low light, distortion, and turbidity.

---

## 🌊 Project Overview

- **Objective:** Robust real-time detection of underwater objects.
- **Challenges Addressed:** Low visibility, color distortion, noise, and underwater image degradation.
- **Models Used:**
  - **YOLOv8:** Fast, lightweight model for real-time detection.
  - **Faster R-CNN:** High-accuracy model, better for complex underwater scenes.
- **Frameworks:** Python, PyTorch, OpenCV
- **Dataset:** Curated underwater datasets, labeled for object detection.

---

## 🏛️ Project Structure

```bash
UOD/
│
├── datasets/         # Underwater datasets (images/videos + annotations)
├── runs/             # YOLOv8 training results
├── fasterrcnn_runs/  # Faster R-CNN training results
├── weights/          # Trained model weights (YOLOv8 best.pt and Faster R-CNN .pth)
├── configs/          # Model configuration files
├── train_yolo.py     # Training script for YOLOv8
├── train_fasterrcnn.py  # Training script for Faster R-CNN
├── detect_yolo.py    # Detection script for YOLOv8
├── detect_fasterrcnn.py # Detection script for Faster R-CNN
├── data.yaml         # Dataset configuration
├── README.md         # Project documentation
└── requirements.txt  # Python dependencies
