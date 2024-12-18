# Detecting Human Life During Fire Using AI-Driven Technologies

This repository contains the implementation of an AI-based system designed to detect and prioritize individuals during fire emergencies. The system leverages state-of-the-art technologies, including the YOLOv8 object detection model and MediaPipe for posture analysis, ensuring efficient and ethical rescue operations. 

## Features
- **Real-Time Object Detection**: Detect humans in fire-affected areas with high accuracy using YOLOv8.
- **Posture Analysis**: Identify the physical condition of individuals (e.g., standing, sitting, lying) with MediaPipe.
- **Privacy Preservation**: Automatically blur faces to maintain anonymity and comply with ethical standards.
- **User-Friendly Interface**: A Streamlit-based dashboard for real-time video upload, processing, and result visualization.
- **Scalability**: Supports deployment on various hardware configurations, including GPU-enabled devices.

---

## Project Workflow
### 1. Video Processing
- Upload a video via the Streamlit interface.
- Real-time processing occurs frame-by-frame.

### 2. Detection and Pose Analysis
- **YOLOv8**: Detects humans with bounding boxes and confidence scores.
- **MediaPipe**: Analyzes poses to prioritize rescue actions.

### 3. Result Visualization
- Processed frames are displayed in real-time.
- Metrics like detection count, posture, and confidence scores are updated live.
- Annotated video can be downloaded post-processing.

---
