import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import torch
import gdown
import os
import tempfile
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Ensure GPU is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Download YOLO weights if not already present
MODEL_URL = "https://drive.google.com/uc?id=14KjAuw8JOox-YUHnQlocBwN_0m_3yi3p"
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading YOLO model weights..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load YOLO model globally
yolo_model = YOLO(MODEL_PATH).to(DEVICE)

def process_frame(frame, conf):
    """Process a single frame for YOLO detection and gather metrics."""
    results = yolo_model.predict(frame, imgsz=640, conf=conf, verbose=False, device=DEVICE)
    result = results[0]
    annotated_frame = result.plot()

    # Extract metrics
    metrics = {
        "boxes_count": len(result.boxes),
        "class_labels": result.boxes.cls.tolist() if len(result.boxes) > 0 else [],
        "confidences": result.boxes.conf.tolist() if len(result.boxes) > 0 else []
    }
    return annotated_frame, metrics

# Initialize Mediapipe Pose and Face Detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

def calculate_angle(a, b, c):
    """Calculate the angle between three points (a, b, c)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

# Streamlit Interface
st.title("Detecting Human Life during Fire")
st.sidebar.title("Settings")
confidence = st.sidebar.slider("YOLO Confidence Threshold", 0.0, 1.0, 0.6, 0.01)
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if video_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_file.read())
        temp_video_path = temp_file.name

    cap = cv2.VideoCapture(temp_video_path)
    stframe = st.empty()

    # Prepare output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = "annotated_output.mp4"
    out = None

    inference_times = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Start inference timer
            start_time = time.time()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process pose detection
            pose_results = pose.process(image)

            # Process face detection
            face_results = face_detection.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Process YOLO detection
            annotated_frame, metrics = process_frame(image, confidence)

            # Analyze posture if pose landmarks are detected
            if pose_results.pose_landmarks:
                # Extract key landmarks
                landmarks = pose_results.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate angles
                upper_body_angle = calculate_angle(shoulder, hip, knee)
                lower_body_angle = calculate_angle(hip, knee, ankle)

                # Determine posture based on angles
                if upper_body_angle > 160 and lower_body_angle > 160:
                    posture = "Standing"
                elif upper_body_angle < 160 and lower_body_angle < 90:
                    posture = "Sitting"
                elif upper_body_angle < 70 and lower_body_angle < 70:
                    posture = "Lying Down"
                else:
                    posture = "Unknown"

                # Display posture on the frame
                cv2.putText(annotated_frame, f"Posture: {posture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                )

            # End inference timer
            end_time = time.time()
            inference_times.append(end_time - start_time)

            # Write to output video
            if out is None:
                height, width, _ = annotated_frame.shape
                out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
            out.write(annotated_frame)

            # Display frame in Streamlit
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)

    cap.release()
    out.release()

    # Display average inference time
    avg_inference_time = np.mean(inference_times)
    st.success(f"Processing complete! Average inference time per frame: {avg_inference_time:.2f} seconds")

    # Provide download link
    with open(output_path, "rb") as f:
        st.download_button("Download Annotated Video", f, file_name="annotated_output.mp4")
