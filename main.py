import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import torch
import gdown
import os
import time

# Ensure GPU is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Download YOLO weights if not already present
MODEL_URL = "https://drive.google.com/uc?id=14KjAuw8JOox-YUHnQlocBwN_0m_3yi3p"
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    print("Downloading YOLO model weights...")
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

# Initialize Mediapipe Pose, Face Detection, and Drawing Utils
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

# Open the video file
cap = cv2.VideoCapture("6.mp4")  # Replace with your actual video path

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Setup Mediapipe Pose and Face Detection instances
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("End of video or failed to read the frame.")
            break

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
        conf = 0.6
        start_time = time.time()
        annotated_frame, metrics = process_frame(image, conf)
        end_time = time.time()
        inference_time = end_time - start_time

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

        # Blur detected faces
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = annotated_frame.shape

                # Convert normalized bounding box to pixel values
                x_min = int(bboxC.xmin * w)
                y_min = int(bboxC.ymin * h)
                box_width = int(bboxC.width * w)
                box_height = int(bboxC.height * h)

                # Ensure the bounding box is within the frame
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_min + box_width)
                y_max = min(h, y_min + box_height)

                # Extract the region of interest (face) and blur it
                face_roi = annotated_frame[y_min:y_max, x_min:x_max]
                face_roi = cv2.GaussianBlur(face_roi, (51, 51), 30)

                # Replace the original face region with the blurred version
                annotated_frame[y_min:y_max, x_min:x_max] = face_roi

        # Resize the frame for better viewing
        small_frame = cv2.resize(annotated_frame, (640, 360))  # Resize to 640x360 resolution

        # Display the frame
        cv2.imshow("Pose and Face Detection with Blurring", small_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
