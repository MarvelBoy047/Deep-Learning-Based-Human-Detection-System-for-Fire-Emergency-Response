import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to classify posture
def classify_posture(landmarks, image_height, image_width):
    try:
        # Get landmark positions
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

        # Convert normalized coordinates to pixel values
        left_shoulder_y = int(left_shoulder.y * image_height)
        right_shoulder_y = int(right_shoulder.y * image_height)
        left_hip_y = int(left_hip.y * image_height)
        right_hip_y = int(right_hip.y * image_height)
        left_knee_y = int(left_knee.y * image_height)
        right_knee_y = int(right_knee.y * image_height)
        left_ankle_y = int(left_ankle.y * image_height)
        right_ankle_y = int(right_ankle.y * image_height)

        # Posture classification logic:
        # Check if the person is lying down by comparing the y-coordinates of the hips, knees, and ankles.
        # If hips and shoulders are on a similar horizontal level (small y-difference), and knees/ankles are not significantly below hips, we consider it lying down.
        
        # Check standing (shoulders should be significantly above hips and hips above knees/ankles)
        if (left_shoulder_y < left_hip_y - 50 and right_shoulder_y < right_hip_y - 50 and
            left_hip_y < left_knee_y and right_hip_y < right_knee_y and
            left_knee_y < left_ankle_y and right_knee_y < right_ankle_y):
            return "Standing", True  # Safe
        else:
            return "Lying", False  # Not safe
    except Exception as e:
        print(f"Error in landmark detection: {e}")
        return "Unknown", False

# Load image
image_path = r"C:\Users\asush\Downloads\fire\fire\val\images\44.jpg"  # Replace with your image path
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape

# Convert the image to RGB as mediapipe expects it
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image to detect poses
results = pose.process(image_rgb)

# Check if any pose landmarks are detected
if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark
    
    # Draw landmarks on the image
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Classify the posture and determine safety status
    posture, is_safe = classify_posture(landmarks, image_height, image_width)

    # Annotate image with the posture and safety status
    status = "Safe" if is_safe else "Not Safe"
    text = f"Posture: {posture}, Status: {status}"

    # Put text on the image
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_safe else (0, 0, 255), 2)

else:
    print("No pose landmarks detected.")

# Show the image
cv2.imshow("Posture Detection", image)

# Press any key to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()