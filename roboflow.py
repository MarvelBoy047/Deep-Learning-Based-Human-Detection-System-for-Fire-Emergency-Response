# import necessary libraries
from inference_sdk import InferenceHTTPClient
import cv2
import matplotlib.pyplot as plt
import json

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="uLvnX47g8aGw9FwAlq9t"
)

# local image path
image_path = r"C:\Users\asush\Downloads\07fire19.jpg"

# infer on a local image
result = CLIENT.infer(image_path, model_id="fire-zorks/1")

# Parse the result to extract bounding boxes and predictions
predictions = result['predictions']

# Load the image using OpenCV
image = cv2.imread(image_path)

# Loop through each prediction and draw bounding boxes and labels
for prediction in predictions:
    # Get bounding box coordinates
    x = int(prediction['x'])
    y = int(prediction['y'])
    width = int(prediction['width'])
    height = int(prediction['height'])

    # Define the top-left and bottom-right points for the bounding box
    top_left = (x - width // 2, y - height // 2)
    bottom_right = (x + width // 2, y + height // 2)

    # Draw the bounding box on the image (color: red, thickness: 2)
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)

    # Add the class label and confidence score
    label = prediction['class']
    confidence = prediction['confidence']
    label_text = f"{label}: {confidence:.2f}"

    # Put the label text above the bounding box
    cv2.putText(image, label_text, (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Convert the image to RGB (from BGR) for displaying with matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image with bounding boxes and labels using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis('off')  # Hide the axis
plt.show()
