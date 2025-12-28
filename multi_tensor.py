import tensorflow as tf
import cv2
import numpy as np

# Load the TensorFlow Lite model (example: PoseNet or other TFLite model)
interpreter = tf.lite.Interpreter(model_path='posenet_model.tflite')
interpreter.allocate_tensors()

# Input & Output Details for the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to process frame
def process_frame(frame):
    # Resize frame to match model input dimensions (e.g., 224x224 or 256x256)
    frame_resized = cv2.resize(frame, (256, 256))
    input_data = np.expand_dims(frame_resized, axis=0).astype(np.float32)

    # Set the model input
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run the inference
    interpreter.invoke()

    # Extract the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data  # Return the model's detected keypoints or results

# Capture video input
cap = cv2.VideoCapture(0)
frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames to reduce lag (process every 5th frame)
    frame_counter += 1
    if frame_counter % 5 != 0:
        continue

    # Reduce resolution for faster processing
    frame_resized = cv2.resize(frame, (640, 480))

    # Run pose detection on the frame
    output = process_frame(frame_resized)

    # Post-process and draw keypoints (implement your logic here)
    # Draw keypoints on the frame (depending on output format)
    # For example:
    for keypoint in output:  # Modify this based on your model's output
        if keypoint['confidence'] > 0.3:  # Confidence threshold
            x, y = int(keypoint['x']), int(keypoint['y'])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Show the frame
    cv2.imshow('PoseNet Pose Detection', frame)

    # Exit on 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()