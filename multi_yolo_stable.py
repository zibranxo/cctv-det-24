from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8n-pose model
model = YOLO('yolov8n-pose.pt')

# Open webcam
cap = cv2.VideoCapture("test.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run pose detection
    results = model(frame, stream=True)

    # Process results
    for r in results:
        boxes = r.boxes
        poses = r.keypoints.data
        
        # Draw poses for each detected person
        for pose in poses:
            # Draw skeleton
            for i in range(len(pose)):
                if pose[i][2] > 0.5:  # Confidence threshold
                    x, y = int(pose[i][0]), int(pose[i][1])
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            
            # Connect keypoints to form skeleton
            skeleton = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 13),
                       (13, 15), (12, 14), (14, 16), (11, 12), (5, 11), (6, 12)]
            
            for pair in skeleton:
                if pose[pair[0]][2] > 0.5 and pose[pair[1]][2] > 0.5:
                    pt1 = (int(pose[pair[0]][0]), int(pose[pair[0]][1]))
                    pt2 = (int(pose[pair[1]][0]), int(pose[pair[1]][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    cv2.imshow('Multi-Person Pose Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()