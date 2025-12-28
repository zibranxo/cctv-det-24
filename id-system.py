from ultralytics import YOLO
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8n-pose model
model = YOLO('yolov8n-pose.pt')

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Open video file
cap = cv2.VideoCapture("cross.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run pose detection
    results = model(frame, stream=True)

    detections = []  # Store detections for DeepSORT

    for r in results:
        poses = r.keypoints.data  # Get keypoints

        for pose in poses:
            # Get all keypoints with confidence > 0.5
            valid_points = [(int(kp[0]), int(kp[1])) for kp in pose if kp[2] > 0.5]

            if valid_points:
                # Compute a tighter bounding box around the keypoints
                x_min = min(p[0] for p in valid_points)
                y_min = min(p[1] for p in valid_points)
                x_max = max(p[0] for p in valid_points)
                y_max = max(p[1] for p in valid_points)

                # Store detection for DeepSORT tracking
                bbox = [x_min, y_min, x_max, y_max]
                detections.append((bbox, 1, 0.9))  # (bbox, class, confidence)

                # Draw tighter bounding box (Green)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Draw keypoints
                for x, y in valid_points:
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                # Draw skeleton
                skeleton = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 13),
                            (13, 15), (12, 14), (14, 16), (11, 12), (5, 11), (6, 12)]
                for pair in skeleton:
                    if pose[pair[0]][2] > 0.5 and pose[pair[1]][2] > 0.5:
                        pt1 = (int(pose[pair[0]][0]), int(pose[pair[0]][1]))
                        pt2 = (int(pose[pair[1]][0]), int(pose[pair[1]][1]))
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # Update DeepSORT tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_tlbr())  # Get bounding box

            # Display ID on the frame (No Red Box)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 0, 255), 2)

    cv2.imshow('Pose Tracking with Re-ID', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
