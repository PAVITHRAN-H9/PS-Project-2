Crowd counting and Density estimation

import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture("peoples.mp4")

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500: # Minimum size to filter noise
            count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Estimate density as number of people per frame area (simple proxy)
    frame_area = frame.shape[0] * frame.shape[1]
    density = count / (frame_area / 10000) # Arbitrary scale

    # Show count and density on frame
    cv2.putText(frame, f"People Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Density: {density:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Show frames
    cv2.imshow("Crowd Detection", frame)
    cv2.imshow("Foreground Mask", fgmask)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()