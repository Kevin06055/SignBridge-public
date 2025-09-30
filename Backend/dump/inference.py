from ultralytics import YOLO
import cv2
import numpy as np

# ----- CONFIGURATION -----
MODEL_PATH = "fine_tuned.pt"  # path to your trained model
CONF_THRESHOLD = 0.5  # confidence threshold for displaying detections
ROI_SIZE = 300  # size of the square ROI for hand placement

# ----- LOAD YOLO MODEL -----
model = YOLO(MODEL_PATH)

# ----- SETUP WEBCAM -----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

# Read one frame to get dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Cannot read from webcam")
    exit()
height, width = frame.shape[:2]
center_x, center_y = width // 2, height // 2

# Define ROI coordinates
roi_x1 = center_x - ROI_SIZE // 2
roi_y1 = center_y - ROI_SIZE // 2
roi_x2 = center_x + ROI_SIZE // 2
roi_y2 = center_y + ROI_SIZE // 2

print("✅ Place your hand inside the green box. Press 'q' to quit.")

# ----- MAIN LOOP -----
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Draw the ROI box on the frame
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)

    # Extract the ROI area
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    # Apply Gaussian blur to smooth noise
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 1.5)

    # Convert ROI to grayscale
    gray = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding to get binary mask
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Run YOLO inference on the ROI
    results = model(roi)

    # Annotate ROI with bounding boxes and labels
    annotated_roi = roi.copy()
    for result in results:
        for box in result.boxes:
            if box.conf.item() < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls.item())
            conf = box.conf.item()
            label = f"{result.names[cls_id]} {conf:.2f}"
            cv2.rectangle(annotated_roi, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_roi, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Insert the annotated ROI back into the frame
    frame[roi_y1:roi_y2, roi_x1:roi_x2] = annotated_roi

    # Prepare binary mask for display by converting grayscale to BGR
    mask_display = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Resize mask_display to match frame's dimensions
    mask_display_resized = cv2.resize(mask_display, (width, height))

    # Combine the frame and mask display horizontally
    combined_display = cv2.hconcat([frame, mask_display_resized])

    # Show the combined output
    cv2.imshow("ISL Detection and Mask", combined_display)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----- CLEANUP -----
cap.release()
cv2.destroyAllWindows()
