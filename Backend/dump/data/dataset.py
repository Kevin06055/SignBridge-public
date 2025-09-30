# Compare training vs validation metrics
from ultralytics import YOLO

model = YOLO('D:/MyWorks/SignBridge/best.pt')
# Check if training mAP >> validation mAP
results = model.val()
print(f"Validation mAP: {results.box.map}")
