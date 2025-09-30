import cv2
import mediapipe as mp

# -------------------------------
# Initialize MediaPipe Hands
# -------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,     # True for single image processing
    max_num_hands=2,
    min_detection_confidence=0.2
)

# -------------------------------
# Load your sample image
# -------------------------------
image_path = "test1.jpg"  # Replace with your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# -------------------------------
# Detect hands
# -------------------------------
results = hands.process(image_rgb)

# -------------------------------
# Draw keypoints if detected
# -------------------------------
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
        )

# -------------------------------
# Show the image
# -------------------------------
cv2.imshow("Hand Keypoints", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -------------------------------
# Print normalized landmark coordinates
# -------------------------------
if results.multi_hand_landmarks:
    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        print(f"\nHand {idx+1} keypoints:")
        for i, lm in enumerate(hand_landmarks.landmark):
            print(f"Landmark {i}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")
