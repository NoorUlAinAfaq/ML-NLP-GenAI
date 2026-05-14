import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,      # False = video mode (faster)
    max_num_hands=1,              # detect only 1 hand
    min_detection_confidence=0.7, # how confident before detecting
    min_tracking_confidence=0.5   # how confident while tracking
)

# Open webcam
cap = cv2.VideoCapture(0)
print("✅ Hand tracking started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame (mirror effect, more natural)
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB (MediaPipe needs RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = hands.process(rgb_frame)

    # If hand detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw the hand skeleton
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            # Print landmark coordinates of index fingertip (point 8)
            h, w, _ = frame.shape
            index_tip = hand_landmarks.landmark[8]
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            cv2.putText(frame, f"Index tip: ({x}, {y})",
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "No hand detected",
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 0, 255), 2)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Done!")