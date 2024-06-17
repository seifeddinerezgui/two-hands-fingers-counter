import cv2
import mediapipe as mp

# Initialize MediaPipe Hand and drawing utilities
mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands
tipIds = [4, 8, 12, 16, 20]

# Open the video capture device (webcam)
video = cv2.VideoCapture(0)

# Create a named window with the option to resize
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

# Initialize the MediaPipe Hands solution
with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, image = video.read()
        if not ret:
            break

        # Flip the image horizontally for a mirror effect
        image = cv2.flip(image, 1)

        # Convert the image to RGB and process it with MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Check if any hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                lmList = []

                # Get hand landmarks
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                # Draw hand landmarks
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hand.HAND_CONNECTIONS)

                # Determine if the hand is left or right
                hand_label = handedness.classification[0].label

                # Count fingers
                fingers = []
                if hand_label == 'Right':  # Adjust thumb logic for the right hand
                    if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:  # Thumb
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:  # Left hand
                    if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:  # Thumb
                        fingers.append(1)
                    else:
                        fingers.append(0)

                for id in range(1, 5):  # Other four fingers
                    if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                total_fingers = fingers.count(1)

                # Display the finger count on the image
                cv2.putText(image, f'Fingers: {total_fingers}', (lmList[0][1] - 10, lmList[0][2] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the result
        cv2.imshow("Frame", image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
video.release()
cv2.destroyAllWindows()
