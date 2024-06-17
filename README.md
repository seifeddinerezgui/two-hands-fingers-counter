# Hand Gesture Recognition using MediaPipe

This project utilizes the MediaPipe library to recognize and count the number of fingers raised in front of a webcam in real-time. It leverages MediaPipe's Hand Landmarks model to detect hand positions and draw landmarks on the image frames captured from the webcam.

![Capture d’écran 2024-06-17 190334](https://github.com/seifeddinerezgui/two-hands-fingers-counter/assets/92888041/30643444-a885-4a3c-92ba-a686a921ef0e)


## Features

- Real-time hand detection using the webcam.
- Drawing of hand landmarks on the detected hands.
- Counting and displaying the number of fingers raised.
- Differentiates between left and right hands.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/hand-gesture-recognition.git
    cd hand-gesture-recognition
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install opencv-python mediapipe
    ```

## Usage

1. Run the script:
    ```sh
    python hand_gesture_recognition.py
    ```

2. A window will open displaying the webcam feed with hand landmarks and finger count annotated on the image. Press `q` to quit the application.

## Code Overview

The script captures video from the webcam, processes each frame to detect hand landmarks, and counts the number of fingers raised. The key steps are:

1. **Initialize MediaPipe and OpenCV:**
    ```python
    import cv2
    import mediapipe as mp

    mp_draw = mp.solutions.drawing_utils
    mp_hand = mp.solutions.hands
    tipIds = [4, 8, 12, 16, 20]
    ```

2. **Open the webcam:**
    ```python
    video = cv2.VideoCapture(0)
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    ```

3. **Process video frames:**
    ```python
    with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ret, image = video.read()
            if not ret:
                break

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    lmList = []
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])

                    mp_draw.draw_landmarks(image, hand_landmarks, mp_hand.HAND_CONNECTIONS)
                    hand_label = handedness.classification[0].label

                    fingers = []
                    if hand_label == 'Right':
                        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)
                    else:
                        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    for id in range(1, 5):
                        if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    total_fingers = fingers.count(1)
                    cv2.putText(image, f'Fingers: {total_fingers}', (lmList[0][1] - 10, lmList[0][2] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow("Frame", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    ```

4. **Release resources:**
    ```python
    video.release()
    cv2.destroyAllWindows()
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

