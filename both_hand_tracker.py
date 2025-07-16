import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

finger_tips = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Flip the image horizontally
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    h, w, _ = img.shape
    total_fingers = 0

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            lm_list = []
            true_label = handedness.classification[0].label  # 'Left' or 'Right'

            # Flip label to match what user sees
            if true_label == 'Right':
                label = 'Left'
            else:
                label = 'Right'

            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            finger_count = 0

            # Count raised fingers (index to pinky)
            for i in range(1, 5):
                tip_id = finger_tips[i]
                pip_id = tip_id - 2
                mcp_id = tip_id - 3

                if (
                    lm_list[tip_id][1] < lm_list[pip_id][1] and
                    lm_list[pip_id][1] < lm_list[mcp_id][1]
                ):
                    finger_count += 1

            # Thumb logic (based on flipped label)
            if label == 'Right':
                if lm_list[4][0] > lm_list[3][0]:
                    finger_count += 1
            else:
                if lm_list[4][0] < lm_list[3][0]:
                    finger_count += 1

            total_fingers += finger_count

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(img, f'{label} Hand: {finger_count}', (lm_list[0][0], lm_list[0][1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.putText(img, f'Total Fingers: {total_fingers}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow("Finger Counter (Both Hands)", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
