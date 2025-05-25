import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Constants for UI buttons and status
BUTTON_HEIGHT = 40
BUTTON_WIDTH = 160
BUTTON_PADDING = 20

# Colors
COLOR_BG = (30, 30, 30)
COLOR_BTN_BG = (60, 60, 60)
COLOR_BTN_BG_ACTIVE = (0, 122, 204)
COLOR_BTN_TEXT = (220, 220, 220)
COLOR_TEXT = (200, 200, 200)
COLOR_CIRCLE = (0, 255, 0)
COLOR_CLICK_FEEDBACK = (0, 120, 255)
COLOR_SCROLL_FEEDBACK = (255, 69, 0)

# Screen size
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Gesture control states
gesture_enabled = True
click_mode_left = True  # True: left click, False: right click
sensitivity = 1.0  # Movement sensitivity multiplier

# Button definitions (label and function to toggle/update)
buttons = [
    {"label": "Disable Gestures", "toggle": "gesture"},
    {"label": "Left Click Mode", "toggle": "click_mode"},
    {"label": "Sensitivity: 1.0", "toggle": "sensitivity_inc"},
]

def draw_button(img, x, y, w, h, label, active):
    color_bg = COLOR_BTN_BG_ACTIVE if active else COLOR_BTN_BG
    cv2.rectangle(img, (x, y), (x + w, y + h), color_bg, cv2.FILLED)
    cv2.putText(img, label, (x + 10, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BTN_TEXT, 2)

def check_button_click(x, y, bx, by, bw, bh):
    return bx <= x <= bx + bw and by <= y <= by + bh

def detect_pointing(landmarks):
    # Index finger extended and others folded
    tip_id = 8
    pip_id = 6
    mcp_id = 5
    # Tip above pip and mcp joints for extended finger in y axis (camera coords, y downwards)
    if landmarks[tip_id].y < landmarks[pip_id].y and landmarks[pip_id].y < landmarks[mcp_id].y:
        # Check others folded: middle(12), ring(16), pinky(20) tips below pip joints
        folded = True
        for tip in [12, 16, 20]:
            pip = tip - 2
            if landmarks[tip].y < landmarks[pip].y:
                folded = False
                break
        return folded
    return False

def detect_pinch(landmarks, threshold=0.05):
    # Distance between tip of thumb (4) and index finger (8) below threshold
    x1, y1 = landmarks[4].x, landmarks[4].y
    x2, y2 = landmarks[8].x, landmarks[8].y
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return dist < threshold, dist

def detect_scroll_gesture(landmarks):
    # Two fingers extended vertically (index 8 and middle 12), thumb folded
    idx_extended = landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y
    thumb_folded = landmarks[4].x > landmarks[3].x  # thumb tip right of IP joint (folded)
    return idx_extended and thumb_folded

def main():
    global gesture_enabled, click_mode_left, sensitivity, buttons

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # For smoothing cursor movement
    prev_x, prev_y = 0, 0
    smoothening = 7  # adjustable for smoother cursor movement

    # Click debounce
    click_pressed = False
    last_click_time = 0
    click_delay = 0.3  # 300ms min delay between clicks

    # Scroll variables
    scroll_mode = False
    scroll_start_y = 0
    scroll_accum = 0

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror image for natural interaction
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            h, w, _ = frame.shape
            cursor_x, cursor_y = None, None
            action_text = ""

            # Draw UI buttons at the top
            for i, btn in enumerate(buttons):
                x = BUTTON_PADDING + i * (BUTTON_WIDTH + BUTTON_PADDING)
                y = BUTTON_PADDING
                active = False
                if btn["toggle"] == "gesture":
                    active = gesture_enabled
                elif btn["toggle"] == "click_mode":
                    active = click_mode_left
                elif btn["toggle"] == "sensitivity_inc":
                    active = True  # Always active, just use to show sensitivity value

                draw_button(frame, x, y, BUTTON_WIDTH, BUTTON_HEIGHT, btn["label"], active)

            if results.multi_hand_landmarks and gesture_enabled:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = hand_landmarks.landmark

                # Draw landmarks for visibility
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                                          mp_drawing.DrawingSpec(color=(0,180,0), thickness=2))

                # Detect pointing
                pointing = detect_pointing(landmarks)

                # Detect pinch (click or drag)
                pinching, pinch_dist = detect_pinch(landmarks)

                # Calculate cursor position based on index finger tip
                index_finger_tip = landmarks[8]
                screen_x = np.interp(index_finger_tip.x, [0, 1], [0, SCREEN_WIDTH])
                screen_y = np.interp(index_finger_tip.y, [0, 1], [0, SCREEN_HEIGHT])

                # Smoothen the cursor movement
                curr_x = prev_x + (screen_x - prev_x) / smoothening * sensitivity
                curr_y = prev_y + (screen_y - prev_y) / smoothening * sensitivity

                prev_x, prev_y = curr_x, curr_y

                cursor_x, cursor_y = int(curr_x), int(curr_y)

                pyautogui.moveTo(cursor_x, cursor_y)

                # Gesture actions
                # Click: Pinch with thumb and index finger close enough
                now = time.time()
                if pinching and not click_pressed and now - last_click_time > click_delay:
                    click_pressed = True
                    last_click_time = now
                    if click_mode_left:
                        pyautogui.click(button='left')
                        action_text = "Left Click"
                    else:
                        pyautogui.click(button='right')
                        action_text = "Right Click"
                if not pinching:
                    click_pressed = False

                # Scroll: Two fingers up/down gesture
                # (Implemented basic scroll mode toggle on two finger vertical gesture)
                scroll_gesture = detect_scroll_gesture(landmarks)
                if scroll_gesture and not scroll_mode:
                    scroll_mode = True
                    scroll_start_y = index_finger_tip.y
                    scroll_accum = 0
                elif not scroll_gesture and scroll_mode:
                    scroll_mode = False

                if scroll_mode:
                    scroll_delta = (scroll_start_y - index_finger_tip.y) * 25  # scale scroll amount
                    if abs(scroll_delta) > 0.05:
                        pyautogui.scroll(int(scroll_delta))
                        action_text = "Scrolling"
                        scroll_start_y = index_finger_tip.y

                # Visual feedback for click
                if pinching:
                    cv2.circle(frame, (cursor_x, cursor_y), 15, COLOR_CLICK_FEEDBACK, cv2.FILLED)
                else:
                    cv2.circle(frame, (cursor_x, cursor_y), 10, COLOR_CIRCLE, cv2.FILLED)

            else:
                # If no hand detected or gestures disabled, reset cursor prev positions to avoid jumps
                prev_x, prev_y = pyautogui.position()

            # Show cursor on frame
            if cursor_x and cursor_y:
                cv2.circle(frame, (cursor_x * w // SCREEN_WIDTH, cursor_y * h // SCREEN_HEIGHT), 10, (0, 255, 255), 2)

            # Show action text
            cv2.putText(frame, f"Action: {action_text}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

            # Show status text
            status_text = "Enabled" if gesture_enabled else "Disabled"
            cv2.putText(frame, f"Gesture Control: {status_text}", (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

            # Show sensitivity
            cv2.putText(frame, f"Sensitivity: {sensitivity:.1f}", (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

            # Detect mouse click on buttons (based on mouse position and left click)
            mouse_x, mouse_y = pyautogui.position()
            # Check if user left clicks in the video frame inside a button rectangle
            # We map mouse screen coordinates to frame coordinates for button detection
            frame_mouse_x = int(mouse_x * w / SCREEN_WIDTH)
            frame_mouse_y = int(mouse_y * h / SCREEN_HEIGHT)

            # We consider left mouse button clicks on the window for UI
            # We'll listen to keyboard mouse left click 'b' as toggle for simpler demo
            # Alternatively, user can click buttons by key presses or mouse clicks outside the window (simplify for this example)

            key = cv2.waitKey(1) & 0xFF

            # Use key presses to control UI buttons due to lack of GUI in OpenCV:
            # 'g' for toggle gesture enable/disable
            # 'c' for toggle click mode left/right
            # '+' for sensitivity increase
            # '-' for sensitivity decrease
            # 'q' to quit

            if key == ord('q'):
                break
            elif key == ord('g'):
                gesture_enabled = not gesture_enabled
                buttons[0]["label"] = "Disable Gestures" if gesture_enabled else "Enable Gestures"
            elif key == ord('c'):
                click_mode_left = not click_mode_left
                buttons[1]["label"] = "Left Click Mode" if click_mode_left else "Right Click Mode"
            elif key == ord('+') or key == ord('='):
                sensitivity = min(3.0, sensitivity + 0.1)
                buttons[2]["label"] = f"Sensitivity: {sensitivity:.1f}"
            elif key == ord('-') or key == ord('_'):
                sensitivity = max(0.2, sensitivity - 0.1)
                buttons[2]["label"] = f"Sensitivity: {sensitivity:.1f}"

            # Draw instruction text
            instructions = [
                "Controls:",
                "'g' - Toggle Gesture Enable/Disable",
                "'c' - Toggle Left/Right Click Mode",
                "'+'/'-' - Adjust Sensitivity",
                "'q' - Quit"
            ]
            for i, text in enumerate(instructions):
                cv2.putText(frame, text, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

            cv2.imshow("Hand Gesture Mouse Control", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()