import sys, os
from time import time
from collections import deque

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyeyesweb.data_models.sliding_window import SlidingWindow

import numpy as np
import cv2
import mediapipe as mp
from pyeyesweb.analysis_primitives.synchronization import Synchronization

# Function to extract the y-coordinate of a specified keypoint (e.g., wrist) from the Mediapipe Pose results.
# The keypoint's visibility is checked to ensure it's sufficiently visible before processing.
def extract_coordinates(results, keypoint_idx):
    """Extract coordinates of a given keypoint."""
    keypoint = results.pose_landmarks.landmark[keypoint_idx]
    if keypoint.visibility > 0.5:  # Only consider keypoints with visibility above the threshold
        return (keypoint.x, keypoint.y, keypoint.z)  # Return the (x, y, z) coordinates
    return None  # Return None if the keypoint is not sufficiently visible

def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(angle)

# Function to interpolate color between green, yellow, and red based on the angle
def interpolate_color(value, min_value=20, max_value=70):
    """Interpolate color between green (aligned), yellow, and red (bent) using HSV."""
    value = np.clip(value, min_value, max_value)  # Clamp the value within the range
    ratio = (value - min_value) / (max_value - min_value)  # Normalize the value
    hue = int(120 - (120 * ratio))  # Interpolate hue from 120 (green) to 0 (red) through yellow
    hsv_color = np.uint8([[[hue, 255, 255]]])  # Create an HSV color
    rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]  # Convert HSV to RGB
    return tuple(int(c) for c in rgb_color)  # Return the RGB color as a tuple

# Function to process and visualize wrist data
def process_wrist(frame, original_frame, width, height, elbow, wrist, pinky, index, angle_history, side="left"):
    """Process and visualize wrist data."""
    if all(coord is not None for coord in [elbow, wrist, pinky, index]):
        palm = ((pinky[0] + index[0]) / 2, (pinky[1] + index[1]) / 2, (pinky[2] + index[2]) / 2)
        forearm = np.array([wrist[i] - elbow[i] for i in range(3)])
        wrist_palm = np.array([palm[i] - wrist[i] for i in range(3)])
        angle = angle_between(forearm, wrist_palm)

        wrist_x = int(wrist[0] * width)
        wrist_y = int(wrist[1] * height)

        elbow_wrist_distance = np.linalg.norm(np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]]) * np.array([width, height]))
        radius = int(elbow_wrist_distance * 0.6)
        if radius > 7:
            center_x, center_y = wrist_x, wrist_y
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            wrist_area = cv2.bitwise_and(original_frame, original_frame, mask=mask)  # Use the unmodified frame

            x1, y1 = max(0, center_x - radius), max(0, center_y - radius)
            x2, y2 = min(width, center_x + radius), min(height, center_y + radius)
            cropped_area = wrist_area[y1:y2, x1:x2]

            output_size = (200, 200)
            zoomed_area = cv2.resize(cropped_area, output_size, interpolation=cv2.INTER_LINEAR)

            circular_mask = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
            cv2.circle(circular_mask, (output_size[0] // 2, output_size[1] // 2), output_size[0] // 2, 255, -1)
            zoomed_area_with_mask = cv2.bitwise_and(zoomed_area, zoomed_area, mask=circular_mask)

            angle_history.append(angle)
            smoothed_angle = np.mean(angle_history)
            border_color = interpolate_color(smoothed_angle)

            cv2.circle(frame, (center_x, center_y), radius, border_color, 3)

            zoomed_x = 10 if side == "left" else width - output_size[0] - 10
            zoomed_y = height // 2 - output_size[1] // 2
            for c in range(3):
                frame[zoomed_y:zoomed_y + output_size[1], zoomed_x:zoomed_x + output_size[0], c] = \
                    np.where(circular_mask == 255, zoomed_area_with_mask[:, :, c], frame[zoomed_y:zoomed_y + output_size[1], zoomed_x:zoomed_x + output_size[0], c])

            text_x = zoomed_x + output_size[0] + 10 if side == "left" else zoomed_x - 100
            text_y = zoomed_y + output_size[1] // 2
            cv2.putText(frame, f'{smoothed_angle:.2f} deg', (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_color, 2)

# Main function
def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture('C:\\Users\\paolo\\Videos\\Legino-Tobia\\Leva 2014\\MOVI0001.avi')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    LEFT_ELBOW_IDX, LEFT_WRIST_IDX, LEFT_PINKY_IDX, LEFT_INDEX_IDX = 13, 15, 17, 19
    RIGHT_ELBOW_IDX, RIGHT_WRIST_IDX, RIGHT_PINKY_IDX, RIGHT_INDEX_IDX = 14, 16, 18, 20

    left_angle_history = deque(maxlen=5)
    right_angle_history = deque(maxlen=5)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            original_frame = frame.copy()  # Make a copy of the original frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            height, width, _ = image.shape

            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            percentage = (current_frame / frame_count) * 100
            cv2.putText(frame, f'Video Progress: {percentage:.2f} %', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if results.pose_landmarks:
                left_elbow = extract_coordinates(results, LEFT_ELBOW_IDX)
                left_wrist = extract_coordinates(results, LEFT_WRIST_IDX)
                left_pinky = extract_coordinates(results, LEFT_PINKY_IDX)
                left_index = extract_coordinates(results, LEFT_INDEX_IDX)

                right_elbow = extract_coordinates(results, RIGHT_ELBOW_IDX)
                right_wrist = extract_coordinates(results, RIGHT_WRIST_IDX)
                right_pinky = extract_coordinates(results, RIGHT_PINKY_IDX)
                right_index = extract_coordinates(results, RIGHT_INDEX_IDX)

                process_wrist(frame, original_frame, width, height, left_elbow, left_wrist, left_pinky, left_index, left_angle_history, side="left")
                process_wrist(frame, original_frame, width, height, right_elbow, right_wrist, right_pinky, right_index, right_angle_history, side="right")

            cv2.imshow('MediaPipe Pose', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

if __name__ == "__main__":
    main()