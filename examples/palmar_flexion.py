import sys, os
from time import time
from collections import deque
import argparse

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyeyesweb.data_models.sliding_window import SlidingWindow

import numpy as np
import cv2
import mediapipe as mp
from pyeyesweb.analysis_primitives.synchronization import Synchronization
import dotenv

# Function to extract the y-coordinate of a specified keypoint (e.g., wrist) from the Mediapipe Pose results.
# The keypoint's visibility is checked to ensure it's sufficiently visible before processing.
def extract_coordinates(results, keypoint_idx):
    """Extract coordinates of a given keypoint."""
    keypoint = results.pose_landmarks.landmark[keypoint_idx]
    if keypoint.visibility > 0.5:  # Only consider keypoints with visibility above the threshold
        return (keypoint.x, keypoint.y, keypoint.z)  # Return the (x, y, z) coordinates
    return None  # Return None if the keypoint is not sufficiently visible

def angle_between(v0, v1, v2):
    """
    # Compute the normal vector of the plane containing v0 and v1
    normal = np.cross(v0, v1)
    normal_u = normal / np.linalg.norm(normal)

    # Compute the angle between the plane's normal and v2
    v2_u = v2 / np.linalg.norm(v2)
    angle = np.abs(np.arcsin(np.clip(np.dot(normal_u, v2_u), -1.0, 1.0)))
    return np.degrees(angle)
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(angle)


# Function to interpolate color between green and red based on the angle
def interpolate_color(value, min_value=15, max_value=45):
    """Interpolate color between green (aligned) and red (bent) using RGB."""
    value = np.clip(value, min_value, max_value)  # Clamp the value within the range
    ratio = max(0.0, min(1.0, (value - min_value) / (max_value - min_value)))  # Normalize the value
    r = int(255 * ratio)  # Red increases proportionally
    g = int(255 * (1 - ratio))  # Green decreases proportionally
    b = 0  # Blue remains 0
    return (b, g, r)  # Return the RGB color as a tuple

# Function to process and visualize wrist data
def process_wrist(frame, original_frame, width, height, elbow, wrist, pinky, index, shoulder, angle_history, side="left"):
    """Process and visualize wrist data."""
    if all(coord is not None for coord in [elbow, wrist, pinky, index, shoulder]):
        palm = ((pinky[0] + index[0]) / 2, (pinky[1] + index[1]) / 2, (pinky[2] + index[2]) / 2)
        arm = np.array([shoulder[i] - elbow[i] for i in range(3)])
        forearm = np.array([wrist[i] - elbow[i] for i in range(3)])
        wrist_palm = np.array([palm[i] - wrist[i] for i in range(3)])
        angle = angle_between(arm, forearm, wrist_palm)

        wrist_x = int(wrist[0] * width)
        wrist_y = int(wrist[1] * height)

        elbow_wrist_distance = np.linalg.norm(np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]]) * np.array([width, height]))
        radius = int(elbow_wrist_distance * 0.6)
        if radius > 7:
            center_x, center_y = wrist_x, wrist_y
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            wrist_area = cv2.bitwise_and(original_frame, original_frame, mask=mask)  # Use the unmodified frame

            x1, y1 = min(width, max(0, center_x - radius)), min(height, max(0, center_y - radius))
            x2, y2 = min(width, max(0, center_x + radius)), min(height, max(0, center_y + radius))
            if x2 > x1 and y2 > y1:
                cropped_area = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)  # Initialize with black
                cropped_area[max(0, -y1):min(y2, height) - y1, max(0, -x1):min(x2, width) - x1] = \
                    wrist_area[max(0, y1):min(y2, height), max(0, x1):min(x2, width)]

                output_size = (max(100, width // 10), max(100, width // 10))
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

# Function to process and display a frame
def process_and_display(frame, original_frame, mp_pose, results, height, width, current_frame, frame_count, live, args, left_angle_history, right_angle_history, LEFT_ELBOW_IDX, LEFT_WRIST_IDX, LEFT_PINKY_IDX, LEFT_INDEX_IDX, LEFT_SHOULDER_IDX, RIGHT_ELBOW_IDX, RIGHT_WRIST_IDX, RIGHT_PINKY_IDX, RIGHT_INDEX_IDX, RIGHT_SHOULDER_IDX):
    if not live:
        percentage = (current_frame / frame_count) * 100
        cv2.putText(frame, f'Video Progress: {percentage:.2f} %', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Add legend in top right
    legend_lines = [
        "p: play/pause",
        "f: frame fwd",
        "b: frame back", 
        "q: quit"
    ]
    font_scale = 0.6
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 0)
    line_height = 18
    start_y = 30
    start_x = width - 160
    for i, line in enumerate(legend_lines):
        y = start_y + i * line_height
        cv2.putText(frame, line, (start_x, y), font, font_scale, color, font_thickness)

    if results.pose_landmarks:
        left_elbow = extract_coordinates(results, LEFT_ELBOW_IDX)
        left_wrist = extract_coordinates(results, LEFT_WRIST_IDX)
        left_pinky = extract_coordinates(results, LEFT_PINKY_IDX)
        left_index = extract_coordinates(results, LEFT_INDEX_IDX)
        left_shoulder = extract_coordinates(results, LEFT_SHOULDER_IDX)

        right_elbow = extract_coordinates(results, RIGHT_ELBOW_IDX)
        right_wrist = extract_coordinates(results, RIGHT_WRIST_IDX)
        right_pinky = extract_coordinates(results, RIGHT_PINKY_IDX)
        right_index = extract_coordinates(results, RIGHT_INDEX_IDX)
        right_shoulder = extract_coordinates(results, RIGHT_SHOULDER_IDX)

        process_wrist(frame, original_frame, width, height, left_elbow, left_wrist, left_pinky, left_index, left_shoulder, left_angle_history, side="left" if not live else "right")
        process_wrist(frame, original_frame, width, height, right_elbow, right_wrist, right_pinky, right_index, right_shoulder, right_angle_history, side="right" if not live else "left")

        if args.draw_skeleton:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    cv2.imshow('MediaPipe Pose', frame)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Process a video file to analyze wrist movement.")
    parser.add_argument(
        "--input_file", type=str, help="Path to the input video file. If not specified tries to use the default input device"
        )
    parser.add_argument("--draw_skeleton", action="store_true", 
                        help="Enable drawing the skeleton on the frame.")
    args = parser.parse_args()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    live = args.input_file is None
    cap = cv2.VideoCapture(args.input_file if not live else 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    LEFT_ELBOW_IDX, LEFT_WRIST_IDX, LEFT_PINKY_IDX, LEFT_INDEX_IDX, LEFT_SHOULDER_IDX = 13, 15, 17, 19, 11
    RIGHT_ELBOW_IDX, RIGHT_WRIST_IDX, RIGHT_PINKY_IDX, RIGHT_INDEX_IDX, RIGHT_SHOULDER_IDX = 14, 16, 18, 20, 12

    left_angle_history = deque(maxlen=5)
    right_angle_history = deque(maxlen=5)

    # Read and process first frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read first frame")
        return

    original_frame = frame.copy()
    if live:
        frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    height, width, _ = image.shape
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if not live else 0
    process_and_display(frame, original_frame, mp_pose, results, height, width, current_frame, frame_count, live, args, left_angle_history, right_angle_history, LEFT_ELBOW_IDX, LEFT_WRIST_IDX, LEFT_PINKY_IDX, LEFT_INDEX_IDX, LEFT_SHOULDER_IDX, RIGHT_ELBOW_IDX, RIGHT_WRIST_IDX, RIGHT_PINKY_IDX, RIGHT_INDEX_IDX, RIGHT_SHOULDER_IDX)

    is_playing = True

    try:
        while True:
            key = cv2.waitKey(0 if not is_playing else 1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                is_playing = not is_playing
            elif key == ord('f') and not is_playing and not live:
                current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos + 1)
                ret, frame = cap.read()
                if ret:
                    original_frame = frame.copy()
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)
                    height, width, _ = image.shape
                    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    process_and_display(frame, original_frame, mp_pose, results, height, width, current_frame, frame_count, live, args, left_angle_history, right_angle_history, LEFT_ELBOW_IDX, LEFT_WRIST_IDX, LEFT_PINKY_IDX, LEFT_INDEX_IDX, LEFT_SHOULDER_IDX, RIGHT_ELBOW_IDX, RIGHT_WRIST_IDX, RIGHT_PINKY_IDX, RIGHT_INDEX_IDX, RIGHT_SHOULDER_IDX)
            elif key == ord('b') and not is_playing and not live:
                current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if current_pos > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
                    ret, frame = cap.read()
                    if ret:
                        original_frame = frame.copy()
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = pose.process(image)
                        height, width, _ = image.shape
                        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        process_and_display(frame, original_frame, mp_pose, results, height, width, current_frame, frame_count, live, args, left_angle_history, right_angle_history, LEFT_ELBOW_IDX, LEFT_WRIST_IDX, LEFT_PINKY_IDX, LEFT_INDEX_IDX, LEFT_SHOULDER_IDX, RIGHT_ELBOW_IDX, RIGHT_WRIST_IDX, RIGHT_PINKY_IDX, RIGHT_INDEX_IDX, RIGHT_SHOULDER_IDX)

            if is_playing:
                ret, frame = cap.read()
                if not ret:
                    break
                original_frame = frame.copy()
                if live:
                    frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                height, width, _ = image.shape
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if not live else 0
                process_and_display(frame, original_frame, mp_pose, results, height, width, current_frame, frame_count, live, args, left_angle_history, right_angle_history, LEFT_ELBOW_IDX, LEFT_WRIST_IDX, LEFT_PINKY_IDX, LEFT_INDEX_IDX, LEFT_SHOULDER_IDX, RIGHT_ELBOW_IDX, RIGHT_WRIST_IDX, RIGHT_PINKY_IDX, RIGHT_INDEX_IDX, RIGHT_SHOULDER_IDX)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

if __name__ == "__main__":
    main()