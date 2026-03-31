"""
Lane Detection System — ADAS Edition
======================================
Real-time road lane detection with full ADAS-style dashboard HUD.
Pipeline: Grayscale > Gaussian Blur > Canny > ROI > Hough > Average > Overlay

Author: Pranay Gupta
Course: Computer Vision
"""

import cv2
import numpy as np
import argparse
import sys
import os
from collections import deque

from utils import (
    apply_gaussian_blur,
    apply_canny_edge_detection,
    region_of_interest,
    apply_hough_transform,
    average_slope_intercept,
    draw_lane_lines,
    draw_dashboard,
    calculate_lane_center_offset,
    calculate_confidence,
)

# Smoothing buffer — keeps last N frames of lane lines to reduce jitter
SMOOTH_N = 8
left_line_buffer  = deque(maxlen=SMOOTH_N)
right_line_buffer = deque(maxlen=SMOOTH_N)
offset_history    = deque(maxlen=200)


def smooth_line(buffer, new_line):
    """
    Temporally smooth a lane line using a rolling average buffer.
    Reduces jitter by averaging coordinates over last SMOOTH_N frames.
    """
    if new_line is not None:
        buffer.append(new_line)
    if not buffer:
        return None
    avg = np.mean(buffer, axis=0).astype(int)
    return avg.tolist()


def process_frame(frame):
    """
    Full pipeline for a single frame.
    Returns (annotated_frame, lane_data, offset, offset_pct, direction, confidence)
    """
    global left_line_buffer, right_line_buffer

    # Step 1: Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 2: Gaussian blur
    blurred = apply_gaussian_blur(gray, kernel_size=5)

    # Step 3: Canny edge detection
    edges = apply_canny_edge_detection(blurred, low_threshold=50, high_threshold=150)

    # Step 4: Region of interest
    height, width = edges.shape
    roi_vertices = np.array([[
        (int(width * 0.08), height),
        (int(width * 0.44), int(height * 0.58)),
        (int(width * 0.56), int(height * 0.58)),
        (int(width * 0.92), height)
    ]], dtype=np.int32)
    masked_edges = region_of_interest(edges, roi_vertices)

    # Step 5: Hough transform
    lines = apply_hough_transform(masked_edges)

    # Step 6: Average lines → (side, coords, votes)
    raw_lane_data = average_slope_intercept(frame, lines)

    # Step 7: Temporal smoothing
    raw_left  = next((l for s, l, _ in raw_lane_data if s == 'left'),  None)
    raw_right = next((l for s, l, _ in raw_lane_data if s == 'right'), None)
    votes_left  = next((v for s, _, v in raw_lane_data if s == 'left'),  0)
    votes_right = next((v for s, _, v in raw_lane_data if s == 'right'), 0)

    smooth_left  = smooth_line(left_line_buffer,  raw_left)
    smooth_right = smooth_line(right_line_buffer, raw_right)

    lane_data = []
    if smooth_left  is not None: lane_data.append(('left',  smooth_left,  votes_left))
    if smooth_right is not None: lane_data.append(('right', smooth_right, votes_right))

    # Step 8: Draw lanes
    result = draw_lane_lines(frame, lane_data)

    # Step 9: Compute analytics
    offset, offset_pct, direction = calculate_lane_center_offset(frame, lane_data)
    confidence = calculate_confidence(lane_data)
    offset_history.append(offset_pct)

    return result, lane_data, offset, offset_pct, direction, confidence


def process_video(input_path, output_path=None, show_preview=True):
    """Process a video file frame by frame."""
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        sys.exit(1)

    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video: {input_path}")
    print(f"[INFO] Resolution: {width}x{height} | FPS: {fps} | Frames: {total}")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"[INFO] Saving output to: {output_path}")

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        result, lane_data, offset, offset_pct, direction, confidence = process_frame(frame)

        result = draw_dashboard(
            result, frame_count, total, fps,
            lane_data, offset, offset_pct, direction, confidence,
            list(offset_history)
        )

        if writer:
            writer.write(result)

        if show_preview:
            cv2.imshow("Lane Detection — ADAS System  |  Press Q to quit", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Quit.")
                break

        if frame_count % 100 == 0:
            pct = frame_count / total * 100
            print(f"[INFO] {pct:.1f}%  ({frame_count}/{total})  |  Confidence: {confidence}%  |  {direction}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Processed {frame_count} frames.")
    if output_path and os.path.exists(output_path):
        print(f"[DONE] Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Lane Detection — ADAS System")
    parser.add_argument('--input',      '-i', required=True,  help='Input video/image path')
    parser.add_argument('--output',     '-o', default=None,   help='Output file path')
    parser.add_argument('--mode',       '-m', choices=['video', 'image'], default='video')
    parser.add_argument('--no-preview', action='store_true',  help='Disable preview window')
    args = parser.parse_args()

    print("=" * 52)
    print("   Lane Detection System — ADAS Edition")
    print("=" * 52)

    process_video(
        input_path=args.input,
        output_path=args.output,
        show_preview=not args.no_preview
    )


if __name__ == "__main__":
    main()
