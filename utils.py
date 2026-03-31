"""
utils.py — Helper Functions for Lane Detection Pipeline
=========================================================
Contains all image processing utilities used by lane_detection.py.

Functions:
    - apply_gaussian_blur
    - apply_canny_edge_detection
    - region_of_interest
    - apply_hough_transform
    - average_slope_intercept
    - make_line_coordinates
    - draw_lane_lines
    - display_info_overlay
"""

"""
utils.py — Helper Functions for Lane Detection Pipeline
=========================================================
Contains all image processing utilities used by lane_detection.py.
"""

import cv2
import numpy as np
import warnings


def apply_gaussian_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_canny_edge_detection(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)


def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(image, mask)


def apply_hough_transform(image, rho=2, theta=np.pi / 180,
                          threshold=100, min_line_length=40, max_line_gap=5):
    return cv2.HoughLinesP(
        image, rho=rho, theta=theta, threshold=threshold,
        lines=np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap
    )


def make_line_coordinates(image, line_params):
    slope, intercept = line_params
    if abs(slope) < 0.1:
        return None
    height = image.shape[0]
    width = image.shape[1]
    y1 = height
    y2 = int(height * 0.60)
    try:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    except (ZeroDivisionError, OverflowError, ValueError):
        return None
    x1 = max(-width, min(2 * width, x1))
    x2 = max(-width, min(2 * width, x2))
    return [int(x1), int(y1), int(x2), int(y2)]


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x1 == x2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            params = np.polyfit((x1, x2), (y1, y2), 1)
        slope = params[0]
        intercept = params[1]
        if abs(slope) < 0.4 or abs(slope) > 5.0:
            continue
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    averaged_lines = []
    if left_fit:
        left_avg = np.average(left_fit, axis=0)
        left_line = make_line_coordinates(image, left_avg)
        if left_line is not None:
            averaged_lines.append(('left', left_line, len(left_fit)))
    if right_fit:
        right_avg = np.average(right_fit, axis=0)
        right_line = make_line_coordinates(image, right_avg)
        if right_line is not None:
            averaged_lines.append(('right', right_line, len(right_fit)))
    return averaged_lines


def calculate_lane_center_offset(frame, lane_data):
    width = frame.shape[1]
    frame_center = width // 2
    left_x = None
    right_x = None
    for side, line, _ in lane_data:
        if side == 'left':
            left_x = line[0]
        elif side == 'right':
            right_x = line[0]
    if left_x is None or right_x is None:
        return 0, 0.0, "Unknown"
    lane_center = (left_x + right_x) // 2
    offset = lane_center - frame_center
    offset_percent = (offset / (width // 2)) * 100
    if abs(offset_percent) < 5:
        direction = "Centered"
    elif offset > 0:
        direction = "Drifting Right"
    else:
        direction = "Drifting Left"
    return offset, offset_percent, direction


def calculate_confidence(lane_data, total_possible=2):
    if not lane_data:
        return 0
    lane_score = (len(lane_data) / total_possible) * 60
    vote_scores = [min(votes / 10.0, 1.0) * 40 for _, _, votes in lane_data]
    avg_vote_score = np.mean(vote_scores) if vote_scores else 0
    return int(min(lane_score + avg_vote_score, 100))


def draw_lane_lines(frame, lane_data):
    line_image = np.zeros_like(frame)
    lines_only = [line for _, line, _ in lane_data]
    if lines_only:
        for line in lines_only:
            x1, y1, x2, y2 = [int(v) for v in line]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 8)
        if len(lines_only) == 2:
            left = lines_only[0]
            right = lines_only[1]
            pts = np.array([
                [int(left[0]),  int(left[1])],
                [int(left[2]),  int(left[3])],
                [int(right[2]), int(right[3])],
                [int(right[0]), int(right[1])]
            ], dtype=np.int32)
            cv2.fillPoly(line_image, [pts], (0, 180, 0))
    return cv2.addWeighted(frame, 0.8, line_image, 1, 0)


def draw_dashboard(frame, frame_count, total_frames, fps,
                   lane_data, offset, offset_pct, direction, confidence,
                   smoothed_offset_history):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 38), (20, 20, 20), -1)
    cv2.rectangle(frame, (0, 38), (w, 40), (0, 200, 80), -1)
    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
    bar_w = int((w - 20) * progress / 100)
    cv2.rectangle(frame, (10, 6), (10 + bar_w, 14), (0, 200, 80), -1)
    cv2.rectangle(frame, (10, 6), (w - 10, 14), (80, 80, 80), 1)
    cv2.putText(frame, "LANE DETECTION SYSTEM", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    cv2.putText(frame, f"Frame {frame_count}/{total_frames}  |  {progress:.0f}%  |  {fps} FPS",
                (w // 2 - 100, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # Left panel
    panel_x, panel_y, panel_w, panel_h = 8, 48, 210, 165
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (0, 200, 80), 1)

    cv2.putText(frame, "LANE STATUS", (panel_x + 8, panel_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 80), 1)
    cv2.line(frame, (panel_x + 8, panel_y + 24),
             (panel_x + panel_w - 8, panel_y + 24), (60, 60, 60), 1)

    left_detected  = any(s == 'left'  for s, _, _ in lane_data)
    right_detected = any(s == 'right' for s, _, _ in lane_data)
    left_color  = (0, 255, 80)  if left_detected  else (80, 80, 80)
    right_color = (0, 255, 80)  if right_detected else (80, 80, 80)
    cv2.putText(frame, "LEFT  LANE: " + ("DETECTED" if left_detected else "---"),
                (panel_x + 8, panel_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.4, left_color, 1)
    cv2.putText(frame, "RIGHT LANE: " + ("DETECTED" if right_detected else "---"),
                (panel_x + 8, panel_y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.4, right_color, 1)

    # Confidence bar
    cv2.putText(frame, "CONFIDENCE:", (panel_x + 8, panel_y + 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    conf_bar_w = int((panel_w - 20) * confidence / 100)
    conf_color = (0, 200, 80) if confidence > 60 else (0, 200, 220) if confidence > 30 else (0, 80, 220)
    cv2.rectangle(frame, (panel_x + 8, panel_y + 90),
                  (panel_x + panel_w - 8, panel_y + 102), (50, 50, 50), -1)
    if conf_bar_w > 0:
        cv2.rectangle(frame, (panel_x + 8, panel_y + 90),
                      (panel_x + 8 + conf_bar_w, panel_y + 102), conf_color, -1)
    cv2.putText(frame, f"{confidence}%", (panel_x + panel_w - 32, panel_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

    # Offset meter
    cv2.putText(frame, "CENTER OFFSET:", (panel_x + 8, panel_y + 122),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    meter_x = panel_x + 8
    meter_y = panel_y + 128
    meter_w = panel_w - 20
    meter_h = 12
    cv2.rectangle(frame, (meter_x, meter_y),
                  (meter_x + meter_w, meter_y + meter_h), (50, 50, 50), -1)
    mid = meter_x + meter_w // 2
    cv2.line(frame, (mid, meter_y), (mid, meter_y + meter_h), (100, 100, 100), 1)
    indicator_x = int(mid + (offset_pct / 100) * (meter_w // 2))
    indicator_x = max(meter_x + 2, min(meter_x + meter_w - 2, indicator_x))
    ind_color = (0, 200, 80) if abs(offset_pct) < 10 else (0, 180, 220) if abs(offset_pct) < 25 else (0, 60, 220)
    cv2.rectangle(frame, (indicator_x - 3, meter_y + 1),
                  (indicator_x + 3, meter_y + meter_h - 1), ind_color, -1)

    dir_color = (0, 200, 80) if direction == "Centered" else (0, 180, 220) if "Drift" in direction else (180, 180, 180)
    cv2.putText(frame, f"{direction}  ({offset_pct:+.1f}%)",
                (panel_x + 8, panel_y + 156),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, dir_color, 1)

    # Departure warning
    if abs(offset_pct) > 30 and len(lane_data) == 2:
        warn_x = w // 2 - 130
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (warn_x, h - 58), (warn_x + 260, h - 8), (0, 0, 180), -1)
        cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (warn_x, h - 58), (warn_x + 260, h - 8), (0, 80, 255), 2)
        cv2.putText(frame, "! LANE DEPARTURE WARNING !", (warn_x + 14, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        side_warn = ">> MOVE LEFT <<" if offset > 0 else "<< MOVE RIGHT >>"
        cv2.putText(frame, side_warn, (warn_x + 50, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 255), 1)

    # Mini offset history graph
    if len(smoothed_offset_history) > 2:
        graph_x, graph_y, graph_w, graph_h = w - 148, 48, 140, 70
        overlay3 = frame.copy()
        cv2.rectangle(overlay3, (graph_x, graph_y),
                      (graph_x + graph_w, graph_y + graph_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay3, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (graph_x, graph_y),
                      (graph_x + graph_w, graph_y + graph_h), (0, 200, 80), 1)
        cv2.putText(frame, "OFFSET HISTORY", (graph_x + 6, graph_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 80), 1)
        mid_y = graph_y + graph_h // 2
        cv2.line(frame, (graph_x + 4, mid_y), (graph_x + graph_w - 4, mid_y), (60, 60, 60), 1)
        vals = smoothed_offset_history[-(graph_w - 8):]
        step = (graph_w - 8) / max(len(vals) - 1, 1)
        for i in range(1, len(vals)):
            x1p = int(graph_x + 4 + (i - 1) * step)
            x2p = int(graph_x + 4 + i * step)
            y1p = int(mid_y - vals[i - 1] * 0.3)
            y2p = int(mid_y - vals[i] * 0.3)
            y1p = max(graph_y + 4, min(graph_y + graph_h - 4, y1p))
            y2p = max(graph_y + 4, min(graph_y + graph_h - 4, y2p))
            cv2.line(frame, (x1p, y1p), (x2p, y2p), (0, 200, 80), 1)

    # Pipeline label
    cv2.putText(frame, "Grayscale > Blur > Canny > ROI > HoughP > Average > Overlay",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1)

    return frame