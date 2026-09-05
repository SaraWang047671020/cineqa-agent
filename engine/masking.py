import cv2
import numpy as np
import os
from typing import List, Tuple

def generate_error_mask_video(
    source_video_path: str,
    output_mask_path: str,
    frame_boxes: List[Tuple[float, list]],
    padding_percent: float = 0.15,
    max_area_fraction: float = 0.6
) -> str:
    """
    Generates a black-and-white mask video based on bounding boxes over time.
    Uses linear interpolation for boxes between timestamps.
    
    Args:
        source_video_path: Path to the original video.
        output_mask_path: Path to save the mask video.
        frame_boxes: List of tuples (timestamp_seconds, [ymin, xmin, ymax, xmax] normalized 0-1000). Must be sorted by timestamp.
        padding_percent: Extra padding to ensure the entire object is masked.
        max_area_fraction: Maximum allowed area of the box relative to the frame. If exceeded, masking is skipped (returns black/unmasked) for safety against LLM hallucinations.
    """
    if not os.path.exists(source_video_path):
        raise FileNotFoundError(f"Source video not found: {source_video_path}")
        
    if not frame_boxes:
        raise ValueError("frame_boxes cannot be empty")

    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {source_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mask_path, fourcc, fps, (width, height))

    def interpolate_box(t):
        if len(frame_boxes) == 1:
            return frame_boxes[0][1]
        if t <= frame_boxes[0][0]:
            return frame_boxes[0][1]
        if t >= frame_boxes[-1][0]:
            return frame_boxes[-1][1]
        for k in range(len(frame_boxes) - 1):
            t0, b0 = frame_boxes[k]
            t1, b1 = frame_boxes[k + 1]
            if t0 <= t <= t1:
                ratio = (t - t0) / max(1e-6, t1 - t0)
                return [b0[j] + (b1[j] - b0[j]) * ratio for j in range(4)]
        return frame_boxes[-1][1]

    frame_idx = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps
        mask_frame = np.zeros((height, width, 3), dtype=np.uint8)

        if frame_boxes[0][0] <= current_time <= frame_boxes[-1][0]:
            box = interpolate_box(current_time)
            
            # Apply padding
            ymin_n, xmin_n, ymax_n, xmax_n = box
            box_width = xmax_n - xmin_n
            box_height = ymax_n - ymin_n
            
            pad_x = box_width * padding_percent
            pad_y = box_height * padding_percent
            
            xmin_n = max(0, xmin_n - pad_x)
            xmax_n = min(1000, xmax_n + pad_x)
            ymin_n = max(0, ymin_n - pad_y)
            ymax_n = min(1000, ymax_n + pad_y)
            
            area_fraction = ((xmax_n - xmin_n) * (ymax_n - ymin_n)) / (1000.0 * 1000.0)
            
            if area_fraction <= max_area_fraction:
                # Convert to actual pixels
                ymin = int(ymin_n * height / 1000.0)
                xmin = int(xmin_n * width / 1000.0)
                ymax = int(ymax_n * height / 1000.0)
                xmax = int(xmax_n * width / 1000.0)
                cv2.rectangle(mask_frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)

        out.write(mask_frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[Masking] Successfully created pure mathematical mask at {output_mask_path}")
    return output_mask_path
