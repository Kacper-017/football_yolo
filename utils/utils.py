import cv2
from typing import List
import numpy as np


def read_video(video_path: str) -> List[np.ndarray]:
    """Reads a video file and returns a list of frames

    Args:
        video_path (str): Path to the video file

    Returns:
        : List of frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def write_video(frames: List[np.ndarray], output_path: str) -> None:
    """Writes frames to a video file

    Args:
        frames (List[np.ndarray]): List of frames
        output_path (str): Path to the output video file
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 24,
                          (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()
