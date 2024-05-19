from ultralytics import YOLO
from typing import List
import supervision as sv
import numpy as np

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path).to('cuda')
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames, batch_size=30):
        """Detects objects in a list of frames

        Args:
            frames (List[np.ndarray]): List of frames
            batch_size (int, optional): How many frames to process at a time. 
                Defaults to 30.

        Returns:
            List[np.ndarray]: List of frames with bounding boxes drawn
        """
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
            break
        return detections

    def get_object_tracks(self, frames):
        
         detections = self.detect_frames(frames)
        
         for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            
            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # Convert Goalkeeper to player
            for objects_num, class_id in enumerate(detection_supervision.class_id):
                if class_id == cls_names["goalkeeper"]:
                    detection_supervision.class_id[objects_num] = cls_names_inv["player"]
            
            detect
            
            print(detection_supervision)