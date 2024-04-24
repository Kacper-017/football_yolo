from ultralytics import YOLO
from torch.cuda import is_available

device = 'cuda' if is_available() else 'cpu'
print(f"Using {device} device")

model = YOLO('yolov8x').to(device)

results = model.predict('input_videos/08fd33_4.mp4', save = True)
print(results[0])
print("===================")
for box in results[0].boxes:
    print(box)

