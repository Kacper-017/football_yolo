from utils import read_video, write_video
from trackers import Tracker

def main():
    
    video_frames = read_video("input_videos/08fd33_4.mp4")
    
    tracker = Tracker("models/best.pt")
    
    tracks = tracker.get_object_tracks(video_frames)
    
    write_video(video_frames, "output_videos/08fd33_4.mp4")
    
if __name__ == "__main__":
    main()
    