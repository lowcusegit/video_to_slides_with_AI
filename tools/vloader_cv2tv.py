import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

default_transform= transforms.ToTensor()

class VideoFrameDataset(Dataset):
    """
    A Dataset class that loads frames from an MP4 video file.
    Each __getitem__ call retrieves a specific frame by index.
    """
    def __init__(self, video_path, transform=default_transform):
        """
        Args:
            video_path (str): Path to the video file.
            transform (callable, optional): Optional transform to be applied on a frame.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        self.video_path = video_path
        self.transform = transform
        
        # Open video to get metadata
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")
            
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        # Not keep the cap object open here because it doesn't play well with multiprocessing (DataLoader workers).
        self.cap = None

    def __len__(self):
        return self.frame_count

    def __getitem__(self, idx):
        # Lazy initialization of VideoCapture for worker safety
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_path)
            
        # Set the reader to the specific frame index
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        
        if not ret:
            # Return a blank frame or handle error if frame reading fails
            return torch.zeros((3, self.height, self.width)), idx

        # OpenCV reads in BGR format; convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image to make it compatible with torchvision transforms
        frame = Image.fromarray(frame)
        
        # Transform and convert to tensor
        frame = self.transform(frame)
            
        return frame, idx

    def __del__(self):
        if self.cap is not None:
            self.cap.release()
