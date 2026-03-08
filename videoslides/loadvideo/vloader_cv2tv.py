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
            print(f"Warning: Could not read frame at index {idx}. Returning blank frame.")
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

def batch_contrast(x:torch.Tensor)->torch.Tensor:
    # batch_size=x.shape[0]
    x= torch.abs(x[1::2,...]-x[0:-1:2,...])
    return x

def batch_avg(x:torch.Tensor, n=2)->torch.Tensor:
    """
    args:
    x (torch.Tensor) : input tesor in the shape (batch_size, feature) or (batch_size, channel,...)
    n (int, optional) : n samples for each mean, unless the bach 
    return:
    x (torch.Tensor) : output tesor in the shape (reduced_size, feature) or (reduced_size, channel,...)
    """
    batch_size=x.shape[0]
    # when it is not for expected average 
    if batch_size < n:
        # not enough for average
        x = torch.mean(x ,dim=0, keepdim=True) #if n>=2 else x
        return x
    reduced_size = batch_size // n
    x = x[:(reduced_size*n),...].reshape( tuple([reduced_size,n] + list(x.shape[1:]) ) ).mean(dim=1, keepdim=False )
    return x
    