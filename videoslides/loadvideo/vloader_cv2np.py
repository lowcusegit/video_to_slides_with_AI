import os, time
import cv2
import numpy as np

from io import BytesIO  # or from io import StringIO for text-based images


def recursive_list_file_paths(PATH, extension=[]):
    '''
    PATH : string , path to search
    extension : list of strings, file extension(s) to included
    '''
    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH) for f in filenames if os.path.splitext(f)[1] in extension] if len(extension)>0 else \
    [os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH) for f in filenames]
    return result

def video_info(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise IOError(f"Could not open video file: {video_path}")
    v_info={
        "frame_count" : int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps" : video.get(cv2.CAP_PROP_FPS),
        "width" : int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height" : int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    
    video.release()
    
    return v_info

# load video frames and identify changes
def cap_frame_change(video_path, pool=None):
    """
    Args:
    video_path : path of a video file
    pool (tuple): H x W
    """
    # Open the video file
    video = cv2.VideoCapture(video_path)
    frames = []
    # frames = {'changes':[], 'frames':[]}
    success, frame = video.read()
    # frame=frame.astype(np.int16)
    # frame (H,W,BGR)

    if pool is not None:
        padding = (
            (0, (pool[0]-(frame.shape[0]%pool[0]))%pool[0] ),
            (0, (pool[1]-(frame.shape[1]%pool[1]))%pool[1] )
        )
        padding = None if np.sum(padding)==0 else padding
        reshape=(frame.shape[0]//pool[0], pool[0], frame.shape[1]//pool[1], pool[1] ) if (
            padding is None
        ) else ((frame.shape[0]+padding[0][1])//pool[0], pool[0], (frame.shape[1]+padding[1][1])//pool[1], pool[1] )
    
    # Iterate over each frame in the video
    if pool is not None:
        while success:
            # Append the current frame to the list
            last_frame = frame
            # Read the next frame
            success, frame = video.read()
            if success:
                # evaluate changes of frames
                # frame=frame.astype(np.int16)
                frame_diff = cv2.absdiff(last_frame, frame).sum(axis=-1, dtype=np.int16) if padding is None else np.pad(
                    cv2.absdiff(last_frame, frame).sum(axis=-1, dtype=np.int16), padding, 'constant', constant_values=0 )
                
                frame_diff=frame_diff.reshape(
                    reshape,
                    order='C')
                max_diff = frame_diff.max(axis=(1,3)).flatten()

            frames.append(max_diff)

    else:
        while success:
            # Append the current frame to the list
            last_frame = frame
            # Read the next frame
            success, frame = video.read()
            if success:
                # evaluate changes of frames
                # frame=frame.astype(np.int16)
                frame_diff = cv2.absdiff(last_frame, frame).sum(axis=-1, dtype=np.int16) # sum change over pixel over rgb channels

                # None then no pooling/ no dimension reduction
                max_diff = frame_diff.flatten()

            frames.append(max_diff)

    frames=np.stack(frames, axis=0)

    # Release the video capture object
    video.release()
    return frames

def cap_frame(video_path, pool=None):
    """
    Args:
    video_path : path of a video file
    pool (tuple): H x W
    """
    # Open the video file
    video = cv2.VideoCapture(video_path)
    frames = []

    success, frame = video.read()
    # frame=frame.astype(np.int16)
    # frame (H,W,BGR)

    if pool is not None:
        padding = (
            (0, (pool[0]-(frame.shape[0]%pool[0]))%pool[0] ),
            (0, (pool[1]-(frame.shape[1]%pool[1]))%pool[1] )
        )
        padding = None if np.sum(padding)==0 else padding
        reshape=(frame.shape[0]//pool[0], pool[0], frame.shape[1]//pool[1], pool[1] ) if (
            padding is None
        ) else ((frame.shape[0]+padding[0][1])//pool[0], pool[0], (frame.shape[1]+padding[1][1])//pool[1], pool[1] )
    
    # Iterate over each frame in the video
    if pool is not None:
        while success:
            # reduce size, color, flatten
            avg_pix = cv2.cvtColor(cv2.resize( frame, (reshape[2], reshape[0]), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY).flatten()
            frames.append(avg_pix)
            
            # Read the next frame
            success, frame = video.read()

    else:
        while success:
            # reduce color, flatten
            avg_pix = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY).flatten()
            frames['frames'].append(avg_pix)
            # Read the next frame
            success, frame = video.read()

    frames=np.stack(frames, axis=0)
    # Release the video capture object
    video.release()
    return frames

def weighted_change(frame_change_array, stable_frame=15, threshold_pix=1.0, threshold_count=13):
    """
    Args:
    frame_change_array: flatten features (frames, features) or (frames, ...,features)
    stable_frame (optional): number of frames before  a frame to evaluate the stability of pixels/features
    threshold_pix (optional): max changes of pixels/features considered as unchange
    threshold_count (optional): min count of unchange considered as stable
    """
    length=frame_change_array.shape[0]
    idx=np.arange(length - stable_frame).reshape(-1,1) + np.arange(stable_frame).reshape(1,-1)
    stable_frame_mask= np.sum( (frame_change_array[idx,:] <= threshold_pix), axis=1 ) >= threshold_count
    weighted_frame_change_array = np.sum(frame_change_array[ int(stable_frame):, :] * stable_frame_mask, axis=1)
    return weighted_frame_change_array
