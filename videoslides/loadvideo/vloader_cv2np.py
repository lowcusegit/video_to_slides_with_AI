import os #, time
import cv2
import numpy as np
# Producer-Consumer Pattern for I/O-bound tasks like video frame reading and processing
import threading
import queue
# from io import BytesIO  # or from io import StringIO for text-based images


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
# Optimized frame change detection using a Producer-Consumer pattern.
# - Producer Thread: Decodes, Grayscales, and Resizes frames.
# - Consumer Thread: Calculates differences and stores results.
# - Decouples I/O (Disk/Decoding) from CPU (Analysis).
def cap_frame_change(video_path, pool=None, v_info=None, next_frame=1):
    """
    Args:
    video_path : path of a video file
    pool (tuple): H x W
    v_info (dict): Video information dictionary with keys 'frame_count', 'fps', 'width', 'height'
    next_frame (int): Step size for frame sampling (e.g., 1 for every frame, 2 for every other frame)
    Returns:
    frame_change_array: array of shape (frames, features)
    """
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Could not open video file: {video_path}")
        return np.array([])

    # 1. Metadata setup
    if v_info is None:
        v_info = video_info(video_path)
    total_frames = v_info["frame_count"]
    width = v_info["width"]
    height = v_info["height"]

    if not( isinstance(next_frame, int) ) or (next_frame < 1):
        print(f"Invalid next_frame value: {next_frame}. It should be a positive integer. Defaulting to 1.")
        next_frame = 1
    total_frames = (total_frames + next_frame - 1) // next_frame  # Adjust total frames based on sampling step
    # success, frame = video.read()
    # frame (H,W,BGR)

    pool_h, pool_w = pool
    target_w, target_h = width // pool_w, height // pool_h
    
    # Pre-allocate results array
    out_changes = np.zeros((max(0, total_frames), target_w * target_h), dtype=np.uint8)
    
    # Thread-safe queue for pre-processed frames
    # Limit maxsize to prevent memory bloat if the producer is much faster than consumer
    frame_queue = queue.Queue(maxsize=128) 
    
    # Define the Producer function
    def frame_producer(cap, q, t_w, t_h, n_f):
        f_count = 0
        while True:
            # OPTIMIZATION: read frames sequentially and skip as needed.
            # This avoids the overhead of random access and allows for more efficient decoding.
            # This is especially beneficial when skipping a few frames (e.g. <30)
            # cap.set(cv2.CAP_PROP_POS_FRAMES, f_count)
            f_count += n_f
            success, frame = cap.read()
            if not success:
                q.put(None) # Sentinel value to signal end of video
                break
            
            # OPTIMIZATION: Process at the source (Producer side)
            # This minimizes the amount of data sent through the queue (RAM)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (t_w, t_h), interpolation=cv2.INTER_AREA)
            
            # Put the small, processed frame into the queue
            q.put(small)

            for _ in range(n_f - 1):  # Skip the next n_f-1 frames
                if not cap.grab(): # grab() decodes the bitstream but skips pixel processing
                    q.put(None)  # Sentinel value to signal end of video
                    break
        cap.release()

    # 2. Start the Producer Thread
    producer_thread = threading.Thread(
        target=frame_producer, 
        args=(video, frame_queue, target_w, target_h, next_frame),
        daemon=True
    )
    producer_thread.start()

    # 3. Consumer Logic (Main Thread)
    count = 1
    prev_small = frame_queue.get() # Get the first frame
    
    if prev_small is None:
        return np.array([])

    while True:
        curr_small = frame_queue.get()
        
        if curr_small is None: # Check for sentinel
            break
        
        # Calculate difference (CPU bound)
        diff = cv2.absdiff(prev_small, curr_small)
        
        # Store in pre-allocated array
        if count < out_changes.shape[0]:
            out_changes[count] = diff.flatten()
        
        prev_small = curr_small
        count += 1

    producer_thread.join()
    
    # Trim in case frame count was slightly off
    return out_changes[:count]

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
    Returns:
    weighted_frame_change_array: array of shape (frames,), i.e. remain length, but reduce features 
    """
    length=frame_change_array.shape[0]
    stable_frame_mask=np.full(frame_change_array.shape, True, dtype=bool)

    idx=np.arange(length - stable_frame).reshape(-1,1) + np.arange(stable_frame).reshape(1,-1)
    stable_frame_mask[int(stable_frame):,:]= np.sum( (frame_change_array[idx,:] <= threshold_pix), axis=1 ) >= threshold_count
    weighted_frame_change_array = np.sum(frame_change_array * stable_frame_mask, axis=1, dtype=np.float32)
    return weighted_frame_change_array
