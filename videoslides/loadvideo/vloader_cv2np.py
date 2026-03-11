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
def gray_downsample_frame_producer(cap, q, t_w, t_h, n_f):
    """
    Producer function to read and preprocess video frames.
    Args:
    cap: OpenCV VideoCapture object
    q: Thread-safe queue to put processed frames
    t_w, t_h: Target width and height for resizing frames
    n_f: Number of frames to skip (e.g., 1 for every frame, 2 for every other frame)

    output thru queue, no return value
    """
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

    def BGR_downsample_frame_producer(cap, q, t_w, t_h, n_f):

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
            # Ensure image quality for text detection
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(frame, (t_w, t_h), interpolation=cv2.INTER_AREA)
            
            # Put the small, processed frame into the queue
            q.put(small)

            for _ in range(n_f - 1):  # Skip the next n_f-1 frames
                if not cap.grab(): # grab() decodes the bitstream but skips pixel processing
                    q.put(None)  # Sentinel value to signal end of video
                    break
        cap.release()

def BGR_downsample_frame_producer(cap, q, t_w, t_h, n_f):

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
        # Ensure image quality for text detection
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(frame, (t_w, t_h), interpolation=cv2.INTER_AREA)
        
        # Put the small, processed frame into the queue
        q.put(small)

        for _ in range(n_f - 1):  # Skip the next n_f-1 frames
            if not cap.grab(): # grab() decodes the bitstream but skips pixel processing
                q.put(None)  # Sentinel value to signal end of video
                break
    cap.release()

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
    
    # Define the Producer function outside


    # 2. Start the Producer Thread
    producer_thread = threading.Thread(
        target=gray_downsample_frame_producer, 
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

def text_detection_db(video_path, model_path, pool=None, v_info=None, next_frame=5):
    # frame_producer
    """
    Args:
    video_path : path of a video file
    model_path : path of the text detection model (e.g., DB18)
    pool (tuple): H x W
    v_info (dict): Video information dictionary with keys 'frame_count', 'fps', 'width', 'height'
    next_frame (int): Step size for frame sampling (e.g., 1 for every frame, 2 for every other frame)
    Returns:
    feature_array: array of shape (frames, features), features include [box_count, overlap_count, missing_count, new_count]
    """
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Could not open video file: {video_path}")
        return np.array([])

    # 1. Metadata setup
    if v_info is None:
        v_info = video_info(video_path)
    # total_frames = v_info["frame_count"]
    width = v_info["width"]
    height = v_info["height"]

    if not( isinstance(next_frame, int) ) or (next_frame < 1):
        print(f"Invalid next_frame value: {next_frame}. It should be a positive integer, i.e. >=1.")
        next_frame = 1
    # total_frames = (total_frames + next_frame - 1) // next_frame  # Adjust total frames based on sampling step
    # success, frame = video.read()
    # frame (H,W,BGR)

    # Set input image size (width, height) as of the pretrained text detection model, video need not be resized to this size
    # However, resize the frame in the producer thread before putting it into the queue can reduce additional resizing overhead in consumer.
    inputSize = (320,320)#(target_w, target_h)
    if pool is None:
        target_w, target_h = inputSize
    else:
        # pool = (2,2) # default pool size for feature extraction
        pool_h, pool_w = pool
        target_w, target_h = width // pool_w, height // pool_h
    
    # Pre-allocate results array
    out_features = []

    # Load pre-trained models
    textDetectorDB = cv2.dnn_TextDetectionModel_DB( model_path) #"./configmodel/DB_TD500_resnet18.onnx"
    bin_thresh = 0.3
    poly_thresh = 0.5
    mean = (122.67891434, 116.66876762, 104.00698793)
    textDetectorDB.setBinaryThreshold(bin_thresh).setPolygonThreshold(poly_thresh)
    textDetectorDB.setInputParams(1.0/255, inputSize, mean, True)

    # Thread-safe queue for pre-processed frames
    # Limit maxsize to prevent memory bloat if the producer is much faster than consumer
    frame_queue = queue.Queue(maxsize=128) 

    # Define the Producer function outside   

    # 2. Start the Producer Thread
    producer_thread = threading.Thread(
        target=BGR_downsample_frame_producer, 
        args=(video, frame_queue, target_w, target_h, next_frame),
        daemon=True
    )
    producer_thread.start()

    # 3. Consumer Logic (Main Thread)

    while True:
        curr_small = frame_queue.get()
        
        if curr_small is None: # Check for sentinel
            break
        
        # Calculate bounding boxes (CPU bound)
        # boxes, _ = textDetectorDB.detect(curr_small)
        out_features.append( textDetectorDB.detect(curr_small) ) # boxes and confidence scores, we will process them later.
        
    producer_thread.join()
    
    return out_features


# As a simpler alternative, we can just return the processed frames without calculating differences.
def cap_frame(video_path, pool=None, v_info=None, next_frame=1):
    """
    Args:
    video_path : path of a video file
    pool (tuple): H x W
    v_info (dict): Video information dictionary with keys 'frame_count', 'fps', 'width', 'height'
    next_frame (int): Step size for frame sampling (e.g., 1 for every frame, 2 for every other frame)
    Returns:
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
    
    # # Define the Producer function
    # def frame_producer(cap, q, t_w, t_h, n_f):
    #     f_count = 0
    #     while True:
    #         # OPTIMIZATION: read frames sequentially and skip as needed.
    #         # This avoids the overhead of random access and allows for more efficient decoding.
    #         # This is especially beneficial when skipping a few frames (e.g. <30)
    #         # cap.set(cv2.CAP_PROP_POS_FRAMES, f_count)
    #         f_count += n_f
    #         success, frame = cap.read()
    #         if not success:
    #             q.put(None) # Sentinel value to signal end of video
    #             break
            
    #         # OPTIMIZATION: Process at the source (Producer side)
    #         # This minimizes the amount of data sent through the queue (RAM)
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         small = cv2.resize(gray, (t_w, t_h), interpolation=cv2.INTER_AREA)
            
    #         # Put the small, processed frame into the queue
    #         q.put(small)

    #         for _ in range(n_f - 1):  # Skip the next n_f-1 frames
    #             if not cap.grab(): # grab() decodes the bitstream but skips pixel processing
    #                 q.put(None)  # Sentinel value to signal end of video
    #                 break
    #     cap.release()

    # 2. Start the Producer Thread
    producer_thread = threading.Thread(
        target=gray_downsample_frame_producer, 
        args=(video, frame_queue, target_w, target_h, next_frame),
        daemon=True
    )
    producer_thread.start()

    # 3. Consumer Logic (Main Thread)
    count = 0
    # prev_small = frame_queue.get() # Get the first frame
    
    # if prev_small is None:
    #     return np.array([])

    while True:
        curr_small = frame_queue.get()
        
        if curr_small is None: # Check for sentinel
            break
        
        # # Calculate difference (CPU bound)
        # diff = cv2.absdiff(prev_small, curr_small)
        
        # Store in pre-allocated array
        if count < out_changes.shape[0]:
            out_changes[count] = curr_small.flatten()
        
        prev_small = curr_small
        count += 1

    producer_thread.join()
    
    # Trim in case frame count was slightly off
    return out_changes[:count]

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
