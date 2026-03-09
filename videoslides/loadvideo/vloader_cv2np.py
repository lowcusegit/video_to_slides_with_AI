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

# def select_frames(all_frame_changes, height=None, distance=6):
#     height = height if height is not None else (all_frame_changes[:,1].mean()+all_frame_changes[:,1].std())
#     #height= np.percentile(all_frame_changes[:,1], pc)
#     frame_peaks, _ = signal.find_peaks(all_frame_changes[:,1],height=height, distance=distance)
#     pixel_peaks, _ = signal.find_peaks(all_frame_changes[:,0], distance=distance)
#     sections = frame_peaks.shape[0] # not include last section
#     selection_frames = []
    
#     if sections < 1 : # when no sharp change
#         selection_frames.append(all_frame_changes.shape[0]//2)
#         return selection_frames

#     # from beginning to 1st sharp change
#     pixel_peaks_in_section = pixel_peaks[ (pixel_peaks < frame_peaks[0]) ]
#     f = (frame_peaks[0])//2 if pixel_peaks_in_section.size == 0 else (pixel_peaks_in_section[-1]+frame_peaks[0])//2
#     selection_frames.append(f) 
    
#     for i in range(1,sections):
#         pixel_peaks_in_section = pixel_peaks[ (pixel_peaks > frame_peaks[i-1]) & (pixel_peaks < frame_peaks[i]) ]
#         f = (frame_peaks[i]-3) if pixel_peaks_in_section.size == 0 else (pixel_peaks_in_section[-1]+frame_peaks[i])//2
#         selection_frames.append(f)
#     selection_frames.append(max(frame_peaks[-1], pixel_peaks[-1], (all_frame_changes.shape[0]-2) )+1)
#     return selection_frames

# def create_powerpoint_with_select_frames(video_path, selection_frames, output_pptx_name="presentation.pptx"):
#     """
#     Creates a PowerPoint presentation and inserts images into slides.
#     Args:
#         selection_frames: A list of frame numbers you want to insert.
#         output_filename: The name of the PowerPoint file to create.
#     """
#     # Create a PowerPoint presentation object
#     prs = Presentation()
#     prs.slide_width = Inches(7.5*16/9)
#     prs.slide_height = Inches(7.5)
    
#     blank_slide_layout = prs.slide_layouts[6]  # Index 6 usually corresponds to a blank slide

#     # Open the video file
#     video = cv2.VideoCapture(video_path)

#     # Iterate over selected frame in the video
#     for f in selection_frames:
#         video.set(cv2.CAP_PROP_POS_FRAMES, f)
#         success, frame = video.read()
#         if success:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image = Image.fromarray(frame_rgb)
#             # Create an in-memory byte stream to save to memory
#             image_stream = BytesIO()
#             image.save(image_stream, format='PNG')
#             image_stream.seek(0)  # Reset the stream position to the beginning

#             slide = prs.slides.add_slide(blank_slide_layout)

#             # Set image position and size
#             left, top = Inches(0), Inches(0)
#             width, height = Inches(7.5*16/9), Inches(7.5)

#             slide.shapes.add_picture(image_stream, left, top, width, height)
            
#     prs.save(output_pptx_name)
#     print(f"PowerPoint presentation saved to {output_pptx_name}")

#     # Release the video capture object
#     video.release()

# def video_to_pptx_pipeline(video_path):
#     print(video_path)
#     export_dir = os.path.splitext(video_path)[0]
#     output_pptx_name = export_dir + '.pptx'

#     # analyze video
#     t0=time.time()
#     all_frame_changes = analyze_frames(video_path)
#     all_frame_changes = np.array(all_frame_changes, dtype=np.float32)
#     t1=time.time()
#     print('analyze video: ', t1-t0)
    
#     # select frame
#     t0=time.time()
#     FRAME_RATE = 30
#     distance = FRAME_RATE//5
#     height = all_frame_changes[:,1].mean()+all_frame_changes[:,1].std()
#     selection_frames = select_frames(all_frame_changes, height=height, distance=distance)
#     t1=time.time()
#     print('select frame: ', t1-t0)
#     # v2.0
#     # frame to pptx
#     t0=time.time()
#     create_powerpoint_with_select_frames(video_path, selection_frames, output_pptx_name=output_pptx_name)
    
#     t1=time.time()
#     print('frames to pptx: ',t1-t0)

# def main():
#     parser = argparse.ArgumentParser(description="Analyze videos and generate PowerPoint presentations from selected frames.")
#     parser.add_argument("--video_folder", type=str, required=True,
#                         help="Path to the folder containing video files.")
#     parser.add_argument("--video_extension", type=str, nargs='+', default=['.mp4', '.MP4'],
#                         help="List of video file extensions to include (e.g., .mp4 .avi).")
#     parser.add_argument("--num_of_thread", type=int, default=2,
#                         help="Number of threads to use for parallel video processing.")
    
#     args = parser.parse_args()

#     video_folder = args.video_folder
#     # Normalize extensions to lowercase for consistent matching
#     video_extension = [ext.lower() for ext in args.video_extension] 
#     num_of_thread = args.num_of_thread

#     print(f"Searching for videos in: {video_folder}")
#     print(f"Including extensions: {video_extension}")
#     print(f"Using {num_of_thread} threads.")

#     video_list = recursive_list_file_paths(
#         video_folder,
#         extension=video_extension)

#     if not video_list:
#         print(f"No video files found with extensions {video_extension} in {video_folder}")
#         return

#     print(f"Found {len(video_list)} video files.")

#     # Multi-threading the video_to_pptx_pipeline
#     with ThreadPoolExecutor(max_workers=num_of_thread) as executor:
#         # Submit each video processing task to the executor
#         futures = {executor.submit(video_to_pptx_pipeline, v): v for v in video_list}
        
#         # Wait for tasks to complete and handle results/exceptions
#         for future in concurrent.futures.as_completed(futures):
#             video_path = futures[future]
#             try:
#                 future.result() # This will re-raise any exception that occurred in the thread
#             except Exception as exc:
#                 print(f'Error processing {video_path}: {exc}')
#             else:
#                 print(f'Finished processing {video_path}.')

# if __name__ == '__main__':
#     main()
