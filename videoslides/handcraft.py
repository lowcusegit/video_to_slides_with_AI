from loadvideo.vloader_cv2np import cap_frame_change, weighted_change, video_info
from exportppt.toppt import recursive_list_file_paths, moving_mean_std, select_frames, create_powerpoint_with_select_frames
# import cv2
import os, time
import numpy as np
import argparse # Added for command-line arguments
import concurrent
from concurrent.futures import ThreadPoolExecutor # Added for multi-threading

def video_to_pptx_pipeline(video_file):
    print(video_file)
    export_dir = os.path.splitext(video_file)[0]
    output_pptx_name = export_dir + '.pptx'
    # start time
    t0=time.time()
    # video info
    v_info=video_info(video_file)
    length = v_info["frame_count"]
    fps = v_info["fps"]
    t1=time.time()
    print(f'  frame_count:{v_info["frame_count"]} FPS:{v_info["fps"]} WxH:({v_info["width"]}, {v_info["height"]})')
    print(f"  load video info time: {t1-t0}")
    
    # extracting
    frame_change_array = cap_frame_change(video_file, pool=(20,20))
    t2=time.time()
    # print(f"  frame_change_array shape: {frame_change_array.shape}")
    print(f"  extract video change time: {t2-t1}")

    # weighting
    weighted_change_array = weighted_change(frame_change_array, stable_frame=int(fps//2), threshold_pix=0.5, threshold_count=(int(fps//2)-1))
    weighted_changes = np.pad(weighted_change_array.astype(np.float32), (length-weighted_change_array.shape[0],0), 'constant', constant_values=0)
    mean,std = moving_mean_std(weighted_changes, d=int(fps*20))
    pad_size= int(length - mean.shape[0])
    padding=(pad_size-int(pad_size//2), int(pad_size//2))
    height =np.pad(mean, padding, 'constant', constant_values=0) + 2*np.pad(std, padding, 'constant', constant_values=0.0)
    height = np.clip(height, a_min=0.5, a_max=None)
    t3=time.time()
    # print(f"weighted_change shape: {weighted_changes.shape}")
    print(f"  weighting video change time: {t3-t2}")

    # selecting frames for slides
    selection_frames = select_frames(weighted_changes, height=height, distance=fps*4)
    t4=time.time()
    print(f"  # selection frame for slide: {len(selection_frames)}")
    print(f"  selecting video frames time: {t4-t3}")

    # frames to pptx file
    create_powerpoint_with_select_frames(video_file, selection_frames, output_pptx_name=output_pptx_name)
    t5=time.time()
    print(f"  exported pptx file: {output_pptx_name}")
    print(f"  preparing ppt time: {t5-t4}")
    print(f"total time: {t5-t0}")

def main():
    parser = argparse.ArgumentParser(description="Analyze videos and generate PowerPoint presentations from selected frames.")
    parser.add_argument("--video_folder", type=str, required=True,
                        help="Path to the folder containing video files.")
    parser.add_argument("--video_extension", type=str, nargs='+', default=['.mp4', '.MP4'],
                        help="List of video file extensions to include (e.g., .mp4 .avi).")
    parser.add_argument("--num_of_thread", type=int, default=2,
                        help="Number of threads to use for parallel video processing.")
    
    args = parser.parse_args()

    video_folder = args.video_folder
    # Normalize extensions to lowercase for consistent matching
    video_extension = [ext.lower() for ext in args.video_extension] 
    num_of_thread = args.num_of_thread

    print(f"Searching for videos in: {video_folder}")
    print(f"Including extensions: {video_extension}")
    print(f"Using {num_of_thread} threads.")

    video_list = recursive_list_file_paths(
        video_folder,
        extension=video_extension)

    if not video_list:
        print(f"No video files found with extensions {video_extension} in {video_folder}")
        return

    print(f"Found {len(video_list)} video files.")

    # Multi-threading the video_to_pptx_pipeline
    with ThreadPoolExecutor(max_workers=num_of_thread) as executor:
        # Submit each video processing task to the executor
        futures = {executor.submit(video_to_pptx_pipeline, v): v for v in video_list}
        
        # Wait for tasks to complete and handle results/exceptions
        for future in concurrent.futures.as_completed(futures):
            video_path = futures[future]
            try:
                future.result() # This will re-raise any exception that occurred in the thread
            except Exception as exc:
                print(f'Error processing {video_path}: {exc}')
            else:
                print(f'Finished processing {video_path}.')

if __name__ == '__main__':
    main()
