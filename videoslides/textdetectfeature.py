from loadvideo.vloader_cv2np import video_info, text_detection_db #, frame_producer, weighted_change
from exportppt.toppt import recursive_list_file_paths, create_powerpoint_with_select_frames #, moving_mean_std, select_frames
from objectdetection.tracking import IoUTracker_with_Miss, to_retangle
import os, time
# import cv2
import numpy as np

import argparse # Added for command-line arguments
import concurrent
from concurrent.futures import ProcessPoolExecutor # Changed from ThreadPoolExecutor to ProcessPoolExecutor for better performance with CPU-bound tasks

def tracking_method(detection_boxes, iou_threshold=0.66, miss_threshold=5):
    """ detection_boxes: list of np.array of shape (num_boxes, 4) for each frame, in (x1, y1, x2, y2) format"""
    IoUTracker_with_Miss_instance = IoUTracker_with_Miss(iou_threshold=iou_threshold, miss_threshold=miss_threshold) # Assuming 30 FPS and 0.6 second tolerance
    for idx, boxes in enumerate(detection_boxes):
        IoUTracker_with_Miss_instance.update(boxes, idx)
    tracking_history = IoUTracker_with_Miss_instance.get_results()
    return tracking_history

def select_frames_method(tracking_history, stability_threshold=10, before=False, after=None):
    """Select frames based on the tracking history of detected objects, focusing on stable appearances.
    Args:
        tracking_history: dict of {id: {'start': [frame_idx], 'end': [frame_idx]}} for each tracked object, where 'start' and 'end' are lists of frame indices.
        stability_threshold (int, optional): Minimum number of consecutive frames an object must be detected to be considered stable.
        before (bool, optional): Whether to include a frame before the first detection.
        after (None or int, optional): when given int of total frames, include a frame after the last detection.
    Returns:
        List of selected frame indices.
    """
    unchecked_objects = list(tracking_history.keys())
    # outstanding_objects = []
    selection_frames =  []
    if before:
        first_detected_frames = tracking_history[unchecked_objects[0]]['start'][0]
        if first_detected_frames > 0:
            selection_frames.append(first_detected_frames//2)
    candidate_frames = set()
    end_stability = stability_threshold//2
    begin_stability = stability_threshold - end_stability

    for obj_id in unchecked_objects:
        curr_start, curr_end = tracking_history[obj_id]['start'], tracking_history[obj_id]['end']
        # evaluate the stability of the object (appearing continuously for at least the stability_threshold)
        stability_set = set()
        for s, e in zip (curr_start, curr_end):
            stability_length = e - s
            if stability_length >= stability_threshold:
                # if the object is stable for at least the threshold, consider it as a candidate for selection
                stability_set.update(range(s+begin_stability, e-end_stability+1))
        if len(stability_set) > 0:
            if len(candidate_frames) == 0:
                # no candidate (at beginning), then add the stability set to it
                candidate_frames = stability_set
            else:
                # if candidate frames exist, try to find if any covering the new stable object, 
                intersection_frames = candidate_frames.intersection(stability_set) # select frames where all objects are stable
                if len(intersection_frames) > 0:
                    # if yes, the new stable object can be captured in the intersection frames,
                    # then update the candidate frames to the finer intersection
                    candidate_frames = intersection_frames
                else:
                    # if no intersection, select the frames with most earlier objects stable before they are missing
                    # new stable object will be captured in the candidate frames next time
                    selection_frames.append(max(candidate_frames))
                    candidate_frames = stability_set
    # the remained candidate frames covering last stable object not selected in above loop, so select last frame
    selection_frames.append(max(candidate_frames))

    if after is not None:
        remaining_frames = after - 1 - max(candidate_frames)
        if remaining_frames > 0:
            selection_frames.append(after - 1 -( remaining_frames//2))

    return selection_frames

def video_to_pptx_pipeline(video_file, model_path):
     # start time
    t0=time.time()
    print(video_file)
    export_dir = os.path.splitext(video_file)[0]
    output_pptx_name = export_dir + '.pptx'
    # video info
    v_info=video_info(video_file)
    length = v_info["frame_count"]
    fps = v_info["fps"]
    t1=time.time()
    print(f'  frame_count:{v_info["frame_count"]} FPS:{v_info["fps"]} WxH:({v_info["width"]}, {v_info["height"]})')
    print(f"  load video info time: {t1-t0}")
    
    # extracting
    next_frame= max(round(fps/5), 1) # skip frames for around 1/5 second, e.g. 6 frames for 30fps video
    new_fps = fps/next_frame
    # frame_features: list of (boxes, confidence) for each processed frame
    frame_features = text_detection_db(video_file, model_path, pool=None, v_info=v_info, next_frame=next_frame)
    processing_frame_count = len(frame_features)
    t2=time.time()
    print(f"  processed: {processing_frame_count} frames")
    print(f"  extract video features time: {t2-t1}")

    # feature processing
    detection_boxes = [to_retangle(np.stack(f[0], axis=0)) if (len(f[0]) > 0) else np.array([]) for f in frame_features ] # convert to (x1, y1, x2, y2) format for tracking
    # some Multi Object Tracking method to process the boxes from frame_features and determine when objects appear and disappear
    # miss_threshold flag of round(new_fps*0.6) for 0.6 second tolerance, or 0 for stop tracking when missing in the next frame
    
    # Tracking_history: {id: {'start': [frame_idx], 'end': [frame_idx]}}, id in the order of appearing
    iou_threshold=0.66
    miss_threshold=round(new_fps*1)
    tracking_history = tracking_method(detection_boxes, iou_threshold=iou_threshold, miss_threshold=miss_threshold)
        
    t3=time.time()
    # print(f"weighted_change shape: {weighted_changes.shape}")
    print(f"  # tracked objects: {len(tracking_history)}")
    print(f"  tracking time: {t3-t2}")

    # selecting frames for slides
    
    # By the playback of the tracking history, identify stable objects
    # stability_threshold = round(1*new_fps)  # e.g. (1*new_fps) for 1 second
    stability_threshold = round(1*new_fps)
    # stability_history = {} # {id: set(frame_idx where the object is stable)}
    selection_frames = select_frames_method(
        tracking_history,
        stability_threshold=stability_threshold,
        before=False,
        after=processing_frame_count
        )

    if next_frame > 1:
        selection_frames = [f*next_frame for f in selection_frames] # convert back to original frame indices    
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
    parser.add_argument("--model_path", type=str, default="./configmodel/DB_TD500_resnet18.onnx",
                        help="Path to the text detection model (e.g., DB18 ONNX file).")
    
    args = parser.parse_args()

    model_path = args.model_path
    if not os.path.isfile(model_path):
        print(f"Model file not found: {model_path}")
        return

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

    # Use ProcessPoolExecutor for CPU-heavy video analysis
    # Note: Ensure video_to_pptx_pipeline is top-level for pickling
    # cap_frame_change called in the video_to_pptx_pipeline will have its own internal Producer thread, so
    # Process Level and Thread Level parallelism can be combined for better performance,
    # but here we focus on ProcessPoolExecutor for the main video processing tasks.
    with ProcessPoolExecutor(max_workers=num_of_thread) as executor:
        # Submit each video processing task to the executor
        futures = {executor.submit(video_to_pptx_pipeline, v, model_path): v for v in video_list}
        
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
