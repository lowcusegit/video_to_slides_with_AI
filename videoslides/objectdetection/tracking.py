import numpy as np
from scipy.optimize import linear_sum_assignment

# Reduce quadrilaterals or polygon to axis-parallel retangle, loss of of accuracy but gain in computation efficiency
def to_retangle(boxes):
    """
    Args:
    box : coordinates of polygon corners in 2D array [[x1,y1],[x2,y2],...]
    or
    boxes: 3D array [box1, box2,...]
    Return:
    box: 1D array [x_min, y_min, x_max, y_max]
    or
    boxes: 2D array [[x_min, y_min, x_max, y_max], ...]
    """
    dim = boxes.ndim
    if dim==2:
        min_c, max_c = np.min(boxes,  axis=0), np.max(boxes,  axis=0)
        return np.concatenate([min_c, max_c])
    if dim==3:
        min_c, max_c = np.min(boxes,  axis=1), np.max(boxes,  axis=1)
        return np.concatenate([min_c, max_c], axis=1)
    print("array dimension mismatch")
    return None

def calculate_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Boxes are expected in [x1, y1, x2, y2] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class SimpleIoUTracker:
    def __init__(self, iou_threshold=0.3):
        self.next_id = 0
        self.tracks = {}  # {id: last_bbox}
        self.history = {} # {id: {'start': frame_idx, 'end': frame_idx}}
        self.iou_threshold = iou_threshold

    def update(self, detections, frame_idx):
        """
        detections: list of [x1, y1, x2, y2]
        frame_idx: current frame number
        """
        if not self.tracks:
            # First frame or all tracks lost: initialize all detections as new tracks
            for det in detections:
                self.add_new_track(det, frame_idx)
            return

        track_ids = list(self.tracks.keys())
        track_boxes = list(self.tracks.values())

        # Create cost matrix based on IoU (we want to maximize IoU, so cost is 1 - IoU)
        iou_matrix = np.zeros((len(track_boxes), len(detections)), dtype=np.float32)
        for t, track_box in enumerate(track_boxes):
            for d, det_box in enumerate(detections):
                iou_matrix[t, d] = calculate_iou(track_box, det_box)

        # Optimal matching using Hungarian Algorithm
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched_indices = []
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched_indices.append((r, c))

        matched_track_indices = [m[0] for m in matched_indices]
        matched_det_indices = [m[1] for m in matched_indices]

        # 1. Update matched tracks
        for t_idx, d_idx in matched_indices:
            tid = track_ids[t_idx]
            self.tracks[tid] = detections[d_idx]
            self.history[tid]['end'] = frame_idx

        # 2. Create new tracks for unmatched detections
        for d_idx in range(len(detections)):
            if d_idx not in matched_det_indices:
                self.add_new_track(detections[d_idx], frame_idx)

        # 3. Clean up tracks not matched in this frame
        # In a real scenario, you might keep them for a few frames (max_age).
        # For this requirement, if they aren't in this frame, they are "lost".
        for t_idx in range(len(track_ids)):
            if t_idx not in matched_track_indices:
                del self.tracks[track_ids[t_idx]]

    def add_new_track(self, bbox, frame_idx):
        self.tracks[self.next_id] = bbox
        self.history[self.next_id] = {'start': frame_idx, 'end': frame_idx}
        self.next_id += 1

    def get_results(self):
        return self.history

class IoUTracker_with_Miss:
    def __init__(self, iou_threshold=0.9, miss_threshold=0):
        self.next_id = 0
        self.tracks = {}  # {id: last_bbox}
        self.history = {} # {id: {'start': [frame_idx], 'end': [frame_idx]}} # start/reappear, lastest-seen (miss/end)
        self.iou_threshold = iou_threshold
        self.miss_threshold = miss_threshold

    def update(self, detections, frame_idx):
        """
        detections: list/array of [x1, y1, x2, y2]
        frame_idx: current frame number
        """
        if not self.tracks:
            # First frame or all tracks lost: initialize all detections as new tracks
            for det in detections:
                self.add_new_track(det, frame_idx)
            return

        track_ids = list(self.tracks.keys())
        track_boxes = list(self.tracks.values())

        # Create cost matrix based on IoU (we want to maximize IoU, so cost is 1 - IoU)
        iou_matrix = np.zeros((len(track_boxes), len(detections)), dtype=np.float32)
        for t, track_box in enumerate(track_boxes):
            for d, det_box in enumerate(detections):
                iou_matrix[t, d] = calculate_iou(track_box, det_box)

        # Optimal matching using Hungarian Algorithm
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched_indices = []
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched_indices.append((r, c))

        matched_track_indices = [m[0] for m in matched_indices]
        matched_det_indices = [m[1] for m in matched_indices]

        # 1. Update matched tracks
        for t_idx, d_idx in matched_indices:
            tid = track_ids[t_idx]
            self.tracks[tid] = detections[d_idx]
            if frame_idx-self.history[tid]['end'][-1] > 1: # reappear
                self.history[tid]['start'].append( frame_idx )
                self.history[tid]['end'].append( frame_idx )
            else:
                self.history[tid]['end'][-1]= frame_idx

        # 2. Create new tracks for unmatched detections
        for d_idx in range(len(detections)):
            if d_idx not in matched_det_indices:
                self.add_new_track(detections[d_idx], frame_idx)

        # 3. Clean up tracks not matched for a few frames (miss_threshold).
        for t_idx in range(len(track_ids)):
            if (t_idx not in matched_track_indices) and (frame_idx-self.history[track_ids[t_idx]]['end'][-1]>self.miss_threshold):
                del self.tracks[track_ids[t_idx]]

    def add_new_track(self, bbox, frame_idx):
        self.tracks[self.next_id] = bbox
        self.history[self.next_id] = {'start': [frame_idx], 'end': [frame_idx]}
        self.next_id += 1

    def get_results(self):
        return self.history

