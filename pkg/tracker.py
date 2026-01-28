import numpy as np


class ReIDTracker:
    """Simple tracker using spatial proximity (Euclidean distance)"""

    def __init__(self, max_distance=100, max_frames_to_skip=30):
        self.tracks = {}  # track_id: {'center': ..., 'box': ..., 'frames_skipped': ...}
        self.next_track_id = 1
        self.max_distance = max_distance
        self.max_frames_to_skip = max_frames_to_skip

    def get_box_center(self, box):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return (cx, cy)

    def compute_distance(self, center1, center2):
        """Compute Euclidean distance between two centers"""
        if center1 is None or center2 is None:
            return float("inf")
        cx1, cy1 = center1
        cx2, cy2 = center2
        return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    def update(self, img, boxes):
        """Update tracks with new detections"""
        # Get centers for all detections
        detections = []
        for box in boxes:
            center = self.get_box_center(box)
            detections.append({"box": box, "center": center})

        for track_id in self.tracks:
            self.tracks[track_id]["frames_skipped"] += 1

        matched_tracks = set()
        matched_detections = set()
        assignments = []

        for det_idx, det in enumerate(detections):
            for track_id, track in self.tracks.items():
                dist = self.compute_distance(det["center"], track["center"])
                assignments.append((dist, track_id, det_idx))

        # Sort by distance and assign greedily
        assignments.sort(key=lambda x: x[0])

        for dist, track_id, det_idx in assignments:
            if track_id in matched_tracks or det_idx in matched_detections:
                continue
            if dist < self.max_distance:
                self.tracks[track_id]["center"] = detections[det_idx]["center"]
                self.tracks[track_id]["box"] = detections[det_idx]["box"]
                self.tracks[track_id]["frames_skipped"] = 0
                matched_tracks.add(track_id)
                matched_detections.add(det_idx)

        for det_idx, det in enumerate(detections):
            if det_idx not in matched_detections:
                self.tracks[self.next_track_id] = {
                    "center": det["center"],
                    "box": det["box"],
                    "frames_skipped": 0,
                }
                self.next_track_id += 1

        # Remove tracks that haven't been seen for too long
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track["frames_skipped"] > self.max_frames_to_skip:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        return self.tracks
