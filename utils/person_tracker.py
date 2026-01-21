"""
Stable person tracking with smoothing and temporal filtering
"""
import numpy as np
from collections import deque
import time
from utils.logger import get_logger

logger = get_logger(__name__)


class PersonTracker:
    """
    Stable person tracker with:
    - Temporal smoothing of bounding boxes
    - Better ID assignment logic
    - Ghost detection removal
    """
    
    def __init__(self, max_disappeared=30, distance_threshold=150, smoothing_window=5):
        """
        Initialize person tracker
        
        Args:
            max_disappeared: Frames before considering person as left
            distance_threshold: Max distance for same person (pixels)
            smoothing_window: Number of frames to smooth bounding boxes
        """
        self.next_id = 1
        self.tracked_objects = {}  # id -> TrackedPerson
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold
        self.smoothing_window = smoothing_window
        
        logger.info(f"Person tracker initialized (threshold={distance_threshold}px)")
    
    def update(self, detections, timestamp=None):
        """
        Update tracker with new detections
        
        Args:
            detections: List of bounding boxes [x1, y1, x2, y2]
            timestamp: Current timestamp
        
        Returns:
            dict: person_id -> smoothed_bbox
        """
        if timestamp is None:
            timestamp = time.time()
        
        # If no detections, increment disappeared counter
        if len(detections) == 0:
            for person_id in list(self.tracked_objects.keys()):
                self.tracked_objects[person_id].disappeared += 1
                
                # Remove persons that disappeared for too long
                if self.tracked_objects[person_id].disappeared > self.max_disappeared:
                    logger.info(f"Person {person_id} removed (disappeared)")
                    del self.tracked_objects[person_id]
            
            return self.get_current_tracked()
        
        # If no tracked objects, register all as new
        if len(self.tracked_objects) == 0:
            for bbox in detections:
                self._register(bbox, timestamp)
            return self.get_current_tracked()
        
        # Calculate cost matrix (distance between all tracked and detected)
        tracked_ids = list(self.tracked_objects.keys())
        tracked_centers = [
            self.tracked_objects[pid].get_center() 
            for pid in tracked_ids
        ]
        
        detected_centers = [self._bbox_center(bbox) for bbox in detections]
        
        # Compute distances
        cost_matrix = np.zeros((len(tracked_ids), len(detections)))
        for i, tc in enumerate(tracked_centers):
            for j, dc in enumerate(detected_centers):
                cost_matrix[i, j] = self._distance(tc, dc)
        
        # Assign detections to tracked objects
        assignments = self._hungarian_assignment(cost_matrix)
        
        # Update tracked objects
        used_detections = set()
        
        for tracked_idx, detected_idx in assignments:
            person_id = tracked_ids[tracked_idx]
            
            # Only assign if distance is reasonable
            if cost_matrix[tracked_idx, detected_idx] < self.distance_threshold:
                self.tracked_objects[person_id].update(detections[detected_idx], timestamp)
                used_detections.add(detected_idx)
            else:
                # Too far, mark as disappeared
                self.tracked_objects[person_id].disappeared += 1
        
        # Register new detections
        for i, bbox in enumerate(detections):
            if i not in used_detections:
                self._register(bbox, timestamp)
        
        # Remove disappeared persons
        for person_id in list(self.tracked_objects.keys()):
            if self.tracked_objects[person_id].disappeared > self.max_disappeared:
                logger.info(f"Person {person_id} removed (disappeared)")
                del self.tracked_objects[person_id]
        
        return self.get_current_tracked()
    
    def _register(self, bbox, timestamp):
        """Register a new person"""
        person_id = self.next_id
        self.next_id += 1
        self.tracked_objects[person_id] = TrackedPerson(
            person_id, bbox, timestamp, self.smoothing_window
        )
        logger.info(f"Person {person_id} registered")
    
    def _bbox_center(self, bbox):
        """Calculate center of bounding box"""
        return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    
    def _distance(self, point1, point2):
        """Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _hungarian_assignment(self, cost_matrix):
        """
        Simple greedy assignment (faster than scipy's Hungarian)
        For each tracked object, assign closest detection
        """
        assignments = []
        used_cols = set()
        
        # Sort by minimum cost
        rows, cols = cost_matrix.shape
        
        for _ in range(min(rows, cols)):
            # Find minimum value not yet assigned
            min_val = float('inf')
            min_row, min_col = -1, -1
            
            for i in range(rows):
                for j in range(cols):
                    if j not in used_cols and cost_matrix[i, j] < min_val:
                        min_val = cost_matrix[i, j]
                        min_row, min_col = i, j
            
            if min_row != -1:
                assignments.append((min_row, min_col))
                used_cols.add(min_col)
        
        return assignments
    
    def get_current_tracked(self):
        """Get all currently tracked persons with smoothed bboxes"""
        result = {}
        for person_id, tracked_person in self.tracked_objects.items():
            if tracked_person.disappeared == 0:  # Only return visible persons
                result[person_id] = tracked_person.get_smoothed_bbox()
        return result
    
    def get_person_info(self, person_id):
        """Get detailed info about a person"""
        if person_id in self.tracked_objects:
            return {
                'id': person_id,
                'bbox': self.tracked_objects[person_id].get_smoothed_bbox(),
                'center': self.tracked_objects[person_id].get_center(),
                'age': self.tracked_objects[person_id].age,
                'disappeared': self.tracked_objects[person_id].disappeared
            }
        return None
    
    def clear(self):
        """Clear all tracked persons"""
        self.tracked_objects.clear()
        self.next_id = 1
        logger.info("Tracker cleared")


class TrackedPerson:
    """Individual tracked person with temporal smoothing"""
    
    def __init__(self, person_id, bbox, timestamp, smoothing_window=5):
        """
        Initialize tracked person
        
        Args:
            person_id: Unique ID
            bbox: Initial bounding box [x1, y1, x2, y2]
            timestamp: Initial timestamp
            smoothing_window: Number of frames for smoothing
        """
        self.person_id = person_id
        self.bbox_history = deque(maxlen=smoothing_window)
        self.bbox_history.append(bbox)
        self.timestamp = timestamp
        self.disappeared = 0
        self.age = 0  # Number of frames tracked
    
    def update(self, bbox, timestamp):
        """Update with new detection"""
        self.bbox_history.append(bbox)
        self.timestamp = timestamp
        self.disappeared = 0
        self.age += 1
    
    def get_smoothed_bbox(self):
        """
        Get temporally smoothed bounding box
        Uses exponential moving average for stability
        """
        if len(self.bbox_history) == 0:
            return None
        
        # Weighted average with more weight on recent frames
        weights = np.exp(np.linspace(-2, 0, len(self.bbox_history)))
        weights /= weights.sum()
        
        smoothed = np.zeros(4)
        for i, bbox in enumerate(self.bbox_history):
            smoothed += np.array(bbox) * weights[i]
        
        return smoothed.tolist()
    
    def get_center(self):
        """Get center of current bbox"""
        bbox = self.get_smoothed_bbox()
        if bbox is None:
            return [0, 0]
        return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]