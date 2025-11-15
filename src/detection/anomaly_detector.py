"""
Anomaly detection using baseline comparison and background subtraction
"""
import cv2
import numpy as np
from pathlib import Path
from utils.logger import get_logger
from config.settings import ANOMALY_SETTINGS, BASELINE_DIR

logger = get_logger(__name__)


class AnomalyDetector:
    """Detect anomalies on desks using baseline frame comparison"""
    
    def __init__(self, baseline_path=None):
        """
        Initialize anomaly detector
        
        Args:
            baseline_path: Path to save/load baseline frame
        """
        self.baseline_frame = None
        self.baseline_path = baseline_path or BASELINE_DIR / "baseline.jpg"
        self.baseline_gray = None
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=ANOMALY_SETTINGS['background_subtractor_history'],
            varThreshold=ANOMALY_SETTINGS['background_subtractor_threshold'],
            detectShadows=True
        )
        
        # Detection parameters
        self.diff_threshold = ANOMALY_SETTINGS['difference_threshold']
        self.min_contour_area = ANOMALY_SETTINGS['min_contour_area']
        
        # Load baseline if exists
        self._load_baseline()
        
        logger.info("Anomaly detector initialized")
    
    def set_baseline(self, frame):
        """
        Set baseline reference frame
        
        Args:
            frame: Reference frame (clean desk)
        """
        self.baseline_frame = frame.copy()
        self.baseline_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        self.baseline_gray = cv2.GaussianBlur(self.baseline_gray, (5, 5), 0)
        
        # Save baseline
        self._save_baseline()
        
        logger.info("Baseline frame set")
    
    def detect_anomalies(self, frame, method='difference'):
        """
        Detect anomalies in current frame
        
        Args:
            frame: Current frame
            method: Detection method ('difference', 'background_subtraction', 'both')
        
        Returns:
            dict: Anomaly detection results
        """
        if self.baseline_frame is None:
            logger.warning("No baseline set, cannot detect anomalies")
            return {'anomalies': [], 'anomaly_mask': None}
        
        anomalies = []
        
        if method in ['difference', 'both']:
            diff_anomalies = self._detect_by_difference(frame)
            anomalies.extend(diff_anomalies)
        
        if method in ['background_subtraction', 'both']:
            bg_anomalies = self._detect_by_background_subtraction(frame)
            anomalies.extend(bg_anomalies)
        
        # Merge overlapping anomalies
        anomalies = self._merge_overlapping_anomalies(anomalies)
        
        # Create anomaly mask
        anomaly_mask = self._create_anomaly_mask(frame.shape, anomalies)
        
        return {
            'anomalies': anomalies,
            'anomaly_mask': anomaly_mask,
            'count': len(anomalies)
        }
    
    def _detect_by_difference(self, frame):
        """
        Detect anomalies by comparing with baseline
        
        Args:
            frame: Current frame
        
        Returns:
            list: List of anomaly regions
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate difference
        diff = cv2.absdiff(self.baseline_gray, gray)
        
        # Threshold
        _, thresh = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        anomalies = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                anomalies.append({
                    'bbox': [x, y, x+w, y+h],
                    'area': area,
                    'type': 'difference',
                    'confidence': min(area / 10000, 1.0)  # Normalized confidence
                })
        
        return anomalies
    
    def _detect_by_background_subtraction(self, frame):
        """
        Detect anomalies using background subtraction
        
        Args:
            frame: Current frame
        
        Returns:
            list: List of anomaly regions
        """
        # Apply background subtractor
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        anomalies = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                anomalies.append({
                    'bbox': [x, y, x+w, y+h],
                    'area': area,
                    'type': 'background_subtraction',
                    'confidence': min(area / 10000, 1.0)
                })
        
        return anomalies
    
    def _merge_overlapping_anomalies(self, anomalies):
        """
        Merge overlapping anomaly regions
        
        Args:
            anomalies: List of anomalies
        
        Returns:
            list: Merged anomalies
        """
        if len(anomalies) < 2:
            return anomalies
        
        merged = []
        used = [False] * len(anomalies)
        
        for i, anom1 in enumerate(anomalies):
            if used[i]:
                continue
            
            bbox1 = anom1['bbox']
            merged_bbox = bbox1[:]
            merged_area = anom1['area']
            
            for j, anom2 in enumerate(anomalies[i+1:], start=i+1):
                if used[j]:
                    continue
                
                bbox2 = anom2['bbox']
                
                # Check overlap
                if self._boxes_overlap(bbox1, bbox2):
                    # Merge boxes
                    merged_bbox = [
                        min(merged_bbox[0], bbox2[0]),
                        min(merged_bbox[1], bbox2[1]),
                        max(merged_bbox[2], bbox2[2]),
                        max(merged_bbox[3], bbox2[3])
                    ]
                    merged_area += anom2['area']
                    used[j] = True
            
            merged.append({
                'bbox': merged_bbox,
                'area': merged_area,
                'type': 'merged',
                'confidence': min(merged_area / 10000, 1.0)
            })
            used[i] = True
        
        return merged
    
    def _boxes_overlap(self, box1, box2, threshold=0.3):
        """
        Check if two boxes overlap
        
        Args:
            box1, box2: Bounding boxes [x1, y1, x2, y2]
            threshold: IoU threshold
        
        Returns:
            bool: True if boxes overlap
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        
        return iou > threshold
    
    def _create_anomaly_mask(self, frame_shape, anomalies):
        """
        Create a mask showing all anomalies
        
        Args:
            frame_shape: Shape of frame
            anomalies: List of anomalies
        
        Returns:
            mask: Binary mask
        """
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        
        for anom in anomalies:
            x1, y1, x2, y2 = map(int, anom['bbox'])
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        return mask
    
    def draw_anomalies(self, frame, anomalies):
        """
        Draw anomaly bounding boxes on frame
        
        Args:
            frame: Input frame
            anomalies: List of anomalies
        
        Returns:
            frame: Frame with drawn anomalies
        """
        result = frame.copy()
        
        for anom in anomalies:
            x1, y1, x2, y2 = map(int, anom['bbox'])
            
            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw label
            label = f"Anomaly: {anom['confidence']:.2f}"
            cv2.putText(
                result,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )
        
        return result
    
    def _save_baseline(self):
        """Save baseline frame to disk"""
        if self.baseline_frame is not None:
            Path(self.baseline_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.baseline_path), self.baseline_frame)
            logger.info(f"Baseline saved: {self.baseline_path}")
    
    def _load_baseline(self):
        """Load baseline frame from disk"""
        if Path(self.baseline_path).exists():
            self.baseline_frame = cv2.imread(str(self.baseline_path))
            if self.baseline_frame is not None:
                self.baseline_gray = cv2.cvtColor(self.baseline_frame, cv2.COLOR_BGR2GRAY)
                self.baseline_gray = cv2.GaussianBlur(self.baseline_gray, (5, 5), 0)
                logger.info(f"Baseline loaded: {self.baseline_path}")
    
    def reset(self):
        """Reset the detector"""
        self.baseline_frame = None
        self.baseline_gray = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=ANOMALY_SETTINGS['background_subtractor_history'],
            varThreshold=ANOMALY_SETTINGS['background_subtractor_threshold'],
            detectShadows=True
        )
        logger.info("Anomaly detector reset")