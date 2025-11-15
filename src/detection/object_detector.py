"""
YOLO-based object detection for exam malpractice items
"""
import cv2
import numpy as np
from ultralytics import YOLO
from utils.logger import get_logger
from config.settings import YOLO_SETTINGS

logger = get_logger(__name__)


class ObjectDetector:
    """YOLO-based object detector for prohibited items"""
    
    def __init__(self, model_path=None, conf_threshold=0.5, device='cpu'):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold
            device: 'cpu' or 'cuda'
        """
        self.model_path = model_path or YOLO_SETTINGS['model_path']
        self.conf_threshold = conf_threshold or YOLO_SETTINGS['confidence_threshold']
        self.device = device or YOLO_SETTINGS['device']
        
        # Load YOLO model
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            logger.info(f"YOLO model loaded: {self.model_path} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
        
        # Target classes for detection
        self.target_classes = YOLO_SETTINGS['target_classes']
        
        # COCO class names
        self.class_names = {
            0: 'person',
            67: 'cell phone',
            73: 'book',
            # Add more as needed
        }
        
        # Detection statistics
        self.detection_count = {class_name: 0 for class_name in self.class_names.values()}
    
    def detect(self, frame, classes=None):
        """
        Detect objects in frame
        
        Args:
            frame: Input frame
            classes: List of class IDs to detect (None = detect all)
        
        Returns:
            dict: Detection results
        """
        if classes is None:
            classes = list(self.target_classes.keys())
        
        # Run inference
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                # Filter by target classes
                if cls in classes:
                    class_name = self.class_names.get(cls, f'class_{cls}')
                    
                    detection = {
                        'class_id': cls,
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox': xyxy.tolist(),  # [x1, y1, x2, y2]
                        'center': self._get_bbox_center(xyxy),
                    }
                    
                    detections.append(detection)
                    self.detection_count[class_name] = self.detection_count.get(class_name, 0) + 1
        
        return {
            'detections': detections,
            'count': len(detections),
            'frame_shape': frame.shape
        }
    
    def detect_prohibited_items(self, frame):
        """
        Detect prohibited items (phones, books, papers)
        
        Args:
            frame: Input frame
        
        Returns:
            dict: Detected prohibited items
        """
        # Classes for prohibited items
        prohibited_classes = [67, 73]  # cell phone, book
        
        results = self.detect(frame, classes=prohibited_classes)
        
        # Categorize detections
        prohibited_items = {
            'phones': [],
            'books': [],
            'papers': [],
            'total': 0
        }
        
        for det in results['detections']:
            if det['class_id'] == 67:  # cell phone
                prohibited_items['phones'].append(det)
            elif det['class_id'] == 73:  # book
                prohibited_items['books'].append(det)
            
            prohibited_items['total'] += 1
        
        return prohibited_items
    
    def detect_persons(self, frame):
        """
        Detect persons in frame
        
        Args:
            frame: Input frame
        
        Returns:
            list: List of person detections
        """
        results = self.detect(frame, classes=[0])  # person class
        return results['detections']
    
    def is_object_in_zone(self, bbox, zone):
        """
        Check if object center is within a zone
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            zone: Zone coordinates (x1, y1, x2, y2) normalized
        
        Returns:
            bool: True if object is in zone
        """
        center = self._get_bbox_center(bbox)
        
        # Assuming zone is normalized (0-1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        frame_h, frame_w = 720, 1280  # Default, should be passed as parameter
        
        zone_pixel = [
            zone[0] * frame_w,
            zone[1] * frame_h,
            zone[2] * frame_w,
            zone[3] * frame_h
        ]
        
        return (zone_pixel[0] <= center[0] <= zone_pixel[2] and
                zone_pixel[1] <= center[1] <= zone_pixel[3])
    
    def _get_bbox_center(self, bbox):
        """Calculate center of bounding box"""
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]
    
    def draw_detections(self, frame, detections, show_conf=True):
        """
        Draw detection bounding boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            show_conf: Show confidence scores
        
        Returns:
            frame: Frame with drawn detections
        """
        result = frame.copy()
        
        colors = {
            'person': (0, 255, 0),      # Green
            'cell phone': (0, 0, 255),  # Red
            'book': (255, 0, 0),        # Blue
            'paper': (255, 255, 0),     # Cyan
        }
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            class_name = det['class_name']
            conf = det['confidence']
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}"
            if show_conf:
                label += f" {conf:.2f}"
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                result,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            cv2.putText(
                result,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return result
    
    def get_statistics(self):
        """Get detection statistics"""
        return self.detection_count.copy()
    
    def reset_statistics(self):
        """Reset detection counters"""
        for key in self.detection_count:
            self.detection_count[key] = 0


# Helper functions for object tracking
def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        bbox1, bbox2: Bounding boxes [x1, y1, x2, y2]
    
    Returns:
        float: IoU value
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def filter_overlapping_detections(detections, iou_threshold=0.5):
    """
    Remove overlapping detections using Non-Maximum Suppression
    
    Args:
        detections: List of detections
        iou_threshold: IoU threshold for overlap
    
    Returns:
        list: Filtered detections
    """
    if len(detections) == 0:
        return []
    
    # Sort by confidence
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    
    while len(sorted_dets) > 0:
        current = sorted_dets.pop(0)
        keep.append(current)
        
        # Remove overlapping detections
        sorted_dets = [
            det for det in sorted_dets
            if calculate_iou(current['bbox'], det['bbox']) < iou_threshold
        ]
    
    return keep