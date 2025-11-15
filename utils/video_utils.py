"""
Video processing utilities
"""
import cv2
import numpy as np
from collections import deque
from threading import Thread, Lock
import time
from utils.logger import get_logger

logger = get_logger(__name__)


class VideoCapture:
    """Threaded video capture for better performance"""
    
    def __init__(self, source=0, buffer_size=128):
        """
        Initialize video capture
        
        Args:
            source: Video source (0 for webcam, RTSP URL for IP camera)
            buffer_size: Frame buffer size
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source: {source}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Frame buffer
        self.buffer = deque(maxlen=buffer_size)
        self.lock = Lock()
        
        # Threading
        self.thread = None
        self.stopped = False
        self.grabbed = False
        self.frame = None
        
        logger.info(f"Video source initialized: {source} ({self.width}x{self.height} @ {self.fps}fps)")
    
    def start(self):
        """Start the threaded video capture"""
        if self.thread is None or not self.thread.is_alive():
            self.stopped = False
            self.thread = Thread(target=self._update, daemon=True)
            self.thread.start()
            logger.info("Video capture thread started")
        return self
    
    def _update(self):
        """Continuously read frames in background thread"""
        while not self.stopped:
            if not self.grabbed:
                self.grabbed, self.frame = self.cap.read()
                if self.grabbed:
                    with self.lock:
                        self.buffer.append(self.frame.copy())
            else:
                time.sleep(0.001)  # Prevent CPU overuse
    
    def read(self):
        """Read the latest frame"""
        with self.lock:
            if len(self.buffer) > 0:
                return True, self.buffer[-1]
        return False, None
    
    def get_buffer(self):
        """Get all frames in buffer"""
        with self.lock:
            return list(self.buffer)
    
    def stop(self):
        """Stop video capture"""
        self.stopped = True
        if self.thread is not None:
            self.thread.join()
        if self.cap is not None:
            self.cap.release()
        logger.info("Video capture stopped")
    
    def __del__(self):
        self.stop()


class VideoWriter:
    """Wrapper for video writing with buffer support"""
    
    def __init__(self, output_path, fps=30, frame_size=(640, 480), codec='mp4v'):
        """
        Initialize video writer
        
        Args:
            output_path: Output video file path
            fps: Frames per second
            frame_size: (width, height)
            codec: Video codec
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.codec = cv2.VideoWriter_fourcc(*codec)
        
        self.writer = cv2.VideoWriter(
            str(output_path),
            self.codec,
            fps,
            frame_size
        )
        
        if not self.writer.isOpened():
            raise ValueError(f"Unable to open video writer: {output_path}")
        
        logger.info(f"Video writer initialized: {output_path}")
    
    def write(self, frame):
        """Write a frame to video"""
        if frame.shape[:2][::-1] != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        self.writer.write(frame)
    
    def write_frames(self, frames):
        """Write multiple frames"""
        for frame in frames:
            self.write(frame)
    
    def release(self):
        """Release the video writer"""
        if self.writer is not None:
            self.writer.release()
        logger.info(f"Video saved: {self.output_path}")
    
    def __del__(self):
        self.release()


class FrameBuffer:
    """Circular frame buffer with timestamp tracking"""
    
    def __init__(self, max_seconds=15, fps=30):
        """
        Initialize frame buffer
        
        Args:
            max_seconds: Maximum seconds to store
            fps: Frames per second
        """
        self.max_size = max_seconds * fps
        self.buffer = deque(maxlen=self.max_size)
        self.timestamps = deque(maxlen=self.max_size)
        self.lock = Lock()
        self.fps = fps
        
        logger.debug(f"Frame buffer initialized: {max_seconds}s ({self.max_size} frames)")
    
    def add(self, frame, timestamp=None):
        """
        Add frame to buffer
        
        Args:
            frame: Frame to add
            timestamp: Optional timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            self.buffer.append(frame.copy())
            self.timestamps.append(timestamp)
    
    def get_recent(self, seconds):
        """
        Get frames from last N seconds
        
        Args:
            seconds: Number of seconds to retrieve
        
        Returns:
            list: List of (frame, timestamp) tuples
        """
        with self.lock:
            if len(self.buffer) == 0:
                return []
            
            current_time = self.timestamps[-1]
            cutoff_time = current_time - seconds
            
            result = []
            for frame, ts in zip(self.buffer, self.timestamps):
                if ts >= cutoff_time:
                    result.append((frame, ts))
            
            return result
    
    def get_range(self, start_time, end_time):
        """
        Get frames in time range
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
        
        Returns:
            list: List of (frame, timestamp) tuples
        """
        with self.lock:
            result = []
            for frame, ts in zip(self.buffer, self.timestamps):
                if start_time <= ts <= end_time:
                    result.append((frame, ts))
            return result
    
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()
            self.timestamps.clear()


def draw_bounding_box(frame, bbox, label, color=(0, 255, 0), thickness=2):
    """
    Draw bounding box with label
    
    Args:
        frame: Input frame
        bbox: Bounding box (x1, y1, x2, y2)
        label: Label text
        color: Box color (B, G, R)
        thickness: Line thickness
    
    Returns:
        frame: Frame with drawn box
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label background
    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y1_label = max(y1, label_size[1] + 10)
    cv2.rectangle(
        frame,
        (x1, y1_label - label_size[1] - 10),
        (x1 + label_size[0], y1_label + baseline - 10),
        color,
        -1
    )
    
    # Draw label text
    cv2.putText(
        frame,
        label,
        (x1, y1_label - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    
    return frame


def create_heatmap(frame_shape, points, radius=20):
    """
    Create a heatmap from points
    
    Args:
        frame_shape: Shape of frame (height, width)
        points: List of (x, y) coordinates
        radius: Heatmap radius
    
    Returns:
        heatmap: Heatmap image
    """
    heatmap = np.zeros(frame_shape[:2], dtype=np.float32)
    
    for x, y in points:
        x, y = int(x), int(y)
        if 0 <= x < frame_shape[1] and 0 <= y < frame_shape[0]:
            cv2.circle(heatmap, (x, y), radius, 1.0, -1)
    
    # Normalize and apply color map
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap_colored


def resize_with_aspect_ratio(frame, target_size):
    """
    Resize frame while maintaining aspect ratio
    
    Args:
        frame: Input frame
        target_size: Target size (width, height)
    
    Returns:
        resized: Resized frame
    """
    h, w = frame.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Create canvas and paste resized frame
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas