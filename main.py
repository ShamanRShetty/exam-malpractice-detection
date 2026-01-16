"""
Main Application - Exam Malpractice Detection System
Complete integration of all modules
"""
import cv2
import time
import threading
from pathlib import Path
import numpy as np

# Import all modules
from config.settings import *
from utils.logger import get_logger
from utils.video_utils import VideoCapture, FrameBuffer
from utils.privacy import FaceBlurrer
from src.detection.object_detector import ObjectDetector
from src.detection.pose_detector import PoseDetector, MultiPersonPoseDetector
from src.detection.anomaly_detector import AnomalyDetector
from src.analysis.behavior_analyzer import BehaviorAnalyzer
from src.analysis.suspicion_scorer import SuspicionScorer

logger = get_logger(__name__, str(LOGS_DIR / 'main.log'))


class ExamMalpracticeDetector:
    """Main application class"""
    
    def __init__(self, camera_source=0):
        """
        Initialize the detection system
        
        Args:
            camera_source: Camera source (0 for webcam, RTSP URL for IP camera)
        """
        logger.info("="*60)
        logger.info("Initializing Exam Malpractice Detection System")
        logger.info("="*60)
        
        # Video capture
        self.camera_source = camera_source
        self.video_capture = VideoCapture(camera_source, buffer_size=CAMERA_SETTINGS['buffer_size'])
        self.frame_buffer = FrameBuffer(
            max_seconds=BEHAVIOR_SETTINGS['timeline_buffer_seconds'],
            fps=self.video_capture.fps
        )
        
        # Detection modules
        logger.info("Loading detection modules...")
        self.object_detector = ObjectDetector()
        self.pose_detector = MultiPersonPoseDetector()
        self.anomaly_detector = AnomalyDetector()
        
        # Analysis modules
        logger.info("Loading analysis modules...")
        self.behavior_analyzer = BehaviorAnalyzer()
        self.suspicion_scorer = SuspicionScorer()
        
        # Privacy
        if PRIVACY_SETTINGS['enable_face_blur']:
            self.face_blurrer = FaceBlurrer()
        else:
            self.face_blurrer = None
        
        # State
        self.running = False
        self.baseline_set = False
        self.frame_count = 0
        self.start_time = None
        
        # Person tracking (simple version)
        self.tracked_persons = {}
        self.next_person_id = 1
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_violations': 0,
            'fps': 0
        }
        
        logger.info("System initialized successfully!")
    
    def start(self):
        """Start the detection system"""
        logger.info("Starting detection system...")
        self.running = True
        self.start_time = time.time()
        
        # Start video capture
        self.video_capture.start()
        
        # Create window with proper flags
        cv2.namedWindow('Exam Malpractice Detection System', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Exam Malpractice Detection System', 1280, 720)
        
        # Wait for camera to initialize
        time.sleep(1)
        
        # Main processing loop
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
    
    def _main_loop(self):
        """Main processing loop"""
        logger.info("Entering main processing loop...")
        
        fps_start_time = time.time()
        fps_frame_count = 0
        
        while self.running:
            loop_start = time.time()
            
            # Read frame
            ret, frame = self.video_capture.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame")
                time.sleep(0.1)
                continue
            
            self.frame_count += 1
            fps_frame_count += 1
            
            # Add to buffer
            self.frame_buffer.add(frame, loop_start)
            
            # Set baseline on first valid frame
            if not self.baseline_set:
                self._setup_baseline(frame)
            
            # Skip frames for performance (process every Nth frame)
            if self.frame_count % PERFORMANCE_SETTINGS['frame_skip'] != 0:
                # Still display the frame even if we don't process it
                display_frame = frame.copy()
                self._draw_info_panel(display_frame)
                cv2.imshow('Exam Malpractice Detection System', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('b'):
                    self._setup_baseline(frame)
                elif key == ord('r'):
                    self._reset_system()
                elif key == ord('s'):
                    self._save_snapshot(display_frame)
                continue
            
            # Process frame
            results = self._process_frame(frame, loop_start)
            
            # Draw visualization
            display_frame = self._draw_visualization(frame, results)
            
            # Apply privacy if enabled
            if self.face_blurrer is not None:
                display_frame = self.face_blurrer.blur_faces(display_frame)
            
            # Display frame
            cv2.imshow('Exam Malpractice Detection System', display_frame)
            
            # Calculate FPS
            if time.time() - fps_start_time >= 1.0:
                self.stats['fps'] = fps_frame_count
                fps_frame_count = 0
                fps_start_time = time.time()
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit requested")
                break
            elif key == ord('b'):
                self._setup_baseline(frame)
            elif key == ord('r'):
                self._reset_system()
            elif key == ord('s'):
                self._save_snapshot(display_frame)
            
            # Print stats periodically
            if self.frame_count % 100 == 0:
                self._print_stats()
        
        logger.info("Main loop ended")
    
    def _process_frame(self, frame, timestamp):
        """
        Process single frame
        
        Args:
            frame: Input frame
            timestamp: Frame timestamp
        
        Returns:
            dict: Processing results
        """
        results = {
            'persons': [],
            'prohibited_items': {'phones': [], 'books': [], 'papers': [], 'total': 0},
            'anomalies': [],
            'high_risk_persons': []
        }
        
        # Detect persons
        person_detections = self.object_detector.detect_persons(frame)
        
        # Detect poses for each person
        person_bboxes = [det['bbox'] for det in person_detections]
        poses = self.pose_detector.detect(frame, person_bboxes)
        
        # Match poses with persons (simple 1:1 matching)
        for i, person_det in enumerate(person_detections):
            person_id = self._get_or_assign_person_id(person_det['bbox'])
            
            pose_data = poses[i] if i < len(poses) else None
            
            # Detect prohibited items near person
            person_bbox = person_det['bbox']
            x1, y1, x2, y2 = map(int, person_bbox)
            person_region = frame[y1:y2, x1:x2]
            
            if person_region.size > 0:
                prohibited_items = self.object_detector.detect_prohibited_items(person_region)
            else:
                prohibited_items = {'phones': [], 'books': [], 'papers': [], 'total': 0}
            
            # Detect anomalies on full frame, not person region
            if self.baseline_set:
                anomaly_results = self.anomaly_detector.detect_anomalies(frame)
                anomalies = anomaly_results['anomalies']
            else:
                anomalies = []
            
            # Prepare frame data for behavior analysis
            frame_data = {
                'detections': person_det,
                'pose': pose_data,
                'prohibited_items': prohibited_items,
                'anomalies': anomalies
            }
            
            # Analyze behavior
            behaviors = self.behavior_analyzer.analyze_frame(
                person_id, frame_data, timestamp
            )
            
            # Update suspicion score
            self.suspicion_scorer.update_score(person_id, behaviors, timestamp)
            
            # Detect patterns
            patterns = self.behavior_analyzer.detect_patterns(person_id)
            if patterns['patterns']:
                self.suspicion_scorer.add_pattern_violation(
                    person_id, patterns['patterns'], timestamp
                )
            
            # Get current score
            score = self.suspicion_scorer.get_score(person_id, timestamp)
            alert_status = self.suspicion_scorer.get_alert_status(person_id)
            
            results['persons'].append({
                'id': person_id,
                'bbox': person_bbox,
                'pose': pose_data,
                'behaviors': behaviors,
                'score': score,
                'alert_status': alert_status,
                'prohibited_items': prohibited_items,
                'anomalies': anomalies
            })
        
        # Get high-risk persons
        results['high_risk_persons'] = self.suspicion_scorer.get_high_risk_persons()
        
        # Update stats
        self.stats['total_frames'] += 1
        self.stats['total_detections'] += len(person_detections)
        
        return results
    
    def _get_or_assign_person_id(self, bbox):
        """
        Simple person tracking - assign ID based on bbox proximity
        
        Args:
            bbox: Person bounding box
        
        Returns:
            int: Person ID
        """
        bbox_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        
        # Find closest tracked person
        min_distance = float('inf')
        closest_id = None
        
        for person_id, tracked_bbox in self.tracked_persons.items():
            tracked_center = [
                (tracked_bbox[0] + tracked_bbox[2]) / 2,
                (tracked_bbox[1] + tracked_bbox[3]) / 2
            ]
            distance = np.sqrt(
                (bbox_center[0] - tracked_center[0]) ** 2 +
                (bbox_center[1] - tracked_center[1]) ** 2
            )
            if distance < min_distance:
                min_distance = distance
                closest_id = person_id
        
        # If close enough, reuse ID
        threshold = 100  # pixels
        if closest_id is not None and min_distance < threshold:
            self.tracked_persons[closest_id] = bbox
            return closest_id
        
        # Assign new ID
        new_id = self.next_person_id
        self.next_person_id += 1
        self.tracked_persons[new_id] = bbox
        return new_id
    
    def _draw_visualization(self, frame, results):
        """
        Draw visualization overlays
        
        Args:
            frame: Input frame
            results: Processing results
        
        Returns:
            frame: Frame with visualizations
        """
        display = frame.copy()
        
        # Draw persons with scores
        for person in results['persons']:
            bbox = person['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Color based on alert status
            alert_status = person['alert_status']
            if alert_status == 'critical':
                color = (0, 0, 255)  # Red
            elif alert_status == 'high':
                color = (0, 165, 255)  # Orange
            elif alert_status == 'medium':
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green
            
            # Draw bounding box
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            score = person['score']
            label = f"ID:{person['id']} Score:{score:.1f} [{alert_status}]"
            
            cv2.putText(
                display, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
            # Draw prohibited items
            for item in person['prohibited_items']['phones']:
                cv2.putText(
                    display, "PHONE!", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )
        
        # Draw info panel
        self._draw_info_panel(display)
        
        return display
    
    def _draw_info_panel(self, frame):
        """Draw information panel on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Text
        info_lines = [
            f"FPS: {self.stats['fps']}",
            f"Frames: {self.stats['total_frames']}",
            f"Detections: {self.stats['total_detections']}",
            f"Tracked: {len(self.tracked_persons)}",
            "",
            "Controls:",
            "Q: Quit | B: Set Baseline | R: Reset | S: Snapshot"
        ]
        
        y = 30
        for line in info_lines:
            cv2.putText(
                frame, line, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            y += 20
    
    def _setup_baseline(self, frame):
        """Setup baseline frame for anomaly detection"""
        logger.info("Setting up baseline frame...")
        self.anomaly_detector.set_baseline(frame)
        self.baseline_set = True
        logger.info("Baseline frame set successfully")
    
    def _reset_system(self):
        """Reset all tracking and scores"""
        logger.info("Resetting system...")
        self.behavior_analyzer = BehaviorAnalyzer()
        self.suspicion_scorer = SuspicionScorer()
        self.tracked_persons.clear()
        self.next_person_id = 1
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_violations': 0,
            'fps': 0
        }
        logger.info("System reset complete")
    
    def _save_snapshot(self, frame):
        """Save current frame as snapshot"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = SNAPSHOTS_DIR / f"snapshot_{timestamp}.jpg"
        cv2.imwrite(str(filename), frame)
        logger.info(f"Snapshot saved: {filename}")
    
    def _print_stats(self):
        """Print system statistics"""
        uptime = time.time() - self.start_time if self.start_time else 0
        logger.info(
            f"Stats - FPS: {self.stats['fps']}, "
            f"Frames: {self.stats['total_frames']}, "
            f"Tracked: {len(self.tracked_persons)}, "
            f"Uptime: {uptime:.1f}s"
        )
    
    def stop(self):
        """Stop the detection system"""
        logger.info("Stopping detection system...")
        self.running = False
        self.video_capture.stop()
        cv2.destroyAllWindows()
        logger.info("Detection system stopped")


def main():
    """Main entry point"""
    print("="*60)
    print("Exam Malpractice Detection System")
    print("="*60)
    print()
    
    # Get camera source from user
    camera_choice = input("Enter camera source (0 for webcam, or RTSP URL): ").strip()
    
    if camera_choice.isdigit():
        camera_source = int(camera_choice)
    else:
        camera_source = camera_choice if camera_choice else 0
    
    # Initialize and start system
    try:
        detector = ExamMalpracticeDetector(camera_source=camera_source)
        detector.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nError: {e}")
        print("Please check the logs for more details.")
    
    print("\nThank you for using Exam Malpractice Detection System!")


if __name__ == "__main__":
    main()