"""
Main Application - Exam Malpractice Detection System - FIXED VERSION
Complete integration of all modules with error handling
"""
import cv2
import time
import sys
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
    """Main application class - FIXED"""
    
    def __init__(self, camera_source=0):
        """
        Initialize the detection system
        
        Args:
            camera_source: Camera source (0 for webcam, RTSP URL for IP camera)
        """
        logger.info("="*60)
        logger.info("Initializing Exam Malpractice Detection System")
        logger.info("="*60)
        
        # Check if model exists
        model_path = Path(YOLO_SETTINGS['model_path'])
        if not model_path.exists():
            logger.error(f"YOLO model not found: {model_path}")
            print("\n" + "="*60)
            print("ERROR: YOLO model not found!")
            print("="*60)
            print(f"\nModel should be at: {model_path}")
            print("\nPlease run: python setup_models.py")
            print("\nThis will download the required model files.")
            print("="*60 + "\n")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Video capture
        self.camera_source = camera_source
        try:
            self.video_capture = VideoCapture(camera_source, buffer_size=CAMERA_SETTINGS['buffer_size'])
            self.frame_buffer = FrameBuffer(
                max_seconds=BEHAVIOR_SETTINGS['timeline_buffer_seconds'],
                fps=self.video_capture.fps
            )
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            raise
        
        # Detection modules
        logger.info("Loading detection modules...")
        try:
            self.object_detector = ObjectDetector()
            logger.info("✓ Object detector loaded")
        except Exception as e:
            logger.error(f"Failed to load object detector: {e}")
            raise
        
        try:
            self.pose_detector = MultiPersonPoseDetector()
            logger.info("✓ Pose detector loaded")
        except Exception as e:
            logger.error(f"Failed to load pose detector: {e}")
            raise
        
        try:
            self.anomaly_detector = AnomalyDetector()
            logger.info("✓ Anomaly detector loaded")
        except Exception as e:
            logger.error(f"Failed to load anomaly detector: {e}")
            raise
        
        # Analysis modules
        logger.info("Loading analysis modules...")
        self.behavior_analyzer = BehaviorAnalyzer()
        self.suspicion_scorer = SuspicionScorer()
        
        # Privacy
        if PRIVACY_SETTINGS['enable_face_blur']:
            try:
                self.face_blurrer = FaceBlurrer()
                logger.info("✓ Face blurring enabled")
            except Exception as e:
                logger.warning(f"Face blurring unavailable: {e}")
                self.face_blurrer = None
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
            'fps': 0,
            'processing_time': 0
        }
        
        logger.info("System initialized successfully!")
    
    def start(self):
        """Start the detection system"""
        logger.info("Starting detection system...")
        self.running = True
        self.start_time = time.time()
        
        # Start video capture
        self.video_capture.start()
        
        # Wait for camera to stabilize
        logger.info("Waiting for camera to stabilize...")
        time.sleep(2)
        
        # Check if camera is working
        ret, test_frame = self.video_capture.read()
        if not ret or test_frame is None:
            logger.error("Failed to read from camera!")
            print("\nERROR: Cannot read from camera!")
            print("Please check:")
            print("1. Camera is connected")
            print("2. Camera permissions are granted")
            print("3. No other application is using the camera")
            self.stop()
            return
        
        logger.info(f"Camera working! Frame size: {test_frame.shape}")
        
        # Create window
        window_name = 'Exam Malpractice Detection System'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        print("\n" + "="*60)
        print("System is running!")
        print("="*60)
        print("\nControls:")
        print("  Q - Quit")
        print("  B - Set new baseline (for anomaly detection)")
        print("  R - Reset all tracking and scores")
        print("  S - Save snapshot")
        print("="*60 + "\n")
        
        # Main processing loop
        try:
            self._main_loop(window_name)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.stop()
    
    def _main_loop(self, window_name):
        """Main processing loop - FIXED"""
        logger.info("Entering main processing loop...")
        
        fps_start_time = time.time()
        fps_frame_count = 0
        
        while self.running:
            loop_start = time.time()
            
            # Read frame
            ret, frame = self.video_capture.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame, retrying...")
                time.sleep(0.1)
                continue
            
            self.frame_count += 1
            fps_frame_count += 1
            
            # Add to buffer
            self.frame_buffer.add(frame, loop_start)
            
            # Set baseline on first valid frame (after 10 frames for stability)
            if not self.baseline_set and self.frame_count == 10:
                self._setup_baseline(frame)
            
            # Skip frames for performance
            if self.frame_count % PERFORMANCE_SETTINGS['frame_skip'] != 0:
                # Still display the frame
                display_frame = frame.copy()
                self._draw_info_panel(display_frame)
                cv2.imshow(window_name, display_frame)
                
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
            try:
                processing_start = time.time()
                results = self._process_frame(frame, loop_start)
                self.stats['processing_time'] = time.time() - processing_start
                
                # Draw visualization
                display_frame = self._draw_visualization(frame, results)
                
                # Apply privacy if enabled
                if self.face_blurrer is not None:
                    display_frame = self.face_blurrer.blur_faces(display_frame)
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}", exc_info=True)
                display_frame = frame.copy()
                cv2.putText(
                    display_frame, f"Processing Error: {str(e)[:50]}", (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                )
            
            # Display frame
            cv2.imshow(window_name, display_frame)
            
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
            if self.frame_count % 300 == 0:  # Every 10 seconds at 30fps
                self._print_stats()
        
        logger.info("Main loop ended")
    
    def _process_frame(self, frame, timestamp):
        """
        Process single frame - FIXED
        
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
        try:
            person_detections = self.object_detector.detect_persons(frame)
        except Exception as e:
            logger.error(f"Person detection failed: {e}")
            person_detections = []
        
        if len(person_detections) == 0:
            return results
        
        # Detect poses for each person
        person_bboxes = [det['bbox'] for det in person_detections]
        try:
            poses = self.pose_detector.detect(frame, person_bboxes)
        except Exception as e:
            logger.error(f"Pose detection failed: {e}")
            poses = []
        
        # Match poses with persons
        for i, person_det in enumerate(person_detections):
            try:
                person_id = self._get_or_assign_person_id(person_det['bbox'])
                
                pose_data = poses[i] if i < len(poses) else None
                
                # Detect prohibited items near person
                person_bbox = person_det['bbox']
                x1, y1, x2, y2 = map(int, person_bbox)
                
                # Ensure valid crop region
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    person_region = frame[y1:y2, x1:x2]
                    try:
                        prohibited_items = self.object_detector.detect_prohibited_items(person_region)
                    except Exception as e:
                        logger.error(f"Prohibited item detection failed: {e}")
                        prohibited_items = {'phones': [], 'books': [], 'papers': [], 'total': 0}
                else:
                    prohibited_items = {'phones': [], 'books': [], 'papers': [], 'total': 0}
                
                # Detect anomalies (only if baseline is set)
                anomalies = []
                if self.baseline_set:
                    try:
                        anomaly_results = self.anomaly_detector.detect_anomalies(frame)
                        anomalies = anomaly_results['anomalies']
                    except Exception as e:
                        logger.error(f"Anomaly detection failed: {e}")
                
                # Prepare frame data
                frame_data = {
                    'detections': person_det,
                    'pose': pose_data,
                    'prohibited_items': prohibited_items,
                    'anomalies': anomalies
                }
                
                # Analyze behavior
                try:
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
                    
                except Exception as e:
                    logger.error(f"Behavior analysis failed: {e}")
                    behaviors = {}
                    score = 0.0
                    alert_status = 'normal'
                
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
                
            except Exception as e:
                logger.error(f"Error processing person: {e}", exc_info=True)
                continue
        
        # Get high-risk persons
        try:
            results['high_risk_persons'] = self.suspicion_scorer.get_high_risk_persons()
        except Exception as e:
            logger.error(f"Failed to get high-risk persons: {e}")
        
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
        logger.info(f"New person detected: ID {new_id}")
        return new_id
    
    def _draw_visualization(self, frame, results):
        """Draw visualization overlays"""
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
            label = f"ID:{person['id']} Score:{score:.1f}"
            
            # Add status indicator
            if alert_status != 'normal':
                label += f" [{alert_status.upper()}]"
            
            cv2.putText(
                display, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
            # Draw prohibited items warning
            prohibited = person['prohibited_items']
            if prohibited['phones']:
                cv2.putText(
                    display, "PHONE DETECTED!", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )
            if prohibited['books']:
                cv2.putText(
                    display, "BOOK DETECTED!", (x1, y2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
                )
        
        # Draw info panel
        self._draw_info_panel(display)
        
        return display
    
    def _draw_info_panel(self, frame):
        """Draw information panel on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Status indicator
        status_color = (0, 255, 0) if self.baseline_set else (0, 165, 255)
        status_text = "ACTIVE" if self.baseline_set else "WARMING UP"
        
        # Text
        info_lines = [
            f"Status: {status_text}",
            f"FPS: {self.stats['fps']} | Process: {self.stats['processing_time']*1000:.0f}ms",
            f"Frames: {self.stats['total_frames']} | Detected: {self.stats['total_detections']}",
            f"Tracked Persons: {len(self.tracked_persons)}",
            f"Baseline: {'SET' if self.baseline_set else 'NOT SET'}",
            "",
            "Controls:",
            "Q: Quit | B: Baseline | R: Reset | S: Snapshot"
        ]
        
        y = 30
        for i, line in enumerate(info_lines):
            color = status_color if i == 0 else (255, 255, 255)
            cv2.putText(
                frame, line, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
            )
            y += 20
    
    def _setup_baseline(self, frame):
        """Setup baseline frame for anomaly detection"""
        logger.info("Setting up baseline frame...")
        try:
            self.anomaly_detector.set_baseline(frame)
            self.baseline_set = True
            logger.info("✓ Baseline frame set successfully")
            print("\n✓ Baseline captured! Anomaly detection active.\n")
        except Exception as e:
            logger.error(f"Failed to set baseline: {e}")
            print(f"\n✗ Failed to set baseline: {e}\n")
    
    def _reset_system(self):
        """Reset all tracking and scores"""
        logger.info("Resetting system...")
        try:
            self.behavior_analyzer = BehaviorAnalyzer()
            self.suspicion_scorer = SuspicionScorer()
            self.tracked_persons.clear()
            self.next_person_id = 1
            self.stats = {
                'total_frames': 0,
                'total_detections': 0,
                'total_violations': 0,
                'fps': 0,
                'processing_time': 0
            }
            logger.info("✓ System reset complete")
            print("\n✓ System reset!\n")
        except Exception as e:
            logger.error(f"Failed to reset system: {e}")
    
    def _save_snapshot(self, frame):
        """Save current frame as snapshot"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = SNAPSHOTS_DIR / f"snapshot_{timestamp}.jpg"
            cv2.imwrite(str(filename), frame)
            logger.info(f"✓ Snapshot saved: {filename}")
            print(f"\n✓ Snapshot saved: {filename.name}\n")
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
    
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
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("ERROR: Python 3.7 or higher is required")
        return
    
    # Get camera source
    print("Camera Options:")
    print("  0 - Default webcam")
    print("  1 - Secondary camera")
    print("  RTSP URL - Network camera (e.g., rtsp://...)")
    print()
    camera_choice = input("Enter camera source [0]: ").strip()
    
    if not camera_choice:
        camera_source = 0
    elif camera_choice.isdigit():
        camera_source = int(camera_choice)
    else:
        camera_source = camera_choice
    
    print()
    
    # Initialize and start system
    try:
        detector = ExamMalpracticeDetector(camera_source=camera_source)
        detector.start()
    except FileNotFoundError as e:
        # Model not found error already printed in __init__
        pass
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n{'='*60}")
        print("FATAL ERROR")
        print("="*60)
        print(f"\nError: {e}")
        print("\nPlease check:")
        print("1. All dependencies are installed (pip install -r requirements.txt)")
        print("2. Camera is accessible")
        print("3. Models are downloaded (python setup_models.py)")
        print(f"\nCheck logs at: {LOGS_DIR}/main.log")
        print("="*60 + "\n")
    
    print("\nThank you for using Exam Malpractice Detection System!")


if __name__ == "__main__":
    main()