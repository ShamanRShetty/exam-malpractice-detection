"""
Simplified Main Application - Better Performance & Display
Run this instead of main.py if you have issues
"""
import cv2
import time
import numpy as np
from pathlib import Path

# Import modules
from config.settings import *
from utils.logger import get_logger
from src.detection.object_detector import ObjectDetector
from src.detection.pose_detector import PoseDetector, MultiPersonPoseDetector
from src.detection.anomaly_detector import AnomalyDetector
from src.analysis.behavior_analyzer import BehaviorAnalyzer
from src.analysis.suspicion_scorer import SuspicionScorer

logger = get_logger(__name__, str(LOGS_DIR / 'main.log'))


class SimplifiedDetector:
    """Simplified detector with better video handling"""
    
    def __init__(self, camera_source=0):
        """Initialize the detection system"""
        logger.info("="*60)
        logger.info("Initializing Simplified Exam Malpractice Detection System")
        logger.info("="*60)
        
        # Video capture - Direct OpenCV without threading
        self.camera_source = camera_source
        self.cap = cv2.VideoCapture(camera_source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open camera: {camera_source}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        
        logger.info(f"Camera opened: {self.width}x{self.height} @ {self.fps}fps")
        
        # Detection modules
        logger.info("Loading detection modules...")
        self.object_detector = ObjectDetector()
        self.pose_detector = MultiPersonPoseDetector()
        self.anomaly_detector = AnomalyDetector()
        
        # Analysis modules
        logger.info("Loading analysis modules...")
        self.behavior_analyzer = BehaviorAnalyzer()
        self.suspicion_scorer = SuspicionScorer()
        
        # State
        self.running = False
        self.baseline_set = False
        self.frame_count = 0
        self.start_time = None
        
        # Person tracking
        self.tracked_persons = {}
        self.next_person_id = 1
        
        # Statistics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        logger.info("System initialized successfully!")
    
    def start(self):
        """Start the detection system"""
        logger.info("Starting detection system...")
        self.running = True
        self.start_time = time.time()
        
        # Create resizable window
        window_name = 'Exam Malpractice Detection System'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        logger.info("Press 'B' to set baseline, 'Q' to quit")
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.stop()
    
    def _main_loop(self):
        """Main processing loop"""
        logger.info("Entering main loop...")
        
        while self.running:
            # Read frame directly
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                logger.warning("Failed to read frame")
                time.sleep(0.1)
                continue
            
            self.frame_count += 1
            self.fps_counter += 1
            
            # Calculate FPS
            if time.time() - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.fps_start_time = time.time()
            
            # Set baseline on first valid frame
            if not self.baseline_set and self.frame_count == 30:
                logger.info("Auto-setting baseline... (Press B to reset)")
                self._setup_baseline(frame)
            
            # Process every Nth frame based on settings
            if self.frame_count % PERFORMANCE_SETTINGS['frame_skip'] == 0:
                results = self._process_frame(frame)
                display_frame = self._draw_visualization(frame, results)
            else:
                # Just display without processing
                display_frame = frame.copy()
                self._draw_info_panel(display_frame)
            
            # Always show the frame
            cv2.imshow('Exam Malpractice Detection System', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                logger.info("Quit requested")
                break
            elif key == ord('b'):
                self._setup_baseline(frame)
            elif key == ord('r'):
                self._reset_system()
            elif key == ord('s'):
                self._save_snapshot(display_frame)
            elif key == ord('f'):
                # Toggle fullscreen
                cv2.setWindowProperty(
                    'Exam Malpractice Detection System',
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN
                )
            elif key == ord('n'):
                # Toggle normal window
                cv2.setWindowProperty(
                    'Exam Malpractice Detection System',
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL
                )
        
        logger.info("Main loop ended")
    
    def _process_frame(self, frame):
        """Process single frame"""
        results = {
            'persons': [],
            'high_risk_persons': []
        }
        
        try:
            # Detect persons
            person_detections = self.object_detector.detect_persons(frame)
            
            if len(person_detections) == 0:
                return results
            
            # Detect poses
            person_bboxes = [det['bbox'] for det in person_detections]
            poses = self.pose_detector.detect(frame, person_bboxes)
            
            # Process each person
            for i, person_det in enumerate(person_detections):
                person_id = self._get_or_assign_person_id(person_det['bbox'])
                pose_data = poses[i] if i < len(poses) else None
                
                # Get person region for item detection
                person_bbox = person_det['bbox']
                x1, y1, x2, y2 = map(int, person_bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                person_region = frame[y1:y2, x1:x2]
                
                if person_region.size > 0:
                    prohibited_items = self.object_detector.detect_prohibited_items(person_region)
                else:
                    prohibited_items = {'phones': [], 'books': [], 'papers': [], 'total': 0}
                
                # Detect anomalies on full frame
                if self.baseline_set:
                    anomaly_results = self.anomaly_detector.detect_anomalies(frame)
                    anomalies = anomaly_results['anomalies']
                else:
                    anomalies = []
                
                # Prepare frame data
                frame_data = {
                    'detections': person_det,
                    'pose': pose_data,
                    'prohibited_items': prohibited_items,
                    'anomalies': anomalies
                }
                
                # Analyze behavior
                behaviors = self.behavior_analyzer.analyze_frame(
                    person_id, frame_data, time.time()
                )
                
                # Update suspicion score
                self.suspicion_scorer.update_score(person_id, behaviors, time.time())
                
                # Detect patterns
                patterns = self.behavior_analyzer.detect_patterns(person_id)
                if patterns['patterns']:
                    self.suspicion_scorer.add_pattern_violation(
                        person_id, patterns['patterns'], time.time()
                    )
                
                # Get current score
                score = self.suspicion_scorer.get_score(person_id, time.time())
                alert_status = self.suspicion_scorer.get_alert_status(person_id)
                
                results['persons'].append({
                    'id': person_id,
                    'bbox': person_bbox,
                    'pose': pose_data,
                    'score': score,
                    'alert_status': alert_status,
                    'prohibited_items': prohibited_items
                })
            
            # Get high-risk persons
            results['high_risk_persons'] = self.suspicion_scorer.get_high_risk_persons()
        
        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
        
        return results
    
    def _get_or_assign_person_id(self, bbox):
        """Simple person tracking based on bbox proximity"""
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
        threshold = 100
        if closest_id is not None and min_distance < threshold:
            self.tracked_persons[closest_id] = bbox
            return closest_id
        
        # Assign new ID
        new_id = self.next_person_id
        self.next_person_id += 1
        self.tracked_persons[new_id] = bbox
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
            
            # Label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                display,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Label text
            cv2.putText(
                display, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            
            # Alert status
            cv2.putText(
                display, alert_status.upper(), (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
            # Show prohibited items
            if person['prohibited_items']['phones']:
                cv2.putText(
                    display, "PHONE DETECTED!", (x1, y2 + 40),
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
        cv2.rectangle(overlay, (10, 10), (450, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Info text
        info_lines = [
            f"FPS: {self.current_fps}",
            f"Frame: {self.frame_count}",
            f"Tracked Persons: {len(self.tracked_persons)}",
            f"Baseline: {'SET' if self.baseline_set else 'NOT SET'}",
            "",
            "CONTROLS:",
            "Q/ESC - Quit  |  B - Set Baseline",
            "R - Reset  |  S - Snapshot",
            "F - Fullscreen  |  N - Normal Window"
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
        logger.info("Baseline frame set successfully!")
        print("\n✓ Baseline set! System is now monitoring for anomalies.")
    
    def _reset_system(self):
        """Reset all tracking and scores"""
        logger.info("Resetting system...")
        self.behavior_analyzer = BehaviorAnalyzer()
        self.suspicion_scorer = SuspicionScorer()
        self.tracked_persons.clear()
        self.next_person_id = 1
        logger.info("System reset complete")
        print("\n✓ System reset!")
    
    def _save_snapshot(self, frame):
        """Save current frame as snapshot"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = SNAPSHOTS_DIR / f"snapshot_{timestamp}.jpg"
        cv2.imwrite(str(filename), frame)
        logger.info(f"Snapshot saved: {filename}")
        print(f"\n✓ Snapshot saved: {filename}")
    
    def stop(self):
        """Stop the detection system"""
        logger.info("Stopping detection system...")
        self.running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Detection system stopped")


def main():
    """Main entry point"""
    print("="*60)
    print("  Exam Malpractice Detection System (Simplified Version)")
    print("="*60)
    print()
    print("This version has:")
    print("✓ Better video capture (no freezing)")
    print("✓ Resizable window (works properly)")
    print("✓ Improved performance")
    print()
    
    # Get camera source
    camera_choice = input("Enter camera source (0 for webcam, or RTSP URL): ").strip()
    
    if camera_choice.isdigit():
        camera_source = int(camera_choice)
    else:
        camera_source = camera_choice if camera_choice else 0
    
    print()
    print("Starting system...")
    print()
    
    # Initialize and start
    try:
        detector = SimplifiedDetector(camera_source=camera_source)
        detector.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("Please check the logs for more details.")
        input("\nPress Enter to exit...")
    
    print("\nThank you for using the system!")


if __name__ == "__main__":
    main()