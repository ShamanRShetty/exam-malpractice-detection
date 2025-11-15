"""
MediaPipe-based pose detection for behavior analysis
"""
import cv2
import numpy as np
import mediapipe as mp
from utils.logger import get_logger
from config.settings import POSE_SETTINGS, BEHAVIOR_SETTINGS

logger = get_logger(__name__)


class PoseDetector:
    """MediaPipe pose detector for tracking student body positions"""
    
    def __init__(self):
        """Initialize MediaPipe Pose"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            model_complexity=POSE_SETTINGS['model_complexity'],
            min_detection_confidence=POSE_SETTINGS['min_detection_confidence'],
            min_tracking_confidence=POSE_SETTINGS['min_tracking_confidence'],
            enable_segmentation=POSE_SETTINGS['enable_segmentation']
        )
        
        self.landmarks_indices = POSE_SETTINGS['landmarks']
        
        logger.info("MediaPipe Pose detector initialized")
    
    def detect(self, frame):
        """
        Detect pose landmarks in frame
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            dict: Pose detection results
        """
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Extract landmarks
        landmarks = self._extract_landmarks(results.pose_landmarks, frame.shape)
        
        # Calculate pose features
        pose_data = {
            'landmarks': landmarks,
            'head_direction': self._calculate_head_direction(landmarks),
            'hand_positions': self._get_hand_positions(landmarks, frame.shape),
            'body_lean': self._calculate_body_lean(landmarks),
            'looking_down': self._is_looking_down(landmarks),
            'hands_below_desk': self._are_hands_below_desk(landmarks, frame.shape),
        }
        
        return pose_data
    
    def _extract_landmarks(self, pose_landmarks, frame_shape):
        """
        Extract landmark coordinates
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            frame_shape: Frame dimensions
        
        Returns:
            dict: Landmark coordinates
        """
        h, w = frame_shape[:2]
        landmarks = {}
        
        for name, idx in self.landmarks_indices.items():
            landmark = pose_landmarks.landmark[idx]
            landmarks[name] = {
                'x': landmark.x * w,
                'y': landmark.y * h,
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        
        return landmarks
    
    def _calculate_head_direction(self, landmarks):
        """
        Calculate head direction angle
        
        Args:
            landmarks: Extracted landmarks
        
        Returns:
            dict: Head direction info
        """
        nose = landmarks['nose']
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        
        # Calculate center of eyes
        eye_center_x = (left_eye['x'] + right_eye['x']) / 2
        
        # Horizontal angle (left/right turn)
        horizontal_angle = np.arctan2(nose['x'] - eye_center_x, 100) * 180 / np.pi
        
        # Vertical angle (up/down tilt)
        eye_center_y = (left_eye['y'] + right_eye['y']) / 2
        vertical_angle = np.arctan2(nose['y'] - eye_center_y, 100) * 180 / np.pi
        
        # Determine direction
        direction = 'center'
        if abs(horizontal_angle) > BEHAVIOR_SETTINGS['head_turn_threshold']:
            direction = 'right' if horizontal_angle > 0 else 'left'
        
        return {
            'horizontal_angle': horizontal_angle,
            'vertical_angle': vertical_angle,
            'direction': direction,
            'is_turned': abs(horizontal_angle) > BEHAVIOR_SETTINGS['head_turn_threshold']
        }
    
    def _get_hand_positions(self, landmarks, frame_shape):
        """
        Get hand positions relative to body
        
        Args:
            landmarks: Extracted landmarks
            frame_shape: Frame dimensions
        
        Returns:
            dict: Hand position info
        """
        h, w = frame_shape[:2]
        
        left_wrist = landmarks['left_wrist']
        right_wrist = landmarks['right_wrist']
        left_shoulder = landmarks['left_shoulder']
        right_shoulder = landmarks['right_shoulder']
        
        # Calculate relative positions
        left_hand_below_shoulder = left_wrist['y'] > left_shoulder['y']
        right_hand_below_shoulder = right_wrist['y'] > right_shoulder['y']
        
        return {
            'left_wrist': left_wrist,
            'right_wrist': right_wrist,
            'left_below_shoulder': left_hand_below_shoulder,
            'right_below_shoulder': right_hand_below_shoulder,
        }
    
    def _calculate_body_lean(self, landmarks):
        """
        Calculate body lean angle
        
        Args:
            landmarks: Extracted landmarks
        
        Returns:
            dict: Body lean info
        """
        left_shoulder = landmarks['left_shoulder']
        right_shoulder = landmarks['right_shoulder']
        left_hip = landmarks['left_hip']
        right_hip = landmarks['right_hip']
        
        # Calculate shoulder line angle
        shoulder_angle = np.arctan2(
            right_shoulder['y'] - left_shoulder['y'],
            right_shoulder['x'] - left_shoulder['x']
        ) * 180 / np.pi
        
        # Calculate hip line angle
        hip_angle = np.arctan2(
            right_hip['y'] - left_hip['y'],
            right_hip['x'] - left_hip['x']
        ) * 180 / np.pi
        
        # Body lean is difference from vertical (90 degrees)
        lean_angle = abs(90 - abs(shoulder_angle))
        
        is_leaning = lean_angle > BEHAVIOR_SETTINGS['lean_angle_threshold']
        
        return {
            'shoulder_angle': shoulder_angle,
            'hip_angle': hip_angle,
            'lean_angle': lean_angle,
            'is_leaning': is_leaning,
            'direction': 'left' if shoulder_angle < 90 else 'right'
        }
    
    def _is_looking_down(self, landmarks):
        """
        Determine if person is looking down
        
        Args:
            landmarks: Extracted landmarks
        
        Returns:
            bool: True if looking down
        """
        nose = landmarks['nose']
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        
        eye_center_y = (left_eye['y'] + right_eye['y']) / 2
        
        # If nose is significantly below eye level
        vertical_diff = nose['y'] - eye_center_y
        
        # Calculate angle
        angle = np.arctan2(vertical_diff, 100) * 180 / np.pi
        
        return angle > BEHAVIOR_SETTINGS['looking_down_angle']
    
    def _are_hands_below_desk(self, landmarks, frame_shape):
        """
        Check if hands are below desk level (lower portion of frame)
        
        Args:
            landmarks: Extracted landmarks
            frame_shape: Frame dimensions
        
        Returns:
            dict: Hand below desk info
        """
        h = frame_shape[0]
        threshold_y = h * BEHAVIOR_SETTINGS['hand_below_desk_threshold']
        
        left_wrist = landmarks['left_wrist']
        right_wrist = landmarks['right_wrist']
        
        left_below = left_wrist['y'] > threshold_y
        right_below = right_wrist['y'] > threshold_y
        
        return {
            'left_below': left_below,
            'right_below': right_below,
            'any_below': left_below or right_below,
            'both_below': left_below and right_below
        }
    
    def draw_pose(self, frame, pose_data):
        """
        Draw pose landmarks on frame
        
        Args:
            frame: Input frame
            pose_data: Pose detection data
        
        Returns:
            frame: Frame with drawn pose
        """
        if pose_data is None:
            return frame
        
        result = frame.copy()
        landmarks = pose_data['landmarks']
        
        # Draw landmarks
        for name, lm in landmarks.items():
            x, y = int(lm['x']), int(lm['y'])
            cv2.circle(result, (x, y), 5, (0, 255, 0), -1)
        
        # Draw connections
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
        ]
        
        for start, end in connections:
            if start in landmarks and end in landmarks:
                start_lm = landmarks[start]
                end_lm = landmarks[end]
                pt1 = (int(start_lm['x']), int(start_lm['y']))
                pt2 = (int(end_lm['x']), int(end_lm['y']))
                cv2.line(result, pt1, pt2, (0, 255, 0), 2)
        
        # Draw head direction indicator
        head_dir = pose_data['head_direction']
        nose = landmarks['nose']
        nose_pt = (int(nose['x']), int(nose['y']))
        
        # Draw arrow showing head direction
        angle_rad = head_dir['horizontal_angle'] * np.pi / 180
        arrow_length = 50
        end_pt = (
            int(nose_pt[0] + arrow_length * np.sin(angle_rad)),
            int(nose_pt[1] + arrow_length * np.cos(angle_rad))
        )
        cv2.arrowedLine(result, nose_pt, end_pt, (255, 0, 0), 2, tipLength=0.3)
        
        return result
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'pose'):
            self.pose.close()


class MultiPersonPoseDetector:
    """Detect poses for multiple persons in frame"""
    
    def __init__(self):
        """Initialize multi-person pose detector"""
        self.single_detector = PoseDetector()
        logger.info("Multi-person pose detector initialized")
    
    def detect(self, frame, person_bboxes):
        """
        Detect poses for multiple persons
        
        Args:
            frame: Input frame
            person_bboxes: List of person bounding boxes
        
        Returns:
            list: List of pose data for each person
        """
        poses = []
        
        for bbox in person_bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Crop person region
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                continue
            
            # Detect pose
            pose_data = self.single_detector.detect(person_crop)
            
            if pose_data:
                # Adjust landmark coordinates to full frame
                for lm in pose_data['landmarks'].values():
                    lm['x'] += x1
                    lm['y'] += y1
                
                pose_data['bbox'] = bbox
                poses.append(pose_data)
        
        return poses