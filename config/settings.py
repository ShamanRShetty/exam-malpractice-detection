"""
Configuration settings for Exam Malpractice Detection System
FIXED VERSION - Replace your config/settings.py with this
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
EVIDENCE_DIR = DATA_DIR / "evidence_clips"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
LOGS_DIR = DATA_DIR / "logs"
BASELINE_DIR = DATA_DIR / "baseline_frames"

# Create directories if they don't exist
for directory in [EVIDENCE_DIR, SNAPSHOTS_DIR, LOGS_DIR, BASELINE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CAMERA SETTINGS
# ============================================================================
CAMERA_SETTINGS = {
    'default_source': 0,  # 0 for webcam, or RTSP URL
    'fps': 30,
    'resolution': (1280, 720),  # (width, height)
    'buffer_size': 450,  # 15 seconds at 30fps
}

# Multiple camera support
CAMERAS = {
    'camera_1': {
        'source': 0,  # Webcam or "rtsp://username:password@ip:port/stream"
        'name': 'Front View',
        'zone': 'hall_main'
    },
}

# ============================================================================
# YOLO DETECTION SETTINGS
# ============================================================================
YOLO_SETTINGS = {
    'model_path': str(MODELS_DIR / "yolov8n.pt"),
    'confidence_threshold': 0.6,  # INCREASED from 0.5
    'iou_threshold': 0.4,
    'device': 'cpu',  # Change to 'cuda' if you have NVIDIA GPU
    
    # Classes to detect (COCO dataset indices)
    'target_classes': {
        0: 'person',
        67: 'cell phone',
        73: 'book',
        84: 'paper',
    },
    
    # Detection regions (adjust based on your exam hall layout)
    'detection_zones': {
        'desk_area': (0.2, 0.4, 0.8, 0.9),  # (x1, y1, x2, y2) normalized
        'restricted_area': (0.0, 0.0, 1.0, 0.3),  # Top area
    }
}

# ============================================================================
# MEDIAPIPE POSE SETTINGS
# ============================================================================
POSE_SETTINGS = {
    'model_complexity': 1,  # 0, 1, or 2 (higher = more accurate but slower)
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'enable_segmentation': False,
    
    # Landmark indices (MediaPipe Pose)
    'landmarks': {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 7,
        'right_ear': 8,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
    }
}

# ============================================================================
# BEHAVIOR ANALYSIS SETTINGS - FIXED
# ============================================================================
BEHAVIOR_SETTINGS = {
    'timeline_buffer_seconds': 15,
    'analysis_interval_frames': 5,
    
    # Head direction thresholds (angles in degrees) - LESS SENSITIVE
    'head_turn_threshold': 45,  # INCREASED from 30
    'suspicious_head_turn_duration': 5.0,  # INCREASED from 3.0
    
    # Hand movement thresholds - LESS SENSITIVE
    'hand_below_desk_threshold': 0.85,  # INCREASED from 0.7 (hands must be much lower)
    'suspicious_hand_duration': 5.0,  # INCREASED from 2.0
    
    # Body leaning thresholds - LESS SENSITIVE
    'lean_angle_threshold': 30,  # INCREASED from 15 (must lean more)
    'suspicious_lean_duration': 6.0,  # INCREASED from 3.0
    
    # Pattern detection
    'pattern_window_seconds': 10,
    'repeated_action_threshold': 5,  # INCREASED from 3
    
    # Looking down detection - LESS SENSITIVE
    'looking_down_angle': 55,  # INCREASED from 40
    'looking_down_suspicious_duration': 8.0,  # INCREASED from 5.0
}

# ============================================================================
# SUSPICION SCORING SYSTEM - FIXED
# ============================================================================
SUSPICION_SETTINGS = {
    'max_score': 100,
    'alert_threshold': 70,
    'high_alert_threshold': 85,
    
    # Score weights for different violations - REDUCED
    'weights': {
        'phone_detected': 30,
        'book_detected': 25,
        'paper_detected': 20,
        'head_turn_suspicious': 4,      # REDUCED from 10
        'looking_down_extended': 6,     # REDUCED from 12
        'hand_below_desk': 5,           # REDUCED from 15 (BIG CHANGE!)
        'body_leaning': 3,              # REDUCED from 8 (BIG CHANGE!)
        'repeated_head_turns': 8,       # REDUCED from 15
        'desk_anomaly': 4,              # REDUCED from 18 (BIG CHANGE!)
        'geofence_violation': 15,
        'seat_swap': 30,
        'talking_detected': 25,
    },
    
    # Score decay (points per second of good behavior) - FASTER
    'decay_rate': 4.0,  # INCREASED from 2.0 (forgives faster)
    'decay_interval': 1.0,  # Seconds between decay updates
    
    # Minimum score (never goes below this)
    'min_score': 0,
}

# ============================================================================
# ANOMALY DETECTION SETTINGS - FIXED
# ============================================================================
ANOMALY_SETTINGS = {
    'baseline_capture_delay': 5.0,
    'baseline_update_interval': 300,
    'difference_threshold': 60,  # INCREASED from 30 (MUCH less sensitive)
    'min_contour_area': 2000,    # INCREASED from 500 (ignore small changes)
    'background_subtractor_history': 500,
    'background_subtractor_threshold': 25,  # INCREASED from 16
}

# ============================================================================
# GEO-FENCING SETTINGS
# ============================================================================
GEOFENCE_SETTINGS = {
    'zones': {
        'safe_zone': {
            'coordinates': [(0.2, 0.3), (0.8, 0.3), (0.8, 0.9), (0.2, 0.9)],
            'violation_threshold': 0.5,
        },
        'restricted_zone': {
            'coordinates': [(0.0, 0.0), (1.0, 0.0), (1.0, 0.2), (0.0, 0.2)],
            'type': 'forbidden',
        },
    },
    'enable': True,
}

# ============================================================================
# PERSON RE-IDENTIFICATION SETTINGS
# ============================================================================
REID_SETTINGS = {
    'enable': True,
    'feature_extraction_interval': 30,
    'similarity_threshold': 0.7,
    'max_tracked_persons': 50,
    'track_timeout': 30,
    
    'seats': {
    },
    'seat_swap_threshold': 0.3,
}

# ============================================================================
# AUDIO DETECTION SETTINGS
# ============================================================================
AUDIO_SETTINGS = {
    'enable': False,  # Disabled by default
    'sample_rate': 44100,
    'chunk_size': 4096,
    'channels': 1,
    'format': 'int16',
    
    'energy_threshold': 500,
    'zero_crossing_rate_threshold': 0.1,
    'talking_duration_threshold': 1.0,
    
    'freq_bands': {
        'low': (0, 300),
        'speech': (300, 3400),
        'high': (3400, 8000),
    },
}

# ============================================================================
# EVIDENCE CLIP SETTINGS
# ============================================================================
EVIDENCE_SETTINGS = {
    'pre_event_duration': 5,
    'post_event_duration': 5,
    'codec': 'mp4v',
    'fps': 30,
    'snapshot_on_alert': True,
    'save_original_resolution': True,
}

# ============================================================================
# DASHBOARD SETTINGS
# ============================================================================
DASHBOARD_SETTINGS = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'update_interval': 1.0,
    'max_alerts_display': 50,
    'enable_heatmap': True,
    'heatmap_update_interval': 30,
}

# ============================================================================
# ALERT SETTINGS
# ============================================================================
ALERT_SETTINGS = {
    'enable_desktop_notifications': True,
    'enable_sound_alerts': True,
    'enable_email': False,
    
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'email_from': 'your_email@gmail.com',
    'email_password': 'your_password',
    'email_to': ['teacher@school.com'],
    
    'levels': {
        'low': {'score_range': (30, 50), 'color': 'yellow'},
        'medium': {'score_range': (50, 70), 'color': 'orange'},
        'high': {'score_range': (70, 85), 'color': 'red'},
        'critical': {'score_range': (85, 100), 'color': 'darkred'},
    },
}

# ============================================================================
# PRIVACY SETTINGS
# ============================================================================
PRIVACY_SETTINGS = {
    'enable_face_blur': False,  # Set to True for privacy mode
    'blur_strength': 25,
    'show_bounding_boxes': True,
    'anonymize_logs': False,
}

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================
PERFORMANCE_SETTINGS = {
    'use_multithreading': True,
    'max_threads': 4,
    'frame_skip': 2,  # Process every 2nd frame
    'resize_for_processing': (640, 480),
    'enable_gpu': False,
}

# ============================================================================
# LOGGING SETTINGS
# ============================================================================
LOGGING_SETTINGS = {
    'level': 'INFO',  # Changed from DEBUG to INFO (less noise)
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': str(LOGS_DIR / 'system.log'),
    'max_bytes': 10 * 1024 * 1024,  # 10 MB
    'backup_count': 5,
}