"""
Main behavior analysis engine - combines all detections
"""
import time
from collections import deque, defaultdict
import numpy as np
from utils.logger import get_logger
from config.settings import BEHAVIOR_SETTINGS, SUSPICION_SETTINGS

logger = get_logger(__name__)


class BehaviorAnalyzer:
    """Analyzes student behavior and generates suspicion scores"""
    
    def __init__(self):
        """Initialize behavior analyzer"""
        self.timeline_buffer_size = int(
            BEHAVIOR_SETTINGS['timeline_buffer_seconds'] * 30  # Assuming 30fps
        )
        
        # Timeline storage for each tracked person
        self.person_timelines = defaultdict(lambda: {
            'events': deque(maxlen=self.timeline_buffer_size),
            'poses': deque(maxlen=self.timeline_buffer_size),
            'detections': deque(maxlen=self.timeline_buffer_size),
            'timestamps': deque(maxlen=self.timeline_buffer_size),
        })
        
        # Behavior counters
        self.behavior_counters = defaultdict(lambda: defaultdict(int))
        
        # Continuous behavior tracking
        self.continuous_behaviors = defaultdict(lambda: {
            'head_turned_start': None,
            'looking_down_start': None,
            'hand_below_start': None,
            'leaning_start': None,
        })
        
        logger.info("Behavior analyzer initialized")
    
    def analyze_frame(self, person_id, frame_data, timestamp=None):
        """
        Analyze single frame for a person
        
        Args:
            person_id: Unique person identifier
            frame_data: Dict containing detections, pose, etc.
            timestamp: Frame timestamp
        
        Returns:
            dict: Analysis results
        """
        if timestamp is None:
            timestamp = time.time()
        
        timeline = self.person_timelines[person_id]
        timeline['timestamps'].append(timestamp)
        
        # Store frame data
        timeline['detections'].append(frame_data.get('detections', []))
        timeline['poses'].append(frame_data.get('pose', None))
        
        # Analyze behaviors
        behaviors = self._analyze_behaviors(person_id, frame_data, timestamp)
        
        # Store behavior events
        timeline['events'].append(behaviors)
        
        return behaviors
    
    def _analyze_behaviors(self, person_id, frame_data, timestamp):
        """
        Analyze all behaviors for current frame
        
        Args:
            person_id: Person ID
            frame_data: Frame detection data
            timestamp: Current timestamp
        
        Returns:
            dict: Detected behaviors
        """
        behaviors = {
            'prohibited_items': [],
            'suspicious_poses': [],
            'anomalies': [],
            'continuous_violations': [],
            'timestamp': timestamp
        }
        
        # Check for prohibited items
        if 'prohibited_items' in frame_data:
            items = frame_data['prohibited_items']
            if items['phones']:
                behaviors['prohibited_items'].append('phone')
            if items['books']:
                behaviors['prohibited_items'].append('book')
            if items['papers']:
                behaviors['prohibited_items'].append('paper')
        
        # Analyze pose
        if 'pose' in frame_data and frame_data['pose'] is not None:
            pose = frame_data['pose']
            
            # Head turning
            if pose['head_direction']['is_turned']:
                behaviors['suspicious_poses'].append('head_turned')
                self._track_continuous_behavior(
                    person_id, 'head_turned', timestamp,
                    BEHAVIOR_SETTINGS['suspicious_head_turn_duration']
                )
            else:
                self._reset_continuous_behavior(person_id, 'head_turned')
            
            # Looking down
            if pose['looking_down']:
                behaviors['suspicious_poses'].append('looking_down')
                self._track_continuous_behavior(
                    person_id, 'looking_down', timestamp,
                    BEHAVIOR_SETTINGS['looking_down_suspicious_duration']
                )
            else:
                self._reset_continuous_behavior(person_id, 'looking_down')
            
            # Hand below desk
            if pose['hands_below_desk']['any_below']:
                behaviors['suspicious_poses'].append('hand_below_desk')
                self._track_continuous_behavior(
                    person_id, 'hand_below', timestamp,
                    BEHAVIOR_SETTINGS['suspicious_hand_duration']
                )
            else:
                self._reset_continuous_behavior(person_id, 'hand_below')
            
            # Body leaning
            if pose['body_lean']['is_leaning']:
                behaviors['suspicious_poses'].append('body_leaning')
                self._track_continuous_behavior(
                    person_id, 'leaning', timestamp,
                    BEHAVIOR_SETTINGS['suspicious_lean_duration']
                )
            else:
                self._reset_continuous_behavior(person_id, 'leaning')
        
        # Check anomalies
        if 'anomalies' in frame_data and frame_data['anomalies']:
            behaviors['anomalies'] = frame_data['anomalies']
        
        # Get continuous violations
        behaviors['continuous_violations'] = self._get_active_continuous_violations(
            person_id, timestamp
        )
        
        return behaviors
    
    def _track_continuous_behavior(self, person_id, behavior_type, timestamp, threshold):
        """
        Track continuous behavior duration
        
        Args:
            person_id: Person ID
            behavior_type: Type of behavior
            timestamp: Current timestamp
            threshold: Duration threshold for violation
        """
        continuous = self.continuous_behaviors[person_id]
        start_key = f'{behavior_type}_start'
        
        if continuous[start_key] is None:
            continuous[start_key] = timestamp
    
    def _reset_continuous_behavior(self, person_id, behavior_type):
        """Reset continuous behavior tracking"""
        start_key = f'{behavior_type}_start'
        self.continuous_behaviors[person_id][start_key] = None
    
    def _get_active_continuous_violations(self, person_id, timestamp):
        """
        Get list of continuous behaviors that exceed thresholds
        
        Args:
            person_id: Person ID
            timestamp: Current timestamp
        
        Returns:
            list: Active violations
        """
        continuous = self.continuous_behaviors[person_id]
        violations = []
        
        # Check head turned
        if continuous['head_turned_start'] is not None:
            duration = timestamp - continuous['head_turned_start']
            if duration >= BEHAVIOR_SETTINGS['suspicious_head_turn_duration']:
                violations.append({
                    'type': 'head_turned_extended',
                    'duration': duration
                })
        
        # Check looking down
        if continuous['looking_down_start'] is not None:
            duration = timestamp - continuous['looking_down_start']
            if duration >= BEHAVIOR_SETTINGS['looking_down_suspicious_duration']:
                violations.append({
                    'type': 'looking_down_extended',
                    'duration': duration
                })
        
        # Check hand below
        if continuous['hand_below_start'] is not None:
            duration = timestamp - continuous['hand_below_start']
            if duration >= BEHAVIOR_SETTINGS['suspicious_hand_duration']:
                violations.append({
                    'type': 'hand_below_extended',
                    'duration': duration
                })
        
        # Check leaning
        if continuous['leaning_start'] is not None:
            duration = timestamp - continuous['leaning_start']
            if duration >= BEHAVIOR_SETTINGS['suspicious_lean_duration']:
                violations.append({
                    'type': 'leaning_extended',
                    'duration': duration
                })
        
        return violations
    
    def detect_patterns(self, person_id, window_seconds=None):
        """
        Detect repeated behavior patterns
        
        Args:
            person_id: Person ID
            window_seconds: Time window to analyze
        
        Returns:
            dict: Detected patterns
        """
        if window_seconds is None:
            window_seconds = BEHAVIOR_SETTINGS['pattern_window_seconds']
        
        timeline = self.person_timelines[person_id]
        
        if len(timeline['timestamps']) == 0:
            return {'patterns': []}
        
        # Get events in window
        current_time = timeline['timestamps'][-1]
        cutoff_time = current_time - window_seconds
        
        recent_events = []
        for event, ts in zip(timeline['events'], timeline['timestamps']):
            if ts >= cutoff_time:
                recent_events.append(event)
        
        # Count behavior occurrences
        behavior_counts = defaultdict(int)
        
        for event in recent_events:
            for pose in event.get('suspicious_poses', []):
                behavior_counts[pose] += 1
        
        # Detect patterns
        patterns = []
        threshold = BEHAVIOR_SETTINGS['repeated_action_threshold']
        
        for behavior, count in behavior_counts.items():
            if count >= threshold:
                patterns.append({
                    'behavior': behavior,
                    'count': count,
                    'window': window_seconds,
                    'severity': 'high' if count >= threshold * 2 else 'medium'
                })
        
        return {'patterns': patterns}
    
    def get_timeline_summary(self, person_id, seconds=None):
        """
        Get summary of person's behavior timeline
        
        Args:
            person_id: Person ID
            seconds: Number of seconds to summarize (None = all)
        
        Returns:
            dict: Timeline summary
        """
        timeline = self.person_timelines[person_id]
        
        if len(timeline['timestamps']) == 0:
            return {'events': [], 'summary': {}}
        
        # Get events in range
        if seconds is not None:
            current_time = timeline['timestamps'][-1]
            cutoff_time = current_time - seconds
            
            events = []
            for event, ts in zip(timeline['events'], timeline['timestamps']):
                if ts >= cutoff_time:
                    events.append({'event': event, 'timestamp': ts})
        else:
            events = [
                {'event': event, 'timestamp': ts}
                for event, ts in zip(timeline['events'], timeline['timestamps'])
            ]
        
        # Create summary
        summary = {
            'total_prohibited_items': 0,
            'total_suspicious_poses': 0,
            'total_anomalies': 0,
            'total_continuous_violations': 0,
        }
        
        for item in events:
            event = item['event']
            summary['total_prohibited_items'] += len(event.get('prohibited_items', []))
            summary['total_suspicious_poses'] += len(event.get('suspicious_poses', []))
            summary['total_anomalies'] += len(event.get('anomalies', []))
            summary['total_continuous_violations'] += len(event.get('continuous_violations', []))
        
        return {
            'events': events,
            'summary': summary,
            'duration': seconds
        }
    
    def clear_timeline(self, person_id):
        """Clear timeline for a person"""
        if person_id in self.person_timelines:
            del self.person_timelines[person_id]
            del self.continuous_behaviors[person_id]
            del self.behavior_counters[person_id]
            logger.info(f"Timeline cleared for person {person_id}")