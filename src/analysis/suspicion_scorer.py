"""
Dynamic suspicion scoring system with decay - FIXED VERSION
Replace your src/analysis/suspicion_scorer.py with this
"""
import time
from collections import defaultdict, deque
from utils.logger import get_logger
from config.settings import SUSPICION_SETTINGS

logger = get_logger(__name__)


class SuspicionScorer:
    """Manages suspicion scores for all tracked persons"""
    
    def __init__(self):
        """Initialize suspicion scorer"""
        self.scores = defaultdict(float)
        self.last_update = defaultdict(float)
        self.violation_history = defaultdict(list)
        self.alert_status = defaultdict(str)
        
        # Track recent violations to prevent spam
        self.recent_violations = defaultdict(lambda: deque(maxlen=10))
        self.violation_cooldown = defaultdict(float)  # Last time violation added
        
        # Score settings
        self.max_score = SUSPICION_SETTINGS['max_score']
        self.min_score = SUSPICION_SETTINGS['min_score']
        self.decay_rate = SUSPICION_SETTINGS['decay_rate']
        self.weights = SUSPICION_SETTINGS['weights']
        
        # Alert thresholds
        self.alert_threshold = SUSPICION_SETTINGS['alert_threshold']
        self.high_alert_threshold = SUSPICION_SETTINGS['high_alert_threshold']
        
        # ANTI-SPAM: Minimum time between same violation type (seconds)
        self.violation_cooldown_time = {
            'desk_anomaly': 2.0,        # Max once per 2 seconds
            'hand_below_desk': 3.0,     # Max once per 3 seconds
            'body_leaning': 3.0,        # Max once per 3 seconds
            'head_turn_suspicious': 2.0,
            'looking_down_extended': 2.0,
        }
        
        logger.info("Suspicion scorer initialized with anti-spam protection")
    
    def add_violation(self, person_id, violation_type, timestamp=None, metadata=None):
        """
        Add a violation and update suspicion score (with cooldown check)
        
        Args:
            person_id: Person identifier
            violation_type: Type of violation
            timestamp: When violation occurred
            metadata: Additional violation data
        """
        if timestamp is None:
            timestamp = time.time()
        
        # CHECK COOLDOWN - Prevent spam
        cooldown_key = f"{person_id}_{violation_type}"
        last_violation_time = self.violation_cooldown.get(cooldown_key, 0)
        cooldown_period = self.violation_cooldown_time.get(violation_type, 1.0)
        
        if timestamp - last_violation_time < cooldown_period:
            # Too soon! Skip this violation
            logger.debug(f"Skipping {violation_type} for person {person_id} (cooldown)")
            return
        
        # Update cooldown tracker
        self.violation_cooldown[cooldown_key] = timestamp
        
        # Apply score decay before adding new violation
        self._apply_decay(person_id, timestamp)
        
        # Get weight for violation type
        weight = self.weights.get(violation_type, 10)
        
        # Update score
        old_score = self.scores[person_id]
        self.scores[person_id] = min(
            self.scores[person_id] + weight,
            self.max_score
        )
        
        # Record violation
        violation_record = {
            'type': violation_type,
            'timestamp': timestamp,
            'weight': weight,
            'score_change': self.scores[person_id] - old_score,
            'metadata': metadata or {}
        }
        self.violation_history[person_id].append(violation_record)
        
        # Update alert status
        self._update_alert_status(person_id)
        
        # Update last update time
        self.last_update[person_id] = timestamp
        
        logger.debug(
            f"Person {person_id}: Violation '{violation_type}' added. "
            f"Score: {old_score:.1f} -> {self.scores[person_id]:.1f}"
        )
    
    def update_score(self, person_id, behaviors, timestamp=None):
        """
        Update score based on behavior analysis
        
        Args:
            person_id: Person ID
            behaviors: Dict of detected behaviors
            timestamp: Current timestamp
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Apply decay FIRST
        self._apply_decay(person_id, timestamp)
        
        # Add violations for detected behaviors
        
        # Prohibited items (always add - these are serious)
        for item in behaviors.get('prohibited_items', []):
            if item == 'phone':
                self.add_violation(person_id, 'phone_detected', timestamp)
            elif item == 'book':
                self.add_violation(person_id, 'book_detected', timestamp)
            elif item == 'paper':
                self.add_violation(person_id, 'paper_detected', timestamp)
        
        # Continuous violations (with cooldown protection)
        for violation in behaviors.get('continuous_violations', []):
            vtype = violation['type']
            
            if vtype == 'head_turned_extended':
                self.add_violation(
                    person_id, 'head_turn_suspicious', timestamp,
                    {'duration': violation['duration']}
                )
            elif vtype == 'looking_down_extended':
                self.add_violation(
                    person_id, 'looking_down_extended', timestamp,
                    {'duration': violation['duration']}
                )
            elif vtype == 'hand_below_extended':
                self.add_violation(
                    person_id, 'hand_below_desk', timestamp,
                    {'duration': violation['duration']}
                )
            elif vtype == 'leaning_extended':
                self.add_violation(
                    person_id, 'body_leaning', timestamp,
                    {'duration': violation['duration']}
                )
        
        # Anomalies (LIMIT THESE - they were causing spam!)
        anomaly_count = len(behaviors.get('anomalies', []))
        if anomaly_count > 0:
            # Only add once even if multiple anomalies detected
            self.add_violation(
                person_id, 'desk_anomaly', timestamp,
                {'count': anomaly_count}
            )
        
        self.last_update[person_id] = timestamp
    
    def add_pattern_violation(self, person_id, patterns, timestamp=None):
        """
        Add violations for detected patterns
        
        Args:
            person_id: Person ID
            patterns: List of detected patterns
            timestamp: Current timestamp
        """
        if timestamp is None:
            timestamp = time.time()
        
        for pattern in patterns:
            if pattern['behavior'] == 'head_turned':
                self.add_violation(
                    person_id, 'repeated_head_turns', timestamp,
                    {'count': pattern['count']}
                )
    
    def _apply_decay(self, person_id, current_time):
        """
        Apply score decay for good behavior
        
        Args:
            person_id: Person ID
            current_time: Current timestamp
        """
        if person_id not in self.last_update:
            self.last_update[person_id] = current_time
            return
        
        # Calculate time elapsed
        time_elapsed = current_time - self.last_update[person_id]
        
        # Calculate decay amount
        decay_amount = self.decay_rate * time_elapsed
        
        # Apply decay
        if self.scores[person_id] > self.min_score:
            old_score = self.scores[person_id]
            self.scores[person_id] = max(
                self.scores[person_id] - decay_amount,
                self.min_score
            )
            
            # Log significant decay
            if old_score - self.scores[person_id] > 5:
                logger.debug(
                    f"Person {person_id}: Score decayed {old_score:.1f} -> {self.scores[person_id]:.1f}"
                )
    
    def _update_alert_status(self, person_id):
        """
        Update alert status based on score
        
        Args:
            person_id: Person ID
        """
        score = self.scores[person_id]
        
        if score >= self.high_alert_threshold:
            self.alert_status[person_id] = 'critical'
        elif score >= self.alert_threshold:
            self.alert_status[person_id] = 'high'
        elif score >= 50:
            self.alert_status[person_id] = 'medium'
        elif score >= 30:
            self.alert_status[person_id] = 'low'
        else:
            self.alert_status[person_id] = 'normal'
    
    def get_score(self, person_id, current_time=None):
        """
        Get current score for a person
        
        Args:
            person_id: Person ID
            current_time: Optional current time for decay calculation
        
        Returns:
            float: Current suspicion score
        """
        if current_time is not None:
            self._apply_decay(person_id, current_time)
        
        return self.scores.get(person_id, 0.0)
    
    def get_alert_status(self, person_id):
        """Get alert status for person"""
        return self.alert_status.get(person_id, 'normal')
    
    def get_all_scores(self, current_time=None):
        """
        Get scores for all persons
        
        Args:
            current_time: Optional current time
        
        Returns:
            dict: person_id -> score
        """
        if current_time is not None:
            for person_id in self.scores:
                self._apply_decay(person_id, current_time)
        
        return dict(self.scores)
    
    def get_high_risk_persons(self, threshold=None):
        """
        Get list of persons with high suspicion scores
        
        Args:
            threshold: Custom threshold (uses alert_threshold if None)
        
        Returns:
            list: List of (person_id, score, alert_status)
        """
        if threshold is None:
            threshold = self.alert_threshold
        
        high_risk = []
        
        for person_id, score in self.scores.items():
            if score >= threshold:
                high_risk.append((
                    person_id,
                    score,
                    self.alert_status[person_id]
                ))
        
        # Sort by score descending
        high_risk.sort(key=lambda x: x[1], reverse=True)
        
        return high_risk
    
    def get_violation_history(self, person_id, limit=None):
        """
        Get violation history for a person
        
        Args:
            person_id: Person ID
            limit: Maximum number of violations to return
        
        Returns:
            list: Violation history
        """
        history = self.violation_history.get(person_id, [])
        
        if limit is not None:
            history = history[-limit:]
        
        return history
    
    def reset_score(self, person_id):
        """Reset score for a person"""
        self.scores[person_id] = self.min_score
        self.alert_status[person_id] = 'normal'
        self.violation_history[person_id] = []
        self.violation_cooldown.clear()
        logger.info(f"Score reset for person {person_id}")
    
    def clear_all(self):
        """Clear all scores and history"""
        self.scores.clear()
        self.last_update.clear()
        self.violation_history.clear()
        self.alert_status.clear()
        self.violation_cooldown.clear()
        logger.info("All scores cleared")
    
    def get_score_breakdown(self, person_id):
        """
        Get detailed score breakdown
        
        Args:
            person_id: Person ID
        
        Returns:
            dict: Score breakdown
        """
        history = self.violation_history.get(person_id, [])
        
        breakdown = defaultdict(lambda: {'count': 0, 'total_weight': 0})
        
        for violation in history:
            vtype = violation['type']
            breakdown[vtype]['count'] += 1
            breakdown[vtype]['total_weight'] += violation['weight']
        
        return {
            'current_score': self.scores.get(person_id, 0.0),
            'alert_status': self.alert_status.get(person_id, 'normal'),
            'violation_breakdown': dict(breakdown),
            'total_violations': len(history)
        }