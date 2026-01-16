"""
Evidence clip generator - saves video clips around suspicious events
"""
import cv2
import time
from pathlib import Path
from datetime import datetime
from utils.logger import get_logger
from utils.video_utils import VideoWriter
from config.settings import EVIDENCE_SETTINGS, EVIDENCE_DIR

logger = get_logger(__name__)


class EvidenceClipGenerator:
    """Generate and save video clips for suspicious events"""
    
    def __init__(self, frame_buffer):
        """
        Initialize clip generator
        
        Args:
            frame_buffer: FrameBuffer instance containing recent frames
        """
        self.frame_buffer = frame_buffer
        self.pre_event_duration = EVIDENCE_SETTINGS['pre_event_duration']
        self.post_event_duration = EVIDENCE_SETTINGS['post_event_duration']
        self.codec = EVIDENCE_SETTINGS['codec']
        self.fps = EVIDENCE_SETTINGS['fps']
        
        # Ensure evidence directory exists
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Track saved events to avoid duplicates
        self.saved_events = set()
        self.last_clip_time = 0
        self.min_clip_interval = 5  # Minimum seconds between clips
        
        logger.info("Evidence clip generator initialized")
    
    def generate_clip(self, event_timestamp, event_type, person_id, metadata=None):
        """
        Generate evidence clip around an event
        
        Args:
            event_timestamp: When the event occurred
            event_type: Type of suspicious event
            person_id: ID of person involved
            metadata: Additional event metadata
        
        Returns:
            str: Path to saved clip, or None if failed
        """
        # Avoid saving clips too frequently
        current_time = time.time()
        if current_time - self.last_clip_time < self.min_clip_interval:
            logger.debug("Skipping clip - too soon after last clip")
            return None
        
        # Create unique event ID
        event_id = f"{person_id}_{event_type}_{int(event_timestamp)}"
        if event_id in self.saved_events:
            logger.debug(f"Event already saved: {event_id}")
            return None
        
        try:
            # Get frames from buffer
            start_time = event_timestamp - self.pre_event_duration
            end_time = event_timestamp + self.post_event_duration
            
            frames = self.frame_buffer.get_range(start_time, end_time)
            
            if len(frames) < 10:  # Need minimum frames
                logger.warning(f"Insufficient frames for clip: {len(frames)}")
                return None
            
            # Generate filename
            timestamp_str = datetime.fromtimestamp(event_timestamp).strftime("%Y%m%d_%H%M%S")
            filename = f"evidence_P{person_id}_{event_type}_{timestamp_str}.mp4"
            output_path = EVIDENCE_DIR / filename
            
            # Get frame size from first frame
            frame_size = (frames[0][0].shape[1], frames[0][0].shape[0])
            
            # Create video writer
            writer = VideoWriter(
                output_path,
                fps=self.fps,
                frame_size=frame_size,
                codec=self.codec
            )
            
            # Write frames
            for frame, ts in frames:
                # Add timestamp overlay
                frame_with_overlay = self._add_overlay(
                    frame.copy(),
                    ts,
                    event_type,
                    person_id,
                    metadata
                )
                writer.write(frame_with_overlay)
            
            writer.release()
            
            # Mark as saved
            self.saved_events.add(event_id)
            self.last_clip_time = current_time
            
            logger.info(f"Evidence clip saved: {output_path}")
            return str(output_path)
        
        except Exception as e:
            logger.error(f"Failed to generate clip: {e}", exc_info=True)
            return None
    
    def _add_overlay(self, frame, timestamp, event_type, person_id, metadata):
        """
        Add informational overlay to frame
        
        Args:
            frame: Input frame
            timestamp: Frame timestamp
            event_type: Event type
            person_id: Person ID
            metadata: Additional info
        
        Returns:
            frame: Frame with overlay
        """
        # Add semi-transparent overlay at top
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Add text information
        dt = datetime.fromtimestamp(timestamp)
        time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        
        texts = [
            f"Time: {time_str}",
            f"Person ID: {person_id}",
            f"Event: {event_type}"
        ]
        
        if metadata:
            if 'score' in metadata:
                texts.append(f"Score: {metadata['score']:.1f}")
        
        y = 20
        for text in texts:
            cv2.putText(
                frame, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            y += 20
        
        # Add red "EVIDENCE" watermark
        cv2.putText(
            frame, "EVIDENCE", (w - 150, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        
        return frame
    
    def save_snapshot(self, frame, event_type, person_id, metadata=None):
        """
        Save a single frame as snapshot
        
        Args:
            frame: Frame to save
            event_type: Event type
            person_id: Person ID
            metadata: Additional info
        
        Returns:
            str: Path to saved snapshot
        """
        try:
            # Generate filename
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_P{person_id}_{event_type}_{timestamp_str}.jpg"
            output_path = EVIDENCE_DIR / filename
            
            # Add overlay
            frame_with_overlay = self._add_overlay(
                frame.copy(),
                time.time(),
                event_type,
                person_id,
                metadata
            )
            
            # Save image
            cv2.imwrite(str(output_path), frame_with_overlay)
            
            logger.info(f"Snapshot saved: {output_path}")
            return str(output_path)
        
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return None
    
    def cleanup_old_clips(self, days=7):
        """
        Remove clips older than specified days
        
        Args:
            days: Age threshold in days
        """
        try:
            current_time = time.time()
            cutoff_time = current_time - (days * 24 * 3600)
            
            removed_count = 0
            
            for clip_path in EVIDENCE_DIR.glob("*.mp4"):
                if clip_path.stat().st_mtime < cutoff_time:
                    clip_path.unlink()
                    removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old clips")
        
        except Exception as e:
            logger.error(f"Failed to cleanup clips: {e}")
    
    def get_clip_count(self):
        """Get number of saved clips"""
        return len(list(EVIDENCE_DIR.glob("*.mp4")))
    
    def get_total_size(self):
        """Get total size of evidence clips in MB"""
        total_size = sum(
            f.stat().st_size for f in EVIDENCE_DIR.glob("*.mp4")
        )
        return total_size / (1024 * 1024)  # Convert to MB