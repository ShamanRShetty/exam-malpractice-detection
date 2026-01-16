"""
Audio analysis for detecting whispers and talking patterns
No speech recognition - only pattern detection
"""
import numpy as np
import sounddevice as sd
from scipy import signal
from scipy.fft import fft
from collections import deque
import threading
import time
from utils.logger import get_logger
from config.settings import AUDIO_SETTINGS

logger = get_logger(__name__)


class WhisperDetector:
    """Detect talking/whispering through audio pattern analysis"""
    
    def __init__(self):
        """Initialize whisper detector"""
        if not AUDIO_SETTINGS['enable']:
            logger.info("Audio detection disabled in settings")
            return
        
        self.sample_rate = AUDIO_SETTINGS['sample_rate']
        self.chunk_size = AUDIO_SETTINGS['chunk_size']
        self.channels = AUDIO_SETTINGS['channels']
        
        # Detection thresholds
        self.energy_threshold = AUDIO_SETTINGS['energy_threshold']
        self.zcr_threshold = AUDIO_SETTINGS['zero_crossing_rate_threshold']
        self.talking_duration_threshold = AUDIO_SETTINGS['talking_duration_threshold']
        
        # Frequency bands for human speech
        self.freq_bands = AUDIO_SETTINGS['freq_bands']
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=100)  # Store last 100 chunks
        
        # Detection state
        self.is_talking = False
        self.talking_start_time = None
        self.talking_events = deque(maxlen=50)
        
        # Threading
        self.running = False
        self.thread = None
        
        logger.info("Whisper detector initialized")
    
    def start(self):
        """Start audio capture and analysis"""
        if not AUDIO_SETTINGS['enable']:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_audio, daemon=True)
        self.thread.start()
        logger.info("Audio capture started")
    
    def stop(self):
        """Stop audio capture"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Audio capture stopped")
    
    def _capture_audio(self):
        """Capture audio in background thread"""
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.chunk_size,
                callback=self._audio_callback
            ):
                while self.running:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            logger.warning(f"Audio status: {status}")
        
        # Convert to numpy array
        audio_chunk = indata[:, 0]  # Take first channel
        
        # Add to buffer
        self.audio_buffer.append(audio_chunk)
        
        # Analyze chunk
        self._analyze_chunk(audio_chunk, time.time())
    
    def _analyze_chunk(self, audio_chunk, timestamp):
        """
        Analyze audio chunk for talking patterns
        
        Args:
            audio_chunk: Audio data
            timestamp: Timestamp
        """
        # Calculate energy (RMS)
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        
        # Calculate zero-crossing rate
        zcr = self._calculate_zcr(audio_chunk)
        
        # Frequency analysis
        speech_energy = self._analyze_frequency(audio_chunk)
        
        # Determine if talking
        is_talking_now = (
            energy > self.energy_threshold and
            zcr > self.zcr_threshold and
            speech_energy > 0.3  # Significant speech frequency energy
        )
        
        # Update state
        if is_talking_now:
            if not self.is_talking:
                self.is_talking = True
                self.talking_start_time = timestamp
                logger.debug("Talking detected")
        else:
            if self.is_talking:
                talking_duration = timestamp - self.talking_start_time
                
                # Check if duration exceeds threshold
                if talking_duration >= self.talking_duration_threshold:
                    self.talking_events.append({
                        'start_time': self.talking_start_time,
                        'end_time': timestamp,
                        'duration': talking_duration
                    })
                    logger.info(f"Talking event recorded: {talking_duration:.2f}s")
                
                self.is_talking = False
                self.talking_start_time = None
    
    def _calculate_zcr(self, audio_chunk):
        """
        Calculate zero-crossing rate
        
        Args:
            audio_chunk: Audio data
        
        Returns:
            float: Zero-crossing rate
        """
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_chunk)))) / 2
        zcr = zero_crossings / len(audio_chunk)
        return zcr
    
    def _analyze_frequency(self, audio_chunk):
        """
        Analyze frequency content for speech detection
        
        Args:
            audio_chunk: Audio data
        
        Returns:
            float: Speech frequency energy ratio
        """
        # Perform FFT
        fft_data = np.abs(fft(audio_chunk))
        freqs = np.fft.fftfreq(len(audio_chunk), 1/self.sample_rate)
        
        # Only positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = fft_data[:len(fft_data)//2]
        
        # Get speech frequency band energy
        speech_band = self.freq_bands['speech']
        speech_indices = np.where(
            (positive_freqs >= speech_band[0]) &
            (positive_freqs <= speech_band[1])
        )
        
        speech_energy = np.sum(positive_fft[speech_indices])
        total_energy = np.sum(positive_fft)
        
        return speech_energy / total_energy if total_energy > 0 else 0
    
    def get_recent_events(self, seconds=60):
        """
        Get talking events from last N seconds
        
        Args:
            seconds: Time window
        
        Returns:
            list: Recent talking events
        """
        current_time = time.time()
        cutoff_time = current_time - seconds
        
        recent = [
            event for event in self.talking_events
            if event['end_time'] >= cutoff_time
        ]
        
        return recent
    
    def is_currently_talking(self):
        """Check if talking is currently detected"""
        return self.is_talking
    
    def get_talking_statistics(self):
        """Get statistics about talking events"""
        if len(self.talking_events) == 0:
            return {
                'total_events': 0,
                'total_duration': 0,
                'average_duration': 0,
                'max_duration': 0
            }
        
        durations = [event['duration'] for event in self.talking_events]
        
        return {
            'total_events': len(self.talking_events),
            'total_duration': sum(durations),
            'average_duration': np.mean(durations),
            'max_duration': max(durations)
        }


def test_audio_system():
    """Test function to check if audio system works"""
    print("Testing audio system...")
    print("Speak into your microphone for 5 seconds...")
    
    detector = WhisperDetector()
    detector.start()
    
    time.sleep(5)
    
    stats = detector.get_talking_statistics()
    print(f"\nDetected events: {stats['total_events']}")
    print(f"Total talking time: {stats['total_duration']:.2f}s")
    print(f"Average duration: {stats['average_duration']:.2f}s")
    
    detector.stop()
    print("Test complete!")


if __name__ == "__main__":
    test_audio_system()