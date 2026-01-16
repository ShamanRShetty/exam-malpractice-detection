"""
Privacy protection - Face blurring functionality
"""
import cv2
from utils.logger import get_logger

logger = get_logger(__name__)


class FaceBlurrer:
    """Blur detected faces in frames for privacy protection"""
    
    def __init__(self, model_path=None, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Initialize face blurrer
        
        Args:
            model_path: Path to face detection model (Haar cascade or DNN)
            scale_factor: Parameter for face detection
            min_neighbors: Minimum neighbors for detection
            min_size: Minimum face size
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        
        # Try to load Haar cascade (fallback)
        if model_path is None:
            # Default OpenCV Haar cascade
            try:
                model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            except:
                # Fallback for older OpenCV versions
                import os
                cv_path = os.path.dirname(cv2.__file__)
                model_path = os.path.join(cv_path, 'data', 'haarcascade_frontalface_default.xml')
        
        try:
            self.face_cascade = cv2.CascadeClassifier(model_path)
            if self.face_cascade.empty():
                raise ValueError("Failed to load cascade classifier")
            logger.info(f"Face detector loaded: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load face detector: {e}")
            # Create a dummy cascade that won't detect anything
            self.face_cascade = None
    
    def blur_faces(self, frame, blur_strength=15):
        """
        Detect and blur faces in the frame
        
        Args:
            frame: Input BGR frame
            blur_strength: Kernel size for Gaussian blur (odd number)
        
        Returns:
            frame: Frame with blurred faces
        """
        if self.face_cascade is None:
            logger.warning("Face cascade not loaded, returning original frame")
            return frame
        
        if blur_strength % 2 == 0:
            blur_strength += 1  # Must be odd
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size
            )
            
            blurred_frame = frame.copy()
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = blurred_frame[y:y+h, x:x+w]
                # Apply Gaussian blur
                face_roi = cv2.GaussianBlur(face_roi, (blur_strength, blur_strength), 0)
                # Put back
                blurred_frame[y:y+h, x:x+w] = face_roi
            
            if len(faces) > 0:
                logger.debug(f"Blurred {len(faces)} face(s)")
            
            return blurred_frame
        except Exception as e:
            logger.error(f"Error blurring faces: {e}")
            return frame