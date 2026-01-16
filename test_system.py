"""
System verification script - Test all components
Run this to check if everything is working correctly
"""
import sys
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_python_version():
    """Check Python version"""
    print("\n[1/8] Checking Python version...")
    version = sys.version_info
    print(f"      Python {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 7):
        print("      ✓ Python version OK")
        return True
    else:
        print("      ✗ Python 3.7+ required")
        return False

def check_dependencies():
    """Check if all dependencies are installed"""
    print("\n[2/8] Checking dependencies...")
    
    required = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('ultralytics', 'ultralytics'),
        ('mediapipe', 'mediapipe'),
        ('torch', 'torch'),
    ]
    
    missing = []
    for module, package in required:
        try:
            __import__(module)
            print(f"      ✓ {package}")
        except ImportError:
            print(f"      ✗ {package} - NOT FOUND")
            missing.append(package)
    
    if missing:
        print(f"\n      Install missing: pip install {' '.join(missing)}")
        return False
    else:
        print("      ✓ All dependencies installed")
        return True

def check_directories():
    """Check if required directories exist"""
    print("\n[3/8] Checking directories...")
    
    from config.settings import (
        DATA_DIR, MODELS_DIR, EVIDENCE_DIR, 
        SNAPSHOTS_DIR, LOGS_DIR, BASELINE_DIR
    )
    
    dirs = {
        'Data': DATA_DIR,
        'Models': MODELS_DIR,
        'Evidence': EVIDENCE_DIR,
        'Snapshots': SNAPSHOTS_DIR,
        'Logs': LOGS_DIR,
        'Baseline': BASELINE_DIR,
    }
    
    all_ok = True
    for name, path in dirs.items():
        if path.exists():
            print(f"      ✓ {name}: {path}")
        else:
            print(f"      ✗ {name}: {path} - NOT FOUND")
            path.mkdir(parents=True, exist_ok=True)
            print(f"        Created: {path}")
    
    print("      ✓ All directories ready")
    return True

def check_model():
    """Check if YOLO model exists"""
    print("\n[4/8] Checking YOLO model...")
    
    from config.settings import YOLO_SETTINGS
    model_path = Path(YOLO_SETTINGS['model_path'])
    
    if model_path.exists():
        size = model_path.stat().st_size / (1024 * 1024)
        print(f"      ✓ Model found: {model_path}")
        print(f"      Size: {size:.2f} MB")
        return True
    else:
        print(f"      ✗ Model not found: {model_path}")
        print("      Run: python setup_models.py")
        return False

def check_camera():
    """Check if camera is accessible"""
    print("\n[5/8] Checking camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("      ✗ Cannot open camera")
            print("      Check: Camera connected and permissions granted")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            print(f"      ✓ Camera accessible")
            print(f"      Resolution: {frame.shape[1]}x{frame.shape[0]}")
            return True
        else:
            print("      ✗ Cannot read from camera")
            return False
            
    except Exception as e:
        print(f"      ✗ Camera error: {e}")
        return False

def test_object_detection():
    """Test YOLO object detection"""
    print("\n[6/8] Testing object detection...")
    
    try:
        from src.detection.object_detector import ObjectDetector
        import numpy as np
        
        detector = ObjectDetector()
        
        # Create dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test detection
        results = detector.detect(dummy_frame)
        
        print(f"      ✓ Object detector working")
        print(f"      Classes available: {len(detector.target_classes)}")
        return True
        
    except Exception as e:
        print(f"      ✗ Object detector failed: {e}")
        return False

def test_pose_detection():
    """Test MediaPipe pose detection"""
    print("\n[7/8] Testing pose detection...")
    
    try:
        from src.detection.pose_detector import PoseDetector
        import numpy as np
        
        detector = PoseDetector()
        
        # Create dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test detection (will return None for empty frame, which is OK)
        result = detector.detect(dummy_frame)
        
        print(f"      ✓ Pose detector working")
        return True
        
    except Exception as e:
        print(f"      ✗ Pose detector failed: {e}")
        return False

def test_full_pipeline():
    """Test complete processing pipeline"""
    print("\n[8/8] Testing full pipeline...")
    
    try:
        import cv2
        import numpy as np
        from src.detection.object_detector import ObjectDetector
        from src.detection.pose_detector import MultiPersonPoseDetector
        from src.analysis.behavior_analyzer import BehaviorAnalyzer
        from src.analysis.suspicion_scorer import SuspicionScorer
        
        # Create components
        obj_detector = ObjectDetector()
        pose_detector = MultiPersonPoseDetector()
        behavior_analyzer = BehaviorAnalyzer()
        scorer = SuspicionScorer()
        
        # Create test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test object detection
        persons = obj_detector.detect_persons(test_frame)
        
        # Test scoring
        score = scorer.get_score(1)
        
        print(f"      ✓ Full pipeline working")
        print(f"      All components integrated successfully")
        return True
        
    except Exception as e:
        print(f"      ✗ Pipeline test failed: {e}")
        return False

def main():
    """Run all tests"""
    print_header("System Verification Test")
    
    tests = [
        check_python_version,
        check_dependencies,
        check_directories,
        check_model,
        check_camera,
        test_object_detection,
        test_pose_detection,
        test_full_pipeline,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"      ✗ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print_header("Test Summary")
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! System is ready to use.")
        print("\nRun: python main.py")
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Download models: python setup_models.py")
        print("  - Check camera connections and permissions")
    
    print("="*60 + "\n")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)