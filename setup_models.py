"""
Setup script to download required models
Run this FIRST before starting the main application
"""
import os
from pathlib import Path
from ultralytics import YOLO

def setup_models():
    """Download and setup required models"""
    print("="*60)
    print("Setting up Exam Malpractice Detection System")
    print("="*60)
    
    # Create models directory
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "yolov8n.pt"
    
    print("\n[1/1] Downloading YOLOv8 nano model...")
    print(f"Target: {model_path}")
    
    if model_path.exists():
        print("✓ Model already exists, skipping download")
    else:
        try:
            # This will automatically download the model
            print("Downloading... (this may take a few minutes)")
            model = YOLO('yolov8n.pt')
            
            # Move to models directory
            import shutil
            source = Path.home() / '.cache' / 'ultralytics' / 'yolov8n.pt'
            if source.exists():
                shutil.copy(source, model_path)
                print(f"✓ Model saved to {model_path}")
            else:
                # If already in current directory
                if Path('yolov8n.pt').exists():
                    shutil.move('yolov8n.pt', model_path)
                    print(f"✓ Model saved to {model_path}")
        except Exception as e:
            print(f"✗ Error downloading model: {e}")
            print("\nManual download instructions:")
            print("1. Go to: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
            print(f"2. Save the file to: {model_path}")
            return False
    
    print("\n" + "="*60)
    print("Setup complete! You can now run main.py")
    print("="*60)
    return True

if __name__ == "__main__":
    setup_models()