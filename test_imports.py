#!/usr/bin/env python3
"""
Test script to verify all required packages are working correctly
"""

def test_imports():
    """Test all required imports"""
    try:
        print("🔍 Testing imports...")
        
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
        
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
        
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO imported successfully")
        
        import math
        print("✅ Math module imported successfully")
        
        import os
        print("✅ OS module imported successfully")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_model_loading():
    """Test if the model file can be loaded"""
    try:
        print("\n🔍 Testing model loading...")
        
        import os
        model_path = "best (2).pt"
        if not os.path.exists(model_path):
            print(f"❌ Model file '{model_path}' not found!")
            return False
            
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Player Re-ID Setup\n")
    
    # Test imports
    imports_ok = test_imports()
    
    # Test model loading
    model_ok = test_model_loading()
    
    print(f"\n📋 Summary:")
    print(f"   Imports: {'✅ OK' if imports_ok else '❌ FAILED'}")
    print(f"   Model: {'✅ OK' if model_ok else '❌ FAILED'}")
    
    if imports_ok and model_ok:
        print("\n🎯 Ready to run player re-identification!")
        print("📝 Next steps:")
        print("   1. Place your input video file (15sec_input_726p.mp4) in this directory")
        print("   2. Run: python main.py")
    else:
        print("\n⚠️  Please fix the issues above before running the main script.") 