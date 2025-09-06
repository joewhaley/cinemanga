#!/usr/bin/env python3
"""
Simple test script for FAL AI integration (without dotenv dependency)
"""

import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using system environment variables only")

def test_fal_availability():
    """Test if FAL AI is properly configured"""
    print("Testing FAL AI integration...")
    
    # Check if FAL_KEY is set
    fal_key = os.getenv('FAL_KEY')
    if not fal_key:
        print("❌ FAL_KEY not found in environment variables")
        print("Please set FAL_KEY environment variable or add it to your .env file")
        return False
    
    print("✅ FAL_KEY found in environment")
    
    # Test fal_client import
    try:
        import fal_client
        print("✅ fal_client imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import fal_client: {e}")
        return False
    
    # Test FALSceneDetector initialization
    try:
        from backend.modules.fal_scene_detector import FALSceneDetector
        detector = FALSceneDetector()
        print("✅ FALSceneDetector initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize FALSceneDetector: {e}")
        return False

def test_enhanced_detector():
    """Test the enhanced scene detector"""
    print("\nTesting Enhanced Scene Detector...")
    
    try:
        from enhanced_scene_detector import EnhancedSceneDetector
        detector = EnhancedSceneDetector()
        print("✅ EnhancedSceneDetector initialized successfully")
        print(f"Available methods: {detector.available_methods}")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize EnhancedSceneDetector: {e}")
        return False

def list_available_videos():
    """List available video files for testing"""
    print("\nAvailable video files:")
    video_extensions = ['*.mkv', '*.webm', '*.mp4', '*.avi', '*.mov']
    videos_found = False
    
    for ext in video_extensions:
        for file in Path(".").glob(ext):
            print(f"  📹 {file}")
            videos_found = True
    
    if not videos_found:
        print("  No video files found in current directory")
    
    return videos_found

def main():
    """Run all tests"""
    print("FAL AI Integration Test Suite")
    print("=" * 40)
    
    # Test 1: FAL AI availability
    fal_available = test_fal_availability()
    
    # Test 2: Enhanced detector
    enhanced_available = test_enhanced_detector()
    
    # Test 3: List available videos
    videos_available = list_available_videos()
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    print(f"FAL AI Available: {'✅' if fal_available else '❌'}")
    print(f"Enhanced Detector: {'✅' if enhanced_available else '❌'}")
    print(f"Video Files Found: {'✅' if videos_available else '❌'}")
    
    if fal_available and enhanced_available and videos_available:
        print("\n🎉 All tests passed! You can now run scene detection.")
        print("\nTo run scene detection:")
        print("  python enhanced_scene_detector.py")
        print("  python fal_scene_detector.py")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        
        if not fal_available:
            print("\nTo fix FAL AI issues:")
            print("1. Get an API key from https://fal.ai/")
            print("2. Set FAL_KEY environment variable: export FAL_KEY=your_api_key")
            print("3. Or add FAL_KEY=your_api_key to your .env file")
        
        if not enhanced_available:
            print("\nTo fix Enhanced Detector issues:")
            print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
            print("2. Check that Google API keys are configured (optional)")

if __name__ == "__main__":
    main()
