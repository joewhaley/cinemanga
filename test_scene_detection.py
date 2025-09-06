#!/usr/bin/env python3
"""
Test script for scene detection functionality.
Creates a test video with distinct scenes and tests the detection.
"""

import cv2
import numpy as np
import os
from scene_detection import detect_and_extract_scenes, DetectionMethod

def create_multi_scene_video(filename="multi_scene_test.webm", duration=20, fps=30):
    """
    Create a test video with multiple distinct scenes for testing detection
    """
    print(f"Creating multi-scene test video: {filename}")
    
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'VP90')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    total_frames = int(duration * fps)
    
    # Define scenes with different characteristics
    scenes = [
        {"start": 0, "end": 0.25, "color": [0, 0, 255], "pattern": "solid"},      # Red scene
        {"start": 0.25, "end": 0.4, "color": [0, 255, 0], "pattern": "gradient"}, # Green gradient
        {"start": 0.4, "end": 0.6, "color": [255, 0, 0], "pattern": "checkers"}, # Blue checkers
        {"start": 0.6, "end": 0.8, "color": [0, 255, 255], "pattern": "noise"},  # Yellow noise
        {"start": 0.8, "end": 1.0, "color": [128, 128, 128], "pattern": "solid"} # Gray scene
    ]
    
    for frame_num in range(total_frames):
        time_progress = frame_num / total_frames
        
        # Find current scene
        current_scene = None
        for scene in scenes:
            if scene["start"] <= time_progress < scene["end"]:
                current_scene = scene
                break
        
        if current_scene is None:
            current_scene = scenes[-1]  # Default to last scene
        
        # Create frame based on scene pattern
        frame = create_scene_frame(width, height, current_scene, frame_num)
        
        # Add frame info text
        time_seconds = frame_num / fps
        scene_info = f"Scene: {current_scene['pattern']} | Time: {time_seconds:.2f}s"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, scene_info, (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_num:04d}", (10, height - 20), font, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    print(f"Multi-scene test video created: {filename}")

def create_scene_frame(width, height, scene, frame_num):
    """Create a frame based on scene characteristics"""
    color = scene["color"]
    pattern = scene["pattern"]
    
    if pattern == "solid":
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        
    elif pattern == "gradient":
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            intensity = int((y / height) * 255)
            frame[y, :] = [c * intensity // 255 for c in color]
            
    elif pattern == "checkers":
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        checker_size = 50
        for y in range(0, height, checker_size):
            for x in range(0, width, checker_size):
                if ((x // checker_size) + (y // checker_size)) % 2 == 0:
                    frame[y:y+checker_size, x:x+checker_size] = color
                    
    elif pattern == "noise":
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        # Tint with scene color
        frame = (frame * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        
    return frame

def test_scene_detection_methods():
    """Test different scene detection methods"""
    test_video = "multi_scene_test.webm"
    
    # Create test video if it doesn't exist
    if not os.path.exists(test_video):
        create_multi_scene_video(test_video)
    
    methods_to_test = [
        (DetectionMethod.HISTOGRAM_DIFF, 0.3),
        (DetectionMethod.PIXEL_DIFF, 0.1),
        (DetectionMethod.EDGE_DIFF, 0.2),
        (DetectionMethod.COMBINED, 0.25)
    ]
    
    results = {}
    
    for method, threshold in methods_to_test:
        print(f"\n{'='*60}")
        print(f"Testing method: {method.value}")
        print(f"Threshold: {threshold}")
        print(f"{'='*60}")
        
        try:
            transitions, files = detect_and_extract_scenes(
                video_path=test_video,
                threshold=threshold,
                method=method,
                min_scene_length=1.0,
                output_dir=f"scene_test_{method.value}",
                extract_before=0.2,
                extract_after=0.2
            )
            
            results[method.value] = {
                'transitions': len(transitions),
                'files': len(files),
                'success': True,
                'timestamps': [t.timestamp for t in transitions]
            }
            
            print(f"\n✓ {method.value}: Found {len(transitions)} transitions")
            for i, t in enumerate(transitions, 1):
                print(f"   {i}. {t.timestamp:.2f}s (confidence: {t.confidence:.3f})")
                
        except Exception as e:
            print(f"✗ {method.value} failed: {e}")
            results[method.value] = {'success': False, 'error': str(e)}
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for method_name, result in results.items():
        if result['success']:
            print(f"{method_name:20}: {result['transitions']} transitions, {result['files']} files")
        else:
            print(f"{method_name:20}: FAILED - {result.get('error', 'Unknown error')}")
    
    return results

def test_with_existing_video():
    """Test with the previously created test video"""
    test_video = "test_video.webm"
    
    if not os.path.exists(test_video):
        print(f"Video {test_video} not found. Run test_frame_extraction.py first.")
        return
    
    print(f"\n{'='*60}")
    print(f"Testing with existing video: {test_video}")
    print(f"{'='*60}")
    
    # This video has gradual color changes, so we need a lower threshold
    transitions, files = detect_and_extract_scenes(
        video_path=test_video,
        threshold=0.15,  # Lower threshold for gradual changes
        method=DetectionMethod.HISTOGRAM_DIFF,
        min_scene_length=1.0,
        output_dir="existing_video_scenes"
    )
    
    print(f"Found {len(transitions)} transitions in the existing test video")

if __name__ == "__main__":
    print("=== Scene Detection Test Suite ===")
    
    # Test 1: Test with multiple distinct scenes
    print("\n1. Testing with multi-scene video...")
    results = test_scene_detection_methods()
    
    # Test 2: Test with existing gradual video
    print("\n2. Testing with existing test video...")
    test_with_existing_video()
    
    print("\n=== Tests Complete ===")