#!/usr/bin/env python3
"""
Test script for frame extraction functionality.
This creates a simple test video and extracts frames from it.
"""

import cv2
import numpy as np
import os
from extract_frames import extract_frames_at_timestamps

def create_test_video(filename="test_video.webm", duration=10, fps=30):
    """
    Create a simple test video with frame numbers displayed.
    """
    print(f"Creating test video: {filename}")
    
    # Video properties
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'VP90')  # WebM codec
    
    # Create video writer
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    total_frames = int(duration * fps)
    
    for frame_num in range(total_frames):
        # Create a frame with changing background color
        hue = int((frame_num / total_frames) * 180)
        frame = np.ones((height, width, 3), dtype=np.uint8)
        frame[:, :] = [hue, 255, 255]  # HSV color
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        
        # Add frame number text
        time_seconds = frame_num / fps
        text = f"Frame: {frame_num:04d} | Time: {time_seconds:.2f}s"
        
        # Add text to frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (50, height//2), font, 1, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
    
    out.release()
    print(f"Test video created: {filename} ({duration}s, {fps} fps)")

def test_frame_extraction():
    """
    Test the frame extraction with the created test video.
    """
    test_video = "test_video.webm"
    
    # Create test video if it doesn't exist
    if not os.path.exists(test_video):
        create_test_video(test_video)
    
    # Test timestamps
    timestamps = [
        1.0,        # 1 second
        "0:02.5",   # 2.5 seconds  
        5.0,        # 5 seconds
        "0:07.8",   # 7.8 seconds
        9.5         # 9.5 seconds
    ]
    
    print(f"\nTesting frame extraction at timestamps: {timestamps}")
    
    try:
        saved_files = extract_frames_at_timestamps(
            video_path=test_video,
            timestamps=timestamps,
            output_dir="test_frames",
            image_format="png"
        )
        
        print(f"\nTest completed successfully!")
        print(f"Extracted {len(saved_files)} frames:")
        for file_path in saved_files:
            print(f"  ✓ {file_path}")
            
        # Verify files exist
        all_exist = all(os.path.exists(f) for f in saved_files)
        print(f"\nAll files exist: {'✓ Yes' if all_exist else '✗ No'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Frame Extraction Test ===")
    test_frame_extraction()