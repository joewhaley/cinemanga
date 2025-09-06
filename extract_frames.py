import cv2
import os
from typing import List, Union

def extract_frames_at_timestamps(
    video_path: str,
    timestamps: List[Union[float, str]],
    output_dir: str = "extracted_frames",
    image_format: str = "png"
) -> List[str]:
    """
    Extract frames from a video at specific timestamps.
    
    Args:
        video_path: Path to the input video file (webm, mp4, etc.)
        timestamps: List of timestamps in seconds (float) or time format (str like "1:30.5")
        output_dir: Directory to save extracted frames
        image_format: Image format for output files (png, jpg, etc.)
    
    Returns:
        List of saved image file paths
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video info:")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    
    saved_files = []
    
    for i, timestamp in enumerate(timestamps):
        # Convert timestamp to seconds if it's in string format
        if isinstance(timestamp, str):
            timestamp_seconds = parse_time_string(timestamp)
        else:
            timestamp_seconds = float(timestamp)
        
        # Validate timestamp
        if timestamp_seconds < 0 or timestamp_seconds > duration:
            print(f"Warning: Timestamp {timestamp} is outside video duration (0-{duration:.2f}s)")
            continue
        
        # Calculate frame number
        frame_number = int(timestamp_seconds * fps)
        
        # Seek to the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read the frame
        ret, frame = cap.read()
        
        if ret:
            # Generate output filename
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = f"{video_name}_frame_{i+1:03d}_t{timestamp_seconds:.2f}s.{image_format}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save the frame
            cv2.imwrite(output_path, frame)
            saved_files.append(output_path)
            print(f"Extracted frame at {timestamp_seconds:.2f}s -> {output_path}")
        else:
            print(f"Error: Could not read frame at timestamp {timestamp}")
    
    cap.release()
    return saved_files

def parse_time_string(time_str: str) -> float:
    """
    Parse time string in format "MM:SS.mmm" or "HH:MM:SS.mmm" to seconds.
    
    Args:
        time_str: Time string (e.g., "1:30.5", "0:05", "1:23:45.2")
    
    Returns:
        Time in seconds as float
    """
    parts = time_str.split(":")
    
    if len(parts) == 2:  # MM:SS or MM:SS.mmm
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:  # HH:MM:SS or HH:MM:SS.mmm
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        raise ValueError(f"Invalid time format: {time_str}")

def extract_frames_sample():
    """
    Sample function showing how to use the frame extractor
    """
    # Example usage - replace with your actual video file
    video_file = "sample_video.webm"
    
    # Define timestamps to extract (mix of formats)
    timestamps = [
        5.0,        # 5 seconds
        "0:15.5",   # 15.5 seconds
        "1:30",     # 1 minute 30 seconds
        45.75,      # 45.75 seconds
        "2:05.25"   # 2 minutes 5.25 seconds
    ]
    
    try:
        saved_files = extract_frames_at_timestamps(
            video_path=video_file,
            timestamps=timestamps,
            output_dir="frames_output",
            image_format="png"
        )
        
        print(f"\nSuccessfully extracted {len(saved_files)} frames:")
        for file_path in saved_files:
            print(f"  {file_path}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Run the sample extraction
    extract_frames_sample()