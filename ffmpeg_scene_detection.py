import subprocess
import re
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scene_detection import SceneTransition, DetectionMethod, extract_transition_frames
import cv2

@dataclass
class FFmpegSceneTransition:
    """Data class for FFmpeg scene transition results"""
    frame_number: int
    timestamp: float
    scene_score: float

def detect_scenes_with_ffmpeg(video_path: str, 
                            threshold: float = 0.3,
                            output_log: Optional[str] = None) -> List[FFmpegSceneTransition]:
    """
    Detect scene transitions using FFmpeg's scene detection filter
    
    Args:
        video_path: Path to input video
        threshold: Scene detection threshold (0.0-1.0, lower = more sensitive)
        output_log: Optional path to save ffmpeg output log
        
    Returns:
        List of detected scene transitions
    """
    print(f"Running FFmpeg scene detection with threshold {threshold}...")
    
    # Construct ffmpeg command for scene detection with showinfo
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f"select='gt(scene,{threshold})',showinfo",
        '-f', 'null',
        '-'
    ]
    
    # Run ffmpeg command and capture output
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = result.stdout
        
        # Save log if requested
        if output_log:
            with open(output_log, 'w') as f:
                f.write(output)
                
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg command failed: {e}")
    
    # Parse the output for scene information from showinfo
    transitions = []
    
    # Look for showinfo output lines that contain timestamps
    # Format: [Parsed_showinfo_1 @ 0x...] n:123 pts:456 pts_time:12.34 duration:...
    showinfo_pattern = r'n:\s*(\d+).*?pts_time:([\d.]+)'
    
    # Get FPS for frame number calculation
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 24.0
    cap.release()
    
    for line in output.split('\n'):
        if 'showinfo' in line and 'pts_time:' in line:
            match = re.search(showinfo_pattern, line)
            if match:
                frame_num_in_output = int(match.group(1))
                timestamp = float(match.group(2))
                
                # Calculate actual frame number from timestamp
                actual_frame_num = int(timestamp * fps)
                
                # Scene score is above the threshold (we don't get the exact score from showinfo)
                # Estimate it as slightly above threshold
                scene_score = threshold + 0.1
                
                transition = FFmpegSceneTransition(
                    frame_number=actual_frame_num,
                    timestamp=timestamp,
                    scene_score=scene_score
                )
                transitions.append(transition)
    
    print(f"FFmpeg detected {len(transitions)} scene transitions")
    return transitions

def extract_ffmpeg_scene_frames(video_path: str,
                              threshold: float = 0.3,
                              output_dir: str = "ffmpeg_scenes",
                              extract_before: float = 0.5,
                              extract_after: float = 0.5) -> Tuple[List[FFmpegSceneTransition], List[str]]:
    """
    Complete pipeline: detect scenes with FFmpeg and extract frames
    
    Args:
        video_path: Path to input video
        threshold: Scene detection threshold (0.0-1.0)
        output_dir: Output directory for frames
        extract_before: Extract frame N seconds before transition
        extract_after: Extract frame N seconds after transition
        
    Returns:
        Tuple of (ffmpeg_transitions, saved_file_paths)
    """
    # Detect transitions using FFmpeg
    ffmpeg_transitions = detect_scenes_with_ffmpeg(video_path, threshold)
    
    if not ffmpeg_transitions:
        print("No scene transitions detected with FFmpeg.")
        return ffmpeg_transitions, []
    
    # Convert to standard SceneTransition format for frame extraction
    scene_transitions = []
    for ft in ffmpeg_transitions:
        transition = SceneTransition(
            frame_number=ft.frame_number,
            timestamp=ft.timestamp,
            confidence=ft.scene_score,
            method=DetectionMethod.HISTOGRAM_DIFF  # Just for compatibility
        )
        scene_transitions.append(transition)
    
    # Extract frames using existing function
    print(f"\nExtracting frames for {len(scene_transitions)} FFmpeg transitions...")
    saved_files = extract_transition_frames(
        video_path, scene_transitions, output_dir, extract_before, extract_after
    )
    
    return ffmpeg_transitions, saved_files

def compare_detection_methods(video_path: str,
                            ffmpeg_threshold: float = 0.3,
                            opencv_threshold: float = 0.15) -> dict:
    """
    Compare FFmpeg scene detection with OpenCV-based methods
    
    Args:
        video_path: Path to input video
        ffmpeg_threshold: Threshold for FFmpeg detection
        opencv_threshold: Threshold for OpenCV detection
        
    Returns:
        Dictionary with comparison results
    """
    from scene_detection import detect_and_extract_scenes, DetectionMethod
    
    print("=== Scene Detection Method Comparison ===")
    print(f"Video: {video_path}")
    
    results = {}
    
    # FFmpeg detection
    print(f"\n1. FFmpeg Scene Detection (threshold: {ffmpeg_threshold})")
    try:
        ffmpeg_transitions, ffmpeg_files = extract_ffmpeg_scene_frames(
            video_path, 
            threshold=ffmpeg_threshold,
            output_dir="comparison_ffmpeg",
            extract_before=0.3,
            extract_after=0.3
        )
        
        results['ffmpeg'] = {
            'transitions': len(ffmpeg_transitions),
            'files': len(ffmpeg_files),
            'success': True,
            'timestamps': [t.timestamp for t in ffmpeg_transitions],
            'scores': [t.scene_score for t in ffmpeg_transitions]
        }
        
        print(f"   Found {len(ffmpeg_transitions)} transitions")
        for i, t in enumerate(ffmpeg_transitions, 1):
            print(f"   {i}. {t.timestamp:.2f}s (score: {t.scene_score:.3f})")
            
    except Exception as e:
        print(f"   FFmpeg detection failed: {e}")
        results['ffmpeg'] = {'success': False, 'error': str(e)}
    
    # OpenCV detection for comparison
    print(f"\n2. OpenCV Scene Detection (threshold: {opencv_threshold})")
    try:
        opencv_transitions, opencv_files = detect_and_extract_scenes(
            video_path,
            threshold=opencv_threshold,
            method=DetectionMethod.HISTOGRAM_DIFF,
            min_scene_length=1.0,
            output_dir="comparison_opencv",
            extract_before=0.3,
            extract_after=0.3
        )
        
        results['opencv'] = {
            'transitions': len(opencv_transitions),
            'files': len(opencv_files),
            'success': True,
            'timestamps': [t.timestamp for t in opencv_transitions],
            'confidence': [t.confidence for t in opencv_transitions]
        }
        
        print(f"   Found {len(opencv_transitions)} transitions")
        
    except Exception as e:
        print(f"   OpenCV detection failed: {e}")
        results['opencv'] = {'success': False, 'error': str(e)}
    
    # Summary comparison
    print(f"\n=== Comparison Summary ===")
    if results['ffmpeg']['success'] and results['opencv']['success']:
        ffmpeg_count = results['ffmpeg']['transitions']
        opencv_count = results['opencv']['transitions']
        
        print(f"FFmpeg transitions:  {ffmpeg_count}")
        print(f"OpenCV transitions:  {opencv_count}")
        print(f"Difference:          {abs(ffmpeg_count - opencv_count)}")
        
        # Find common transitions (within 1 second tolerance)
        common_transitions = 0
        ffmpeg_times = set(results['ffmpeg']['timestamps'])
        opencv_times = results['opencv']['timestamps']
        
        for opencv_time in opencv_times:
            for ffmpeg_time in ffmpeg_times:
                if abs(opencv_time - ffmpeg_time) <= 1.0:  # 1 second tolerance
                    common_transitions += 1
                    break
        
        print(f"Common transitions:  {common_transitions} (±1s tolerance)")
        
    return results

def test_ffmpeg_thresholds(video_path: str, 
                         thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]) -> dict:
    """
    Test FFmpeg scene detection with different thresholds
    
    Args:
        video_path: Path to input video
        thresholds: List of thresholds to test
        
    Returns:
        Dictionary with results for each threshold
    """
    print(f"=== Testing FFmpeg Thresholds on {os.path.basename(video_path)} ===")
    
    results = {}
    
    for threshold in thresholds:
        print(f"\nTesting threshold {threshold}...")
        try:
            transitions = detect_scenes_with_ffmpeg(video_path, threshold)
            results[threshold] = {
                'transitions': len(transitions),
                'timestamps': [t.timestamp for t in transitions],
                'scores': [t.scene_score for t in transitions],
                'success': True
            }
            print(f"   Found {len(transitions)} transitions")
            
        except Exception as e:
            print(f"   Failed: {e}")
            results[threshold] = {'success': False, 'error': str(e)}
    
    # Summary
    print(f"\n=== Threshold Comparison ===")
    print("Threshold | Transitions")
    print("----------|------------")
    for threshold in thresholds:
        if results[threshold]['success']:
            count = results[threshold]['transitions']
            print(f"   {threshold:4.1f}   |     {count:3d}")
        else:
            print(f"   {threshold:4.1f}   |   ERROR")
    
    return results

if __name__ == "__main__":
    # Test with YoureInADream.webm
    video_file = "YoureInADream.webm"
    
    if os.path.exists(video_file):
        # Test different thresholds
        print("Testing different FFmpeg thresholds...")
        threshold_results = test_ffmpeg_thresholds(video_file, [0.1, 0.2, 0.3, 0.4, 0.5])
        
        # Compare with OpenCV
        print("\nComparing FFmpeg vs OpenCV detection...")
        comparison = compare_detection_methods(video_file, ffmpeg_threshold=0.2, opencv_threshold=0.15)
        
    else:
        print(f"Video file '{video_file}' not found.")