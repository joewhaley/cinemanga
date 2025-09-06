import cv2
import numpy as np
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class DetectionMethod(Enum):
    """Different methods for detecting scene transitions"""
    HISTOGRAM_DIFF = "histogram_diff"
    PIXEL_DIFF = "pixel_diff" 
    EDGE_DIFF = "edge_diff"
    COMBINED = "combined"

@dataclass
class SceneTransition:
    """Data class to store scene transition information"""
    frame_number: int
    timestamp: float
    confidence: float
    method: DetectionMethod

class SceneDetector:
    """Scene transition detection using various methods"""
    
    def __init__(self, 
                 threshold: float = 0.3,
                 method: DetectionMethod = DetectionMethod.HISTOGRAM_DIFF,
                 min_scene_length: float = 1.0):
        """
        Initialize scene detector
        
        Args:
            threshold: Sensitivity threshold (0-1, higher = less sensitive)
            method: Detection method to use
            min_scene_length: Minimum scene length in seconds
        """
        self.threshold = threshold
        self.method = method
        self.min_scene_length = min_scene_length
        
    def detect_transitions(self, video_path: str, 
                          sample_rate: int = 1) -> List[SceneTransition]:
        """
        Detect scene transitions in a video
        
        Args:
            video_path: Path to video file
            sample_rate: Process every Nth frame (1 = every frame)
            
        Returns:
            List of detected scene transitions
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Analyzing video: {video_path}")
        print(f"FPS: {fps}, Total frames: {total_frames}")
        print(f"Using method: {self.method.value}")
        
        transitions = []
        prev_features = None
        frame_count = 0
        
        # Skip frames based on sample rate
        min_frame_gap = int(self.min_scene_length * fps)
        last_transition_frame = -min_frame_gap
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames based on sample rate
            if frame_count % sample_rate != 0:
                frame_count += 1
                continue
            
            # Extract features based on method
            features = self._extract_features(frame, self.method)
            
            if prev_features is not None:
                # Calculate difference
                diff = self._calculate_difference(prev_features, features, self.method)
                
                # Check if it's a transition
                if (diff > self.threshold and 
                    frame_count - last_transition_frame >= min_frame_gap):
                    
                    timestamp = frame_count / fps
                    transition = SceneTransition(
                        frame_number=frame_count,
                        timestamp=timestamp,
                        confidence=min(diff, 1.0),
                        method=self.method
                    )
                    transitions.append(transition)
                    last_transition_frame = frame_count
                    
                    print(f"Transition detected at {timestamp:.2f}s "
                          f"(frame {frame_count}) - confidence: {diff:.3f}")
            
            prev_features = features
            frame_count += 1
            
            # Progress indicator
            if frame_count % (total_frames // 20) == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
        
        cap.release()
        print(f"Detection complete. Found {len(transitions)} transitions.")
        return transitions
    
    def _extract_features(self, frame: np.ndarray, method: DetectionMethod) -> np.ndarray:
        """Extract features from frame based on detection method"""
        if method == DetectionMethod.HISTOGRAM_DIFF:
            # RGB histogram
            hist_r = cv2.calcHist([frame], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([frame], [2], None, [256], [0, 256])
            return np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
            
        elif method == DetectionMethod.PIXEL_DIFF:
            # Downsampled frame for pixel comparison
            small_frame = cv2.resize(frame, (64, 64))
            return small_frame.flatten().astype(np.float32)
            
        elif method == DetectionMethod.EDGE_DIFF:
            # Edge detection features
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            # Create edge histogram
            hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
            return hist.flatten()
            
        elif method == DetectionMethod.COMBINED:
            # Combine multiple features
            hist_features = self._extract_features(frame, DetectionMethod.HISTOGRAM_DIFF)
            edge_features = self._extract_features(frame, DetectionMethod.EDGE_DIFF)
            # Normalize and combine
            hist_norm = hist_features / np.linalg.norm(hist_features)
            edge_norm = edge_features / np.linalg.norm(edge_features)
            return np.concatenate([hist_norm[:256], edge_norm])  # Use subset of histogram
    
    def _calculate_difference(self, features1: np.ndarray, features2: np.ndarray, 
                            method: DetectionMethod) -> float:
        """Calculate difference between two feature vectors"""
        if method in [DetectionMethod.HISTOGRAM_DIFF, DetectionMethod.EDGE_DIFF]:
            # Normalize histograms
            features1 = features1 / (np.sum(features1) + 1e-10)
            features2 = features2 / (np.sum(features2) + 1e-10)
            # Use Bhattacharyya distance for histograms
            return 1.0 - np.sum(np.sqrt(features1 * features2))
            
        elif method == DetectionMethod.PIXEL_DIFF:
            # Mean squared difference
            diff = np.mean((features1 - features2) ** 2)
            return diff / (255.0 ** 2)  # Normalize by max possible difference
            
        elif method == DetectionMethod.COMBINED:
            # Weighted combination of different metrics
            hist_diff = 1.0 - np.sum(np.sqrt(features1[:256] * features2[:256]))
            edge_diff = 1.0 - np.sum(np.sqrt(features1[256:] * features2[256:]))
            return 0.7 * hist_diff + 0.3 * edge_diff

def extract_transition_frames(video_path: str, 
                            transitions: List[SceneTransition],
                            output_dir: str = "scene_transitions",
                            extract_before: float = 0.0,
                            extract_after: float = 0.0) -> List[str]:
    """
    Extract frames at scene transition points
    
    Args:
        video_path: Path to video file
        transitions: List of detected transitions
        output_dir: Output directory for frames
        extract_before: Extract frame N seconds before transition
        extract_after: Extract frame N seconds after transition
        
    Returns:
        List of saved image file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    saved_files = []
    
    for i, transition in enumerate(transitions):
        # Calculate frame numbers to extract
        frames_to_extract = []
        
        if extract_before > 0:
            before_frame = max(0, transition.frame_number - int(extract_before * fps))
            frames_to_extract.append(("before", before_frame, transition.timestamp - extract_before))
        
        # Main transition frame
        frames_to_extract.append(("transition", transition.frame_number, transition.timestamp))
        
        if extract_after > 0:
            after_frame = transition.frame_number + int(extract_after * fps)
            frames_to_extract.append(("after", after_frame, transition.timestamp + extract_after))
        
        # Extract each frame
        for frame_type, frame_num, timestamp in frames_to_extract:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                confidence_str = f"{transition.confidence:.3f}"
                filename = (f"{video_name}_scene{i+1:03d}_{frame_type}_"
                           f"t{timestamp:.2f}s_conf{confidence_str}.png")
                output_path = os.path.join(output_dir, filename)
                
                cv2.imwrite(output_path, frame)
                saved_files.append(output_path)
                print(f"Extracted {frame_type} frame: {output_path}")
    
    cap.release()
    return saved_files

def detect_and_extract_scenes(video_path: str,
                            threshold: float = 0.3,
                            method: DetectionMethod = DetectionMethod.HISTOGRAM_DIFF,
                            min_scene_length: float = 1.0,
                            output_dir: str = "scene_transitions",
                            extract_before: float = 0.5,
                            extract_after: float = 0.5) -> Tuple[List[SceneTransition], List[str]]:
    """
    Complete pipeline: detect scene transitions and extract frames
    
    Args:
        video_path: Path to input video
        threshold: Detection sensitivity (0-1)
        method: Detection method to use
        min_scene_length: Minimum scene length in seconds
        output_dir: Output directory for frames
        extract_before: Extract frame N seconds before transition
        extract_after: Extract frame N seconds after transition
        
    Returns:
        Tuple of (transitions, saved_file_paths)
    """
    print("=== Scene Transition Detection and Extraction ===")
    
    # Detect transitions
    detector = SceneDetector(threshold=threshold, method=method, min_scene_length=min_scene_length)
    transitions = detector.detect_transitions(video_path)
    
    if not transitions:
        print("No scene transitions detected.")
        return transitions, []
    
    # Extract frames
    print(f"\nExtracting frames for {len(transitions)} transitions...")
    saved_files = extract_transition_frames(
        video_path, transitions, output_dir, extract_before, extract_after
    )
    
    print(f"\nComplete! Extracted {len(saved_files)} frames to '{output_dir}'")
    return transitions, saved_files

if __name__ == "__main__":
    # Example usage
    video_file = "test_video.webm"
    
    if os.path.exists(video_file):
        transitions, files = detect_and_extract_scenes(
            video_path=video_file,
            threshold=0.4,
            method=DetectionMethod.HISTOGRAM_DIFF,
            min_scene_length=2.0,
            extract_before=0.5,
            extract_after=0.5
        )
        
        print(f"\nFound transitions:")
        for t in transitions:
            print(f"  {t.timestamp:.2f}s (confidence: {t.confidence:.3f})")
    else:
        print(f"Video file '{video_file}' not found. Run test_frame_extraction.py first.")