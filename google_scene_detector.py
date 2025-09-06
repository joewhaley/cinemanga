#!/usr/bin/env python3
"""
Google Scene Detector with AI Descriptions
Detects scene changes using Google Video Intelligence API and generates descriptions using Gemini
"""

import os
import cv2
import base64
import json
from typing import List, Dict, Tuple
from pathlib import Path

# Google Cloud imports
from google.cloud import videointelligence
import google.genai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GoogleSceneDetector:
    """Detect scene changes using Google Video Intelligence API and describe them with Gemini"""
    
    def __init__(self):
        # Configure Gemini first
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set in .env file")
            
        self.gemini_client = genai.Client(api_key=api_key)
        
        # Try to set up Google Cloud Video Intelligence client
        self.video_client = None
        self.use_google_api = False
        
        try:
            # Set up Google Cloud credentials
            if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            
            self.video_client = videointelligence.VideoIntelligenceServiceClient()
            self.use_google_api = True
            print("Using Google Video Intelligence API for scene detection")
        except Exception as e:
            print(f"Google Video Intelligence API not available: {e}")
            print("Falling back to OpenCV-based scene detection")
            self.use_google_api = False
    
    def detect_scene_changes(self, video_path: str) -> List[float]:
        """
        Detect scene changes in video using Google Video Intelligence API or OpenCV fallback
        Returns list of timestamps (in seconds) where scene changes occur
        """
        print(f"Detecting scene changes in {video_path}...")
        
        if self.use_google_api:
            return self._detect_scenes_google_api(video_path)
        else:
            return self._detect_scenes_opencv(video_path)
    
    def _detect_scenes_google_api(self, video_path: str) -> List[float]:
        """Google Video Intelligence API scene detection"""
        # Read video file
        with open(video_path, 'rb') as video_file:
            input_content = video_file.read()
        
        # Configure the request
        features = [videointelligence.Feature.SHOT_CHANGE_DETECTION]
        
        request = videointelligence.AnnotateVideoRequest(
            input_content=input_content,
            features=features
        )
        
        # Start the video annotation operation
        operation = self.video_client.annotate_video(request=request)
        
        print("Processing video for scene detection...")
        result = operation.result(timeout=300)  # 5 minute timeout
        
        # Extract shot annotations
        shot_annotations = result.annotation_results[0].shot_annotations
        
        print(f"Detected {len(shot_annotations)} shots/scenes")
        
        # Extract timestamps
        timestamps = []
        
        # Add timestamp for the first scene (start of video)
        timestamps.append(0.0)
        
        # Add timestamps for scene changes (end of each shot becomes start of next scene)
        for shot in shot_annotations[:-1]:  # Skip last shot since it ends the video
            # Convert protobuf duration to seconds
            end_time = shot.end_time_offset
            seconds = end_time.seconds
            nanos = end_time.nanos if hasattr(end_time, 'nanos') else 0
            timestamp = seconds + (nanos / 1e9)
            timestamps.append(timestamp)
        
        return timestamps
    
    def _detect_scenes_opencv(self, video_path: str, threshold: float = 0.3) -> List[float]:
        """OpenCV-based scene detection fallback"""
        import numpy as np
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        timestamps = [0.0]  # Always include start
        prev_hist = None
        frame_count = 0
        
        print("Using OpenCV histogram-based scene detection...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Calculate histogram
            hist = cv2.calcHist([frame], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            if prev_hist is not None:
                # Calculate correlation coefficient
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                
                # If correlation is low, we have a scene change
                if correlation < (1 - threshold):
                    timestamp = frame_count / fps
                    # Avoid very close timestamps
                    if not timestamps or (timestamp - timestamps[-1]) > 1.0:
                        timestamps.append(timestamp)
            
            prev_hist = hist
            frame_count += 1
            
            # Progress indicator
            if frame_count % 100 == 0:
                current_time = frame_count / fps
                print(f"Processing: {current_time:.1f}s", end='\r')
        
        cap.release()
        print(f"\nDetected {len(timestamps)} scene changes using OpenCV")
        return timestamps
    
    def extract_frames_at_timestamps(self, video_path: str, timestamps: List[float], output_dir: str = "scene_frames") -> List[str]:
        """
        Extract frames from video at specified timestamps
        Returns list of saved image file paths
        """
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        video_name = Path(video_path).stem
        image_paths = []
        
        print(f"Extracting {len(timestamps)} frames...")
        
        for i, timestamp in enumerate(timestamps):
            # Calculate frame number
            frame_number = int(timestamp * fps)
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read frame
            ret, frame = cap.read()
            if ret:
                # Create filename
                filename = f"{video_name}_scene_{i+1:03d}_t{timestamp:.2f}s.jpg"
                filepath = Path(output_dir) / filename
                
                # Save frame
                cv2.imwrite(str(filepath), frame)
                image_paths.append(str(filepath))
                
                print(f"Extracted scene {i+1}/{len(timestamps)}: {timestamp:.2f}s")
            else:
                print(f"Failed to extract frame at {timestamp:.2f}s")
        
        cap.release()
        return image_paths
    
    def describe_image(self, image_path: str) -> str:
        """
        Generate description of image using Gemini
        """
        try:
            # Load and encode image as base64
            with open(image_path, 'rb') as img_file:
                image_data = img_file.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Generate description
            prompt = """Describe this scene in detail. Focus on:
            - The main subjects and their actions
            - The setting and environment
            - The mood and atmosphere
            - Any notable visual elements, lighting, or composition
            - Keep the description concise but comprehensive (2-3 sentences)"""
            
            response = self.gemini_client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=[
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ]
            )
            
            return response.candidates[0].content.parts[0].text.strip()
            
        except Exception as e:
            print(f"Error describing image {image_path}: {e}")
            return "Description unavailable"
    
    def process_video(self, video_path: str, output_dir: str = "google_scene_output") -> Dict:
        """
        Full pipeline: detect scenes, extract frames, and generate descriptions
        """
        print(f"\n=== Processing {video_path} ===")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Step 1: Detect scene changes
        timestamps = self.detect_scene_changes(video_path)
        
        # Step 2: Extract frames
        image_paths = self.extract_frames_at_timestamps(video_path, timestamps, output_dir)
        
        # Step 3: Generate descriptions
        print(f"\nGenerating AI descriptions for {len(image_paths)} scenes...")
        scenes = []
        
        for i, (timestamp, image_path) in enumerate(zip(timestamps, image_paths)):
            print(f"Describing scene {i+1}/{len(image_paths)}...")
            
            description = self.describe_image(image_path)
            
            scene_data = {
                'scene_number': i + 1,
                'timestamp': timestamp,
                'image_path': image_path,
                'description': description
            }
            
            scenes.append(scene_data)
            
            print(f"Scene {i+1}: {timestamp:.2f}s - {description[:100]}...")
        
        # Step 4: Save results to JSON
        results = {
            'video_path': video_path,
            'total_scenes': len(scenes),
            'scenes': scenes
        }
        
        results_file = Path(output_dir) / f"{Path(video_path).stem}_scene_descriptions.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n=== Processing Complete ===")
        print(f"Detected {len(scenes)} scenes")
        print(f"Results saved to: {results_file}")
        
        return results

def main():
    """Test the scene detector"""
    detector = GoogleSceneDetector()
    
    # Test with a video file
    video_path = "InceptionOpeningScene-lowres.mkv"
    
    if not os.path.exists(video_path):
        print(f"Video file {video_path} not found!")
        return
    
    # Process the video
    results = detector.process_video(video_path)
    
    # Print summary
    print(f"\nScene Summary:")
    for scene in results['scenes'][:3]:  # Show first 3 scenes
        print(f"Scene {scene['scene_number']}: {scene['timestamp']:.2f}s")
        print(f"  {scene['description']}")
        print()

if __name__ == "__main__":
    main()