#!/usr/bin/env python3
"""
FAL AI Video Understanding Scene Detector
Integrates with FAL AI's video understanding API for advanced scene analysis
"""

import os
import json
import time
from typing import List, Dict, Optional
from pathlib import Path
import fal_client
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not available. Using system environment variables only.")

class FALSceneDetector:
    """Scene detection and understanding using FAL AI's video understanding API"""
    
    def __init__(self):
        """Initialize the FAL AI client"""
        self.api_key = os.getenv('FAL_KEY')
        if not self.api_key:
            raise ValueError("FAL_KEY must be set in .env file")
        
        # Set the API key for fal_client
        os.environ['FAL_KEY'] = self.api_key
        print("FAL AI Video Understanding client initialized")
    
    def upload_video(self, video_path: str) -> str:
        """
        Upload video file to FAL AI and return the URL
        """
        print(f"Uploading video: {video_path}")
        
        try:
            # Upload the video file
            video_url = fal_client.upload_file(video_path)
            print(f"Video uploaded successfully: {video_url}")
            return video_url
        except Exception as e:
            print(f"Error uploading video: {e}")
            raise
    
    def analyze_video_scenes(self, video_url: str, detailed_analysis: bool = True) -> Dict:
        """
        Analyze video content using FAL AI's video understanding API
        
        Args:
            video_url: URL of the uploaded video
            detailed_analysis: Whether to request detailed analysis
            
        Returns:
            Dictionary containing the analysis results
        """
        print("Analyzing video with FAL AI...")
        
        # Define prompts for comprehensive scene analysis
        prompts = [
            "Describe the main scenes and key moments in this video. Include timestamps if possible.",
            #"What are the main visual elements, settings, and characters in this video?",
            #"Describe the narrative flow and any significant transitions or scene changes.",
            #"What is the overall mood, atmosphere, and visual style of this video?"
        ]
        
        all_results = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"Processing prompt {i}/{len(prompts)}: {prompt[:50]}...")
            
            try:
                # Submit the request
                result = fal_client.subscribe(
                    "fal-ai/video-understanding",
                    arguments={
                        "video_url": video_url,
                        "prompt": prompt,
                        "detailed_analysis": detailed_analysis
                    },
                    with_logs=True,
                    on_queue_update=self._on_queue_update
                )
                
                analysis_result = {
                    "prompt": prompt,
                    "analysis": result.get("output", ""),
                    "timestamp": time.time()
                }
                
                all_results.append(analysis_result)
                print(f"Completed analysis {i}/{len(prompts)}")
                
            except Exception as e:
                print(f"Error in analysis {i}: {e}")
                all_results.append({
                    "prompt": prompt,
                    "analysis": f"Error: {str(e)}",
                    "timestamp": time.time()
                })
        
        return {
            "video_url": video_url,
            "total_analyses": len(all_results),
            "analyses": all_results,
            "processed_at": time.time()
        }
    
    def _on_queue_update(self, update):
        """Handle queue updates during processing"""
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(f"FAL AI: {log.get('message', '')}")
    
    def extract_scene_timestamps(self, analysis_results: Dict) -> List[Dict]:
        """
        Extract scene timestamps and descriptions from analysis results
        
        Args:
            analysis_results: Results from analyze_video_scenes
            
        Returns:
            List of scene dictionaries with timestamps and descriptions
        """
        scenes = []
        
        # Look for timestamp patterns in the analysis
        for analysis in analysis_results.get("analyses", []):
            text = analysis.get("analysis", "")
            
            # Simple timestamp extraction (can be enhanced with more sophisticated parsing)
            import re
            
            # Look for patterns like "at 1:23", "around 2:45", "timestamp: 3:12"
            timestamp_patterns = [
                r'at\s+(\d+):(\d+)',
                r'around\s+(\d+):(\d+)',
                r'timestamp[:\s]+(\d+):(\d+)',
                r'(\d+):(\d+)\s+seconds?',
                r'(\d+):(\d+)\s+minutes?'
            ]
            
            for pattern in timestamp_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    minutes, seconds = int(match[0]), int(match[1])
                    timestamp = minutes * 60 + seconds
                    
                    # Extract context around the timestamp
                    context_start = max(0, text.find(match[0]) - 100)
                    context_end = min(len(text), text.find(match[0]) + 200)
                    context = text[context_start:context_end].strip()
                    
                    scenes.append({
                        "timestamp": timestamp,
                        "description": context,
                        "source_analysis": analysis.get("prompt", "")
                    })
        
        # Remove duplicates and sort by timestamp
        unique_scenes = []
        seen_timestamps = set()
        
        for scene in sorted(scenes, key=lambda x: x["timestamp"]):
            if scene["timestamp"] not in seen_timestamps:
                unique_scenes.append(scene)
                seen_timestamps.add(scene["timestamp"])
        
        return unique_scenes
    
    def process_video(self, video_path: str, output_dir: str = "fal_scene_output") -> Dict:
        """
        Complete pipeline: upload video, analyze with FAL AI, and extract scene information
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing all processing results
        """
        print(f"\n=== FAL AI Video Processing: {video_path} ===")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        try:
            # Step 1: Upload video
            video_url = self.upload_video(video_path)
            
            # Step 2: Analyze video
            analysis_results = self.analyze_video_scenes(video_url, detailed_analysis=True)
            
            # Step 3: Extract scene information
            scenes = self.extract_scene_timestamps(analysis_results)
            
            # Step 4: Compile results
            results = {
                "video_path": video_path,
                "video_url": video_url,
                "analysis_results": analysis_results,
                "extracted_scenes": scenes,
                "total_scenes": len(scenes),
                "processed_at": time.time()
            }
            
            # Step 5: Save results
            video_name = Path(video_path).stem
            results_file = Path(output_dir) / f"{video_name}_fal_analysis.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n=== FAL AI Processing Complete ===")
            print(f"Analyzed video: {video_path}")
            print(f"Extracted {len(scenes)} scenes")
            print(f"Results saved to: {results_file}")
            
            return results
            
        except Exception as e:
            print(f"Error processing video: {e}")
            raise
    
    def get_scene_summary(self, results: Dict) -> str:
        """
        Generate a human-readable summary of the scene analysis
        
        Args:
            results: Results from process_video
            
        Returns:
            Formatted summary string
        """
        video_name = Path(results["video_path"]).name
        total_scenes = results["total_scenes"]
        
        summary = f"FAL AI Video Analysis Summary\n"
        summary += f"Video: {video_name}\n"
        summary += f"Total Scenes Detected: {total_scenes}\n\n"
        
        # Add scene details
        for i, scene in enumerate(results["extracted_scenes"][:5], 1):  # Show first 5 scenes
            timestamp = scene["timestamp"]
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            summary += f"Scene {i}: {minutes}:{seconds:02d}\n"
            summary += f"  {scene['description'][:150]}...\n\n"
        
        if total_scenes > 5:
            summary += f"... and {total_scenes - 5} more scenes\n"
        
        return summary


def main():
    """Test the FAL AI scene detector"""
    try:
        detector = FALSceneDetector()
        
        # Test with a video file
        video_path = "InceptionOpeningScene-lowres.mkv"
        
        if not os.path.exists(video_path):
            print(f"Video file {video_path} not found!")
            print("Available video files:")
            for file in Path(".").glob("*.mkv"):
                print(f"  - {file}")
            for file in Path(".").glob("*.webm"):
                print(f"  - {file}")
            return
        
        # Process the video
        results = detector.process_video(video_path)
        
        # Print summary
        summary = detector.get_scene_summary(results)
        print(summary)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure FAL_KEY is set in your .env file")


if __name__ == "__main__":
    main()
