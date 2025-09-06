#!/usr/bin/env python3
"""
Enhanced Scene Detector
Combines Google Video Intelligence API, FAL AI, and OpenCV for comprehensive scene detection
"""

import os
import json
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not available. Using system environment variables only.")

# Import our custom modules
from google_scene_detector import GoogleSceneDetector
from fal_scene_detector import FALSceneDetector

class EnhancedSceneDetector:
    """Enhanced scene detector combining multiple AI services and computer vision"""
    
    def __init__(self):
        """Initialize all available scene detection methods"""
        self.detectors = {}
        self.available_methods = []
        
        # Initialize Google Scene Detector
        try:
            self.detectors['google'] = GoogleSceneDetector()
            self.available_methods.append('google')
            print("✓ Google Video Intelligence API available")
        except Exception as e:
            print(f"✗ Google Video Intelligence API not available: {e}")
        
        # Initialize FAL AI Scene Detector
        try:
            self.detectors['fal'] = FALSceneDetector()
            self.available_methods.append('fal')
            print("✓ FAL AI Video Understanding available")
        except Exception as e:
            print(f"✗ FAL AI not available: {e}")
        
        if not self.available_methods:
            raise ValueError("No scene detection methods available. Please check your API keys and dependencies.")
        
        print(f"Available detection methods: {', '.join(self.available_methods)}")
    
    def detect_scenes_google(self, video_path: str) -> Dict:
        """Detect scenes using Google Video Intelligence API"""
        if 'google' not in self.detectors:
            raise ValueError("Google Video Intelligence API not available")
        
        print("\n=== Google Video Intelligence Analysis ===")
        return self.detectors['google'].process_video(video_path, "google_scene_output")
    
    def detect_scenes_fal(self, video_path: str) -> Dict:
        """Detect scenes using FAL AI Video Understanding"""
        if 'fal' not in self.detectors:
            raise ValueError("FAL AI not available")
        
        print("\n=== FAL AI Video Understanding Analysis ===")
        return self.detectors['fal'].process_video(video_path, "fal_scene_output")
    
    def detect_scenes_combined(self, video_path: str, methods: List[str] = None) -> Dict:
        """
        Combine results from multiple scene detection methods
        
        Args:
            video_path: Path to the video file
            methods: List of methods to use ('google', 'fal'). If None, uses all available.
            
        Returns:
            Combined analysis results
        """
        if methods is None:
            methods = self.available_methods
        
        print(f"\n=== Enhanced Scene Detection: {video_path} ===")
        print(f"Using methods: {', '.join(methods)}")
        
        results = {
            "video_path": video_path,
            "methods_used": methods,
            "individual_results": {},
            "combined_scenes": [],
            "processing_times": {},
            "processed_at": time.time()
        }
        
        # Run each method
        for method in methods:
            if method not in self.detectors:
                print(f"Warning: Method '{method}' not available, skipping...")
                continue
            
            print(f"\n--- Running {method.upper()} analysis ---")
            start_time = time.time()
            
            try:
                if method == 'google':
                    method_result = self.detect_scenes_google(video_path)
                elif method == 'fal':
                    method_result = self.detect_scenes_fal(video_path)
                else:
                    print(f"Unknown method: {method}")
                    continue
                
                processing_time = time.time() - start_time
                results["processing_times"][method] = processing_time
                results["individual_results"][method] = method_result
                
                print(f"✓ {method.upper()} completed in {processing_time:.2f}s")
                
            except Exception as e:
                print(f"✗ {method.upper()} failed: {e}")
                results["individual_results"][method] = {"error": str(e)}
        
        # Combine scene information
        results["combined_scenes"] = self._combine_scene_results(results["individual_results"])
        
        # Save combined results
        self._save_combined_results(results)
        
        return results
    
    def _combine_scene_results(self, individual_results: Dict) -> List[Dict]:
        """
        Combine scene results from different methods
        
        Args:
            individual_results: Results from individual detection methods
            
        Returns:
            Combined list of scenes with metadata
        """
        combined_scenes = []
        
        # Process Google results
        if 'google' in individual_results and 'error' not in individual_results['google']:
            google_scenes = individual_results['google'].get('scenes', [])
            for scene in google_scenes:
                combined_scenes.append({
                    "timestamp": scene.get('timestamp', 0),
                    "description": scene.get('description', ''),
                    "image_path": scene.get('image_path', ''),
                    "source_method": "google",
                    "confidence": "high"  # Google API provides high confidence
                })
        
        # Process FAL AI results
        if 'fal' in individual_results and 'error' not in individual_results['fal']:
            fal_scenes = individual_results['fal'].get('extracted_scenes', [])
            for scene in fal_scenes:
                combined_scenes.append({
                    "timestamp": scene.get('timestamp', 0),
                    "description": scene.get('description', ''),
                    "image_path": "",  # FAL doesn't extract frames
                    "source_method": "fal",
                    "confidence": "medium"  # FAL provides contextual analysis
                })
        
        # Sort by timestamp and remove duplicates
        combined_scenes.sort(key=lambda x: x['timestamp'])
        
        # Remove scenes that are too close together (within 2 seconds)
        filtered_scenes = []
        for scene in combined_scenes:
            if not filtered_scenes or (scene['timestamp'] - filtered_scenes[-1]['timestamp']) > 2.0:
                filtered_scenes.append(scene)
        
        return filtered_scenes
    
    def _save_combined_results(self, results: Dict):
        """Save combined results to JSON file"""
        video_name = Path(results["video_path"]).stem
        output_dir = Path("enhanced_scene_output")
        output_dir.mkdir(exist_ok=True)
        
        results_file = output_dir / f"{video_name}_enhanced_analysis.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nCombined results saved to: {results_file}")
    
    def generate_comprehensive_report(self, results: Dict) -> str:
        """
        Generate a comprehensive report of the scene analysis
        
        Args:
            results: Results from detect_scenes_combined
            
        Returns:
            Formatted report string
        """
        video_name = Path(results["video_path"]).name
        methods_used = results["methods_used"]
        total_scenes = len(results["combined_scenes"])
        
        report = f"""
ENHANCED SCENE DETECTION REPORT
{'=' * 50}
Video: {video_name}
Methods Used: {', '.join(methods_used)}
Total Scenes Detected: {total_scenes}
Processing Time: {sum(results['processing_times'].values()):.2f}s

METHOD PERFORMANCE:
"""
        
        for method, time_taken in results["processing_times"].items():
            report += f"  {method.upper()}: {time_taken:.2f}s\n"
        
        report += f"\nSCENE BREAKDOWN:\n"
        
        for i, scene in enumerate(results["combined_scenes"][:10], 1):  # Show first 10 scenes
            timestamp = scene["timestamp"]
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            method = scene["source_method"].upper()
            confidence = scene["confidence"]
            
            report += f"\nScene {i}: {minutes}:{seconds:02d} ({method}, {confidence} confidence)\n"
            report += f"  {scene['description'][:200]}...\n"
        
        if total_scenes > 10:
            report += f"\n... and {total_scenes - 10} more scenes\n"
        
        # Add method-specific insights
        report += f"\nMETHOD INSIGHTS:\n"
        
        if 'google' in results["individual_results"] and 'error' not in results["individual_results"]['google']:
            google_scenes = len(results["individual_results"]['google'].get('scenes', []))
            report += f"  Google API detected {google_scenes} scenes with frame extraction\n"
        
        if 'fal' in results["individual_results"] and 'error' not in results["individual_results"]['fal']:
            fal_analyses = results["individual_results"]['fal'].get('analysis_results', {}).get('total_analyses', 0)
            report += f"  FAL AI performed {fal_analyses} detailed content analyses\n"
        
        return report
    
    def compare_methods(self, video_path: str) -> Dict:
        """
        Compare different scene detection methods side by side
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Comparison results
        """
        print(f"\n=== Method Comparison: {video_path} ===")
        
        comparison = {
            "video_path": video_path,
            "methods": {},
            "comparison_metrics": {},
            "processed_at": time.time()
        }
        
        for method in self.available_methods:
            print(f"\nTesting {method.upper()}...")
            start_time = time.time()
            
            try:
                if method == 'google':
                    result = self.detect_scenes_google(video_path)
                    scene_count = len(result.get('scenes', []))
                elif method == 'fal':
                    result = self.detect_scenes_fal(video_path)
                    scene_count = len(result.get('extracted_scenes', []))
                else:
                    continue
                
                processing_time = time.time() - start_time
                
                comparison["methods"][method] = {
                    "success": True,
                    "scene_count": scene_count,
                    "processing_time": processing_time,
                    "result": result
                }
                
                print(f"✓ {method.upper()}: {scene_count} scenes in {processing_time:.2f}s")
                
            except Exception as e:
                comparison["methods"][method] = {
                    "success": False,
                    "error": str(e),
                    "processing_time": time.time() - start_time
                }
                print(f"✗ {method.upper()}: {e}")
        
        # Calculate comparison metrics
        successful_methods = [m for m, data in comparison["methods"].items() if data["success"]]
        if len(successful_methods) > 1:
            scene_counts = [comparison["methods"][m]["scene_count"] for m in successful_methods]
            comparison["comparison_metrics"] = {
                "scene_count_range": f"{min(scene_counts)} - {max(scene_counts)}",
                "average_scenes": sum(scene_counts) / len(scene_counts),
                "fastest_method": min(successful_methods, key=lambda m: comparison["methods"][m]["processing_time"])
            }
        
        return comparison


def main():
    """Test the enhanced scene detector"""
    try:
        detector = EnhancedSceneDetector()
        
        # Test with a video file
        video_path = "InceptionOpeningScene-lowres.mkv"
        
        if not os.path.exists(video_path):
            print(f"Video file {video_path} not found!")
            print("Available video files:")
            for ext in ['*.mkv', '*.webm', '*.mp4']:
                for file in Path(".").glob(ext):
                    print(f"  - {file}")
            return
        
        print("Choose an option:")
        print("1. Run combined analysis (all methods)")
        print("2. Compare methods side by side")
        print("3. Run Google analysis only")
        print("4. Run FAL AI analysis only")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            results = detector.detect_scenes_combined(video_path)
            report = detector.generate_comprehensive_report(results)
            print(report)
            
        elif choice == "2":
            comparison = detector.compare_methods(video_path)
            print(f"\nMethod Comparison Results:")
            for method, data in comparison["methods"].items():
                if data["success"]:
                    print(f"{method.upper()}: {data['scene_count']} scenes in {data['processing_time']:.2f}s")
                else:
                    print(f"{method.upper()}: Failed - {data['error']}")
            
        elif choice == "3":
            if 'google' in detector.available_methods:
                results = detector.detect_scenes_google(video_path)
                print(f"Google detected {len(results.get('scenes', []))} scenes")
            else:
                print("Google Video Intelligence API not available")
                
        elif choice == "4":
            if 'fal' in detector.available_methods:
                results = detector.detect_scenes_fal(video_path)
                print(f"FAL AI detected {len(results.get('extracted_scenes', []))} scenes")
            else:
                print("FAL AI not available")
        else:
            print("Invalid choice")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
