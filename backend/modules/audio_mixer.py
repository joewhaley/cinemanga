#!/usr/bin/env python3
"""
Audio Mixer for Comic Panel Movies

This module mixes narrative audio into panel movie MP4 files using FFmpeg.
It takes the generated panel movies and overlays the corresponding narrative audio.

Requirements:
- FFmpeg must be installed and available in PATH
- Panel movies (MP4 files)
- Narrative audio files (MP3 files from ElevenLabs TTS)

Usage:
    from modules.audio_mixer import mix_audio_into_movies
    
    result = mix_audio_into_movies(
        panels=comic_panels,
        audio_files=audio_result["files"],
        output_dir="path/to/output"
    )
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

# Set up logging
logger = logging.getLogger(__name__)

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("✅ FFmpeg is available")
            return True
        else:
            logger.error("❌ FFmpeg not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.error("❌ FFmpeg not found in PATH")
        return False

def find_narrative_audio(audio_files: List[Dict[str, Any]], panel_number: int) -> Optional[str]:
    """
    Find the narrative audio file for a specific panel
    
    Args:
        audio_files: List of audio file dictionaries from storyboard_to_audio
        panel_number: Panel number to find audio for
        
    Returns:
        Path to narrative audio file or None if not found
    """
    for audio_panel in audio_files:
        if audio_panel.get("panel_number") == panel_number:
            files = audio_panel.get("files", [])
            for file_info in files:
                if file_info.get("type") == "narrative":
                    return file_info.get("path")
    return None

def mix_audio_into_movie(video_path: str, audio_path: str, output_path: str) -> Dict[str, Any]:
    """
    Mix narrative audio into a panel movie using FFmpeg
    
    Args:
        video_path: Path to the input video file
        audio_path: Path to the narrative audio file
        output_path: Path for the output video with mixed audio
        
    Returns:
        Dictionary with mixing results
    """
    try:
        # Check if input files exist
        if not os.path.exists(video_path):
            return {
                "status": "error",
                "error": f"Video file not found: {video_path}"
            }
        
        if not os.path.exists(audio_path):
            return {
                "status": "error", 
                "error": f"Audio file not found: {audio_path}"
            }
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # FFmpeg command to mix audio into video
        # -i video.mp4: input video
        # -i audio.mp3: input audio
        # -c:v copy: copy video stream without re-encoding (faster)
        # -c:a aac: encode audio as AAC
        # -map 0:v:0: map first video stream from first input
        # -map 1:a:0: map first audio stream from second input
        # -shortest: end when shortest input ends
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',  # Copy video without re-encoding
            '-c:a', 'aac',   # Encode audio as AAC
            '-map', '0:v:0', # Map video from first input
            '-map', '1:a:0', # Map audio from second input
            '-shortest',     # End when shortest stream ends
            '-y',           # Overwrite output file
            output_path
        ]
        
        logger.info(f"🎵 Mixing audio into video: {Path(video_path).name}")
        
        # Run FFmpeg command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            # Verify output file was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size = os.path.getsize(output_path)
                logger.info(f"✅ Audio mixed successfully: {Path(output_path).name} ({file_size} bytes)")
                return {
                    "status": "success",
                    "output_path": output_path,
                    "file_size": file_size
                }
            else:
                return {
                    "status": "error",
                    "error": "Output file was not created or is empty"
                }
        else:
            error_msg = f"FFmpeg failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f": {result.stderr}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg
            }
            
    except subprocess.TimeoutExpired:
        error_msg = "FFmpeg command timed out (5 minutes)"
        logger.error(f"❌ {error_msg}")
        return {
            "status": "error",
            "error": error_msg
        }
    except Exception as e:
        error_msg = f"Unexpected error during audio mixing: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return {
            "status": "error",
            "error": error_msg
        }

def mix_audio_into_movies(panels: List[Dict[str, Any]], 
                         audio_files: List[Dict[str, Any]], 
                         output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Mix narrative audio into all panel movies
    
    Args:
        panels: List of panel dictionaries with movie information
        audio_files: List of audio file dictionaries from storyboard_to_audio
        output_dir: Optional output directory (defaults to same as panels)
        
    Returns:
        Dictionary with mixing results for all panels
    """
    start_time = time.time()
    
    # Check FFmpeg availability
    if not check_ffmpeg():
        return {
            "status": "error",
            "error": "FFmpeg is not available. Please install FFmpeg to mix audio into videos."
        }
    
    mixed_panels = []
    successful_mixes = 0
    failed_mixes = 0
    
    logger.info(f"🎵 Starting audio mixing for {len(panels)} panels")
    
    for panel in panels:
        panel_number = panel.get("panel_number")
        movie_info = panel.get("movie", {})
        
        if not panel_number:
            logger.warning(f"⚠️  Panel missing panel_number, skipping")
            continue
            
        if movie_info.get("status") != "success":
            logger.warning(f"⚠️  Panel {panel_number} movie not available, skipping")
            mixed_panels.append({
                "panel_number": panel_number,
                "status": "skipped",
                "reason": "No movie available"
            })
            continue
        
        movie_path = movie_info.get("movie_path")
        if not movie_path:
            logger.warning(f"⚠️  Panel {panel_number} missing movie path, skipping")
            mixed_panels.append({
                "panel_number": panel_number,
                "status": "skipped", 
                "reason": "No movie path"
            })
            continue
        
        # Find corresponding narrative audio
        audio_path = find_narrative_audio(audio_files, panel_number)
        if not audio_path:
            logger.warning(f"⚠️  Panel {panel_number} missing narrative audio, skipping")
            mixed_panels.append({
                "panel_number": panel_number,
                "status": "skipped",
                "reason": "No narrative audio available"
            })
            continue
        
        # Determine output path
        if output_dir:
            output_base = Path(output_dir)
        else:
            output_base = Path(movie_path).parent
        
        # Create mixed video filename
        movie_name = Path(movie_path).stem
        mixed_output_path = output_base / f"{movie_name}_with_audio.mp4"
        
        # Mix audio into video
        mix_result = mix_audio_into_movie(movie_path, audio_path, str(mixed_output_path))
        
        if mix_result["status"] == "success":
            successful_mixes += 1
            mixed_panels.append({
                "panel_number": panel_number,
                "status": "success",
                "original_movie": movie_path,
                "mixed_movie": mix_result["output_path"],
                "file_size": mix_result["file_size"]
            })
        else:
            failed_mixes += 1
            mixed_panels.append({
                "panel_number": panel_number,
                "status": "error",
                "error": mix_result["error"]
            })
    
    total_time = time.time() - start_time
    
    logger.info(f"🎵 Audio mixing completed in {total_time:.2f}s")
    logger.info(f"✅ Successful: {successful_mixes}, ❌ Failed: {failed_mixes}")
    
    return {
        "status": "success" if successful_mixes > 0 else "error",
        "mixed_panels": mixed_panels,
        "successful_mixes": successful_mixes,
        "failed_mixes": failed_mixes,
        "total_panels": len(panels),
        "processing_time": total_time
    }

def create_audio_mixed_panels(panels: List[Dict[str, Any]], 
                             audio_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create a new list of panels with mixed audio movies
    
    Args:
        panels: Original panel list
        audio_files: Audio files from storyboard_to_audio
        
    Returns:
        New panel list with mixed audio movies
    """
    mix_result = mix_audio_into_movies(panels, audio_files)
    
    if mix_result["status"] != "success":
        logger.error(f"❌ Audio mixing failed: {mix_result.get('error', 'Unknown error')}")
        return panels  # Return original panels if mixing fails
    
    # Create new panel list with mixed audio
    mixed_panels = []
    mix_data = {item["panel_number"]: item for item in mix_result["mixed_panels"]}
    
    for panel in panels:
        panel_number = panel.get("panel_number")
        new_panel = panel.copy()
        
        if panel_number in mix_data:
            mix_info = mix_data[panel_number]
            if mix_info["status"] == "success":
                # Update movie info with mixed audio version
                new_panel["movie"] = {
                    "status": "success",
                    "movie_path": mix_info["mixed_movie"],
                    "file_size": mix_info["file_size"],
                    "has_audio": True
                }
                new_panel["original_movie"] = mix_info["original_movie"]
            else:
                # Keep original movie if mixing failed
                new_panel["movie_mix_error"] = mix_info.get("error", "Unknown error")
        
        mixed_panels.append(new_panel)
    
    return mixed_panels
