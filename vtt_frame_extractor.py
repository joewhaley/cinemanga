import re
import os
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import timedelta

@dataclass
class VTTCaption:
    """Data class to store VTT caption information"""
    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds
    text: str          # Caption text
    index: int         # Caption index/number

class SubtitleParser:
    """Universal parser for subtitle files (VTT and SRT formats)"""
    
    def __init__(self):
        self.captions = []
    
    def parse_time(self, time_str: str) -> float:
        """
        Parse timestamp string to seconds
        Supports VTT and SRT formats: HH:MM:SS.mmm, HH:MM:SS,mmm, MM:SS.mmm, SS.mmm
        """
        time_str = time_str.strip()
        
        # Remove any leading/trailing whitespace and handle different formats
        if time_str.count(':') == 2:
            # HH:MM:SS.mmm or HH:MM:SS,mmm format (VTT uses ., SRT uses ,)
            match = re.match(r'(\d+):(\d+):(\d+)[.,](\d+)', time_str)
            if match:
                hours = int(match.group(1))
                minutes = int(match.group(2))
                seconds = int(match.group(3))
                # Handle 3-digit milliseconds
                milliseconds = int(match.group(4))
                if len(match.group(4)) == 3:
                    milliseconds = milliseconds
                else:
                    milliseconds = milliseconds * 10 if len(match.group(4)) == 2 else milliseconds * 100
                return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
        elif time_str.count(':') == 1:
            # MM:SS.mmm format  
            match = re.match(r'(\d+):(\d+)\.(\d+)', time_str)
            if match:
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                milliseconds = int(match.group(3))
                return minutes * 60 + seconds + milliseconds / 1000
        else:
            # SS.mmm format
            match = re.match(r'(\d+)\.(\d+)', time_str)
            if match:
                seconds = int(match.group(1))
                milliseconds = int(match.group(2))
                return seconds + milliseconds / 1000
        
        raise ValueError(f"Invalid time format: {time_str}")
    
    def detect_subtitle_format(self, file_path: str) -> str:
        """
        Detect subtitle file format (VTT or SRT)
        
        Args:
            file_path: Path to subtitle file
            
        Returns:
            Format string: 'vtt' or 'srt'
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            first_lines = ''.join(f.readlines()[:5]).lower()
            
        if 'webvtt' in first_lines:
            return 'vtt'
        elif re.search(r'^\d+$', first_lines.strip().split('\n')[0]):
            return 'srt'
        else:
            # Try to guess from file extension
            if file_path.lower().endswith('.vtt'):
                return 'vtt'
            elif file_path.lower().endswith('.srt'):
                return 'srt'
            else:
                raise ValueError(f"Could not detect subtitle format for: {file_path}")
    
    def parse_srt_file(self, srt_path: str) -> List[VTTCaption]:
        """
        Parse an SRT file and extract all captions with timestamps
        
        Args:
            srt_path: Path to the SRT file
            
        Returns:
            List of VTTCaption objects
        """
        print(f"Parsing SRT file: {srt_path}")
        
        if not os.path.exists(srt_path):
            raise FileNotFoundError(f"SRT file not found: {srt_path}")
        
        captions = []
        
        with open(srt_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        
        # Split by double newline to separate subtitle blocks
        subtitle_blocks = re.split(r'\n\s*\n', content)
        
        for block in subtitle_blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue  # Skip incomplete blocks
            
            # First line should be the subtitle number
            try:
                subtitle_number = int(lines[0])
            except ValueError:
                continue  # Skip if first line is not a number
            
            # Second line should contain timestamps
            timestamp_line = lines[1]
            timestamp_match = re.match(r'(.+?)\s+-->\s+(.+)', timestamp_line)
            
            if timestamp_match:
                start_time_str = timestamp_match.group(1)
                end_time_str = timestamp_match.group(2)
                
                try:
                    start_time = self.parse_time(start_time_str)
                    end_time = self.parse_time(end_time_str)
                    
                    # Remaining lines are the caption text
                    caption_text_lines = lines[2:]
                    caption_text = ' '.join(line.strip() for line in caption_text_lines if line.strip())
                    
                    # Clean up HTML tags and formatting
                    caption_text = re.sub(r'<[^>]+>', '', caption_text)
                    caption_text = caption_text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                    
                    if caption_text:  # Only add if there's actual text
                        caption = VTTCaption(
                            start_time=start_time,
                            end_time=end_time,
                            text=caption_text,
                            index=subtitle_number - 1  # Convert to 0-based index
                        )
                        captions.append(caption)
                
                except ValueError as e:
                    print(f"Warning: Could not parse timestamp in block {subtitle_number}: {timestamp_line}")
                    continue
        
        print(f"Parsed {len(captions)} captions from SRT file")
        return captions
    
    def parse_vtt_file(self, vtt_path: str) -> List[VTTCaption]:
        """
        Parse a VTT file and extract all captions with timestamps
        
        Args:
            vtt_path: Path to the VTT file
            
        Returns:
            List of VTTCaption objects
        """
        print(f"Parsing VTT file: {vtt_path}")
        
        if not os.path.exists(vtt_path):
            raise FileNotFoundError(f"VTT file not found: {vtt_path}")
        
        captions = []
        current_caption = None
        
        with open(vtt_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        i = 0
        caption_index = 0
        
        # Skip the WEBVTT header
        while i < len(lines) and not lines[i].strip().startswith('WEBVTT'):
            i += 1
        i += 1  # Skip the WEBVTT line itself
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Check if this line contains a timestamp (indicates start of new caption)
            # Handle both simple format and format with positioning/alignment
            timestamp_match = re.match(r'(.+?)\s+-->\s+([^\s]+)', line)
            if timestamp_match:
                start_time_str = timestamp_match.group(1)
                end_time_str = timestamp_match.group(2)
                
                try:
                    start_time = self.parse_time(start_time_str)
                    end_time = self.parse_time(end_time_str)
                    
                    # Collect caption text from following lines
                    caption_text_lines = []
                    i += 1
                    
                    # Skip empty lines after timestamp
                    while i < len(lines) and not lines[i].strip():
                        i += 1
                    
                    # Collect non-empty caption lines
                    while i < len(lines) and lines[i].strip():
                        text = lines[i].strip()
                        if text:  # Only add non-empty lines
                            caption_text_lines.append(text)
                        i += 1
                    
                    caption_text = ' '.join(caption_text_lines)
                    
                    # Clean up HTML tags and formatting
                    caption_text = re.sub(r'<[^>]+>', '', caption_text)
                    caption_text = caption_text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                    
                    if caption_text:  # Only add if there's actual text
                        caption = VTTCaption(
                            start_time=start_time,
                            end_time=end_time,
                            text=caption_text,
                            index=caption_index
                        )
                        captions.append(caption)
                        caption_index += 1
                
                except ValueError as e:
                    print(f"Warning: Could not parse timestamp on line {i}: {line}")
                    i += 1
                    continue
            else:
                i += 1
        
        print(f"Parsed {len(captions)} captions from VTT file")
        return captions
    
    def parse_subtitle_file(self, subtitle_path: str) -> List[VTTCaption]:
        """
        Universal parser that automatically detects and parses VTT or SRT files
        
        Args:
            subtitle_path: Path to subtitle file (.vtt or .srt)
            
        Returns:
            List of VTTCaption objects
        """
        format_type = self.detect_subtitle_format(subtitle_path)
        
        if format_type == 'vtt':
            return self.parse_vtt_file(subtitle_path)
        elif format_type == 'srt':
            return self.parse_srt_file(subtitle_path)
        else:
            raise ValueError(f"Unsupported subtitle format: {format_type}")
    
    def get_caption_at_time(self, captions: List[VTTCaption], timestamp: float) -> Optional[VTTCaption]:
        """
        Get the caption that is active at a specific timestamp
        
        Args:
            captions: List of VTT captions
            timestamp: Time in seconds
            
        Returns:
            VTTCaption if found, None otherwise
        """
        for caption in captions:
            if caption.start_time <= timestamp <= caption.end_time:
                return caption
        return None

def write_caption_text_files(captions: List[VTTCaption], 
                            saved_files: List[str],
                            text_files: dict,
                            frame_type: str,
                            video_name: str) -> None:
    """
    Write parallel text files containing caption information
    
    Args:
        captions: List of VTT captions
        saved_files: List of saved image file paths
        text_files: Dictionary of text file paths
        frame_type: Type of frame extraction used
        video_name: Name of the video
    """
    # Write captions file (one caption per line)
    with open(text_files['captions'], 'w', encoding='utf-8') as f:
        f.write(f"# Captions extracted from {video_name} at {frame_type} timestamps\n")
        f.write(f"# Format: Caption text\n")
        f.write(f"# Total captions: {len(captions)}\n\n")
        
        for caption in captions:
            f.write(f"{caption.text}\n")
    
    # Write timestamps file (structured data)
    with open(text_files['timestamps'], 'w', encoding='utf-8') as f:
        f.write(f"# Timestamps for {video_name} at {frame_type} positions\n")
        f.write(f"# Format: Index | Start Time | End Time | Duration | Extracted Time | Caption\n")
        f.write(f"# Times in seconds\n\n")
        
        for i, caption in enumerate(captions):
            if frame_type == "start":
                extracted_time = caption.start_time
            elif frame_type == "end":
                extracted_time = caption.end_time
            else:  # middle
                extracted_time = (caption.start_time + caption.end_time) / 2
            
            duration = caption.end_time - caption.start_time
            f.write(f"{caption.index+1:3d} | {caption.start_time:7.2f} | {caption.end_time:7.2f} | {duration:6.2f} | {extracted_time:7.2f} | {caption.text}\n")
    
    # Write metadata file (detailed information)
    with open(text_files['metadata'], 'w', encoding='utf-8') as f:
        f.write(f"# Metadata for {video_name} extraction\n")
        f.write(f"# Generated by VTT Frame Extractor\n\n")
        f.write(f"Video: {video_name}\n")
        f.write(f"Frame extraction type: {frame_type}\n")
        f.write(f"Total captions: {len(captions)}\n")
        f.write(f"Total frames extracted: {len(saved_files)}\n")
        f.write(f"First caption time: {captions[0].start_time:.2f}s\n")
        f.write(f"Last caption time: {captions[-1].end_time:.2f}s\n")
        f.write(f"Total duration: {captions[-1].end_time - captions[0].start_time:.2f}s\n")
        f.write(f"Average caption duration: {sum(c.end_time - c.start_time for c in captions) / len(captions):.2f}s\n\n")
        
        f.write("# Image Files and Corresponding Captions:\n")
        for i, (caption, image_file) in enumerate(zip(captions, saved_files)):
            if frame_type == "start":
                extracted_time = caption.start_time
            elif frame_type == "end":
                extracted_time = caption.end_time
            else:  # middle
                extracted_time = (caption.start_time + caption.end_time) / 2
            
            image_name = os.path.basename(image_file)
            f.write(f"{i+1:3d}. {image_name}\n")
            f.write(f"     Time: {extracted_time:.2f}s ({caption.start_time:.2f}s - {caption.end_time:.2f}s)\n")
            f.write(f"     Text: {caption.text}\n\n")
    
    print(f"Text files created:")
    print(f"  Captions: {text_files['captions']}")
    print(f"  Timestamps: {text_files['timestamps']}")
    print(f"  Metadata: {text_files['metadata']}")

def extract_frames_from_subtitles(video_path: str, 
                                 subtitle_path: str,
                                 output_dir: str = "subtitle_frames",
                                 frame_type: str = "start",  # "start", "middle", "end"
                                 max_captions: Optional[int] = None,
                                 save_text: bool = True) -> Tuple[List[VTTCaption], List[str]]:
    """
    Extract video frames at subtitle timestamps (supports VTT and SRT)
    
    Args:
        video_path: Path to video file
        subtitle_path: Path to subtitle file (.vtt or .srt)
        output_dir: Directory to save extracted frames
        frame_type: Which timestamp to use ("start", "middle", "end")
        max_captions: Maximum number of captions to process (None for all)
        save_text: Whether to save parallel text files with captions
        
    Returns:
        Tuple of (captions, saved_file_paths)
    """
    # Parse subtitle file (auto-detects VTT or SRT)
    parser = SubtitleParser()
    captions = parser.parse_subtitle_file(subtitle_path)
    
    if max_captions:
        captions = captions[:max_captions]
    
    if not captions:
        print("No captions found in VTT file")
        return [], []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    print(f"Extracting frames from {video_path} (FPS: {fps:.2f})")
    print(f"Processing {len(captions)} captions...")
    
    saved_files = []
    
    # Prepare text output files if requested
    text_files = {}
    if save_text:
        text_files = {
            'captions': os.path.join(output_dir, f"{video_name}_captions_{frame_type}.txt"),
            'timestamps': os.path.join(output_dir, f"{video_name}_timestamps_{frame_type}.txt"),
            'metadata': os.path.join(output_dir, f"{video_name}_metadata_{frame_type}.txt")
        }
    
    for caption in captions:
        # Determine which timestamp to use
        if frame_type == "start":
            timestamp = caption.start_time
        elif frame_type == "end":
            timestamp = caption.end_time
        else:  # middle
            timestamp = (caption.start_time + caption.end_time) / 2
        
        # Calculate frame number
        frame_number = int(timestamp * fps)
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            # Clean caption text for filename (remove special characters)
            clean_text = re.sub(r'[^\w\s-]', '', caption.text)
            clean_text = re.sub(r'\s+', '_', clean_text.strip())
            
            # Truncate text if too long
            if len(clean_text) > 50:
                clean_text = clean_text[:47] + "..."
            
            # Create filename
            filename = f"{video_name}_caption{caption.index+1:03d}_t{timestamp:.2f}s_{frame_type}_{clean_text}.png"
            output_path = os.path.join(output_dir, filename)
            
            # Save frame
            cv2.imwrite(output_path, frame)
            saved_files.append(output_path)
            
            print(f"Extracted frame {caption.index+1}/{len(captions)}: {timestamp:.2f}s - '{caption.text[:50]}{'...' if len(caption.text) > 50 else ''}'")
        else:
            print(f"Warning: Could not extract frame at {timestamp:.2f}s for caption: '{caption.text[:50]}'")
    
    cap.release()
    
    # Write text files if requested
    if save_text and saved_files:
        write_caption_text_files(captions, saved_files, text_files, frame_type, video_name)
    
    print(f"\nExtraction complete! Saved {len(saved_files)} frames to '{output_dir}'")
    return captions, saved_files

def extract_frames_all_positions(video_path: str,
                                vtt_path: str,
                                output_dir: str = "vtt_frames_all",
                                max_captions: Optional[int] = None,
                                save_text: bool = True) -> dict:
    """
    Extract frames at start, middle, and end of each caption
    
    Args:
        video_path: Path to video file
        vtt_path: Path to VTT subtitle file
        output_dir: Base directory for output
        max_captions: Maximum number of captions to process
        save_text: Whether to save parallel text files
        
    Returns:
        Dictionary with results for each position
    """
    results = {}
    
    for position in ["start", "middle", "end"]:
        position_dir = os.path.join(output_dir, position)
        print(f"\n=== Extracting {position} frames ===")
        
        captions, files = extract_frames_from_subtitles(
            video_path, vtt_path, position_dir, position, max_captions, save_text
        )
        
        results[position] = {
            'captions': captions,
            'files': files,
            'count': len(files)
        }
    
    return results

def analyze_subtitle_file(subtitle_path: str) -> dict:
    """
    Analyze a subtitle file (VTT or SRT) and return statistics
    
    Args:
        subtitle_path: Path to subtitle file (.vtt or .srt)
        
    Returns:
        Dictionary with analysis results
    """
    parser = SubtitleParser()
    captions = parser.parse_subtitle_file(subtitle_path)
    
    if not captions:
        return {'error': 'No captions found'}
    
    total_duration = captions[-1].end_time - captions[0].start_time
    avg_caption_duration = sum(c.end_time - c.start_time for c in captions) / len(captions)
    total_text_length = sum(len(c.text) for c in captions)
    
    analysis = {
        'total_captions': len(captions),
        'first_caption_time': captions[0].start_time,
        'last_caption_time': captions[-1].end_time,
        'total_duration': total_duration,
        'average_caption_duration': avg_caption_duration,
        'total_text_length': total_text_length,
        'average_text_length': total_text_length / len(captions),
        'sample_captions': captions[:3]  # First 3 captions as samples
    }
    
    return analysis

# Backward compatibility functions
def extract_frames_from_vtt(video_path: str, vtt_path: str, **kwargs) -> Tuple[List[VTTCaption], List[str]]:
    """Backward compatibility wrapper for VTT files"""
    return extract_frames_from_subtitles(video_path, vtt_path, **kwargs)

def analyze_vtt_file(vtt_path: str) -> dict:
    """Backward compatibility wrapper for VTT analysis"""
    return analyze_subtitle_file(vtt_path)

# New SRT-specific functions for convenience
def extract_frames_from_srt(video_path: str, srt_path: str, **kwargs) -> Tuple[List[VTTCaption], List[str]]:
    """Extract frames from SRT subtitles"""
    return extract_frames_from_subtitles(video_path, srt_path, **kwargs)

def analyze_srt_file(srt_path: str) -> dict:
    """Analyze SRT subtitle file"""
    return analyze_subtitle_file(srt_path)

if __name__ == "__main__":
    # Test with InceptionOpeningScene files
    video_file = "InceptionOpeningScene.mkv"
    vtt_files = ["InceptionOpeningScene.en.vtt", "InceptionOpeningScene.en-orig.vtt"]
    
    for vtt_file in vtt_files:
        if os.path.exists(video_file) and os.path.exists(vtt_file):
            print(f"\n{'='*60}")
            print(f"Processing: {vtt_file}")
            print(f"{'='*60}")
            
            # Analyze subtitle file first
            analysis = analyze_subtitle_file(vtt_file)
            if 'error' in analysis:
                print(f"VTT Analysis Error: {analysis['error']}")
                continue
            
            print(f"\nVTT Analysis:")
            print(f"  Total captions: {analysis['total_captions']}")
            print(f"  Duration: {analysis['total_duration']:.1f} seconds")
            print(f"  Avg caption length: {analysis['average_caption_duration']:.2f}s")
            print(f"  First caption: {analysis['first_caption_time']:.1f}s")
            print(f"  Last caption: {analysis['last_caption_time']:.1f}s")
            
            # Extract frames (limit to first 10 captions for testing)
            base_name = os.path.splitext(vtt_file)[0]
            captions, files = extract_frames_from_subtitles(
                video_file, vtt_file, 
                output_dir=f"subtitle_frames_{base_name}",
                frame_type="middle",
                max_captions=10
            )
            
            print(f"Extracted {len(files)} frames")
            
        else:
            print(f"Files not found: {video_file} or {vtt_file}")