#!/usr/bin/env python3
"""Test script for SRT and VTT subtitle format support"""

from vtt_frame_extractor import SubtitleParser

def test_subtitle_formats():
    parser = SubtitleParser()
    
    print("Testing SRT format:")
    srt_captions = parser.parse_subtitle_file('test_sample.srt')
    print(f"Parsed {len(srt_captions)} SRT captions:")
    for i, caption in enumerate(srt_captions):
        print(f"  {i+1}: {caption.start_time:.2f}s-{caption.end_time:.2f}s: {caption.text}")
    
    print("\nTesting VTT format:")
    vtt_file = "Inception (2010) - You're in a Dream Scene (2⧸10) ｜ Movieclips [i3-jlhJgU9U].en.vtt"
    try:
        vtt_captions = parser.parse_subtitle_file(vtt_file)
        print(f"Parsed {len(vtt_captions)} VTT captions:")
        for i, caption in enumerate(vtt_captions[:3]):  # Show first 3 only
            print(f"  {i+1}: {caption.start_time:.2f}s-{caption.end_time:.2f}s: {caption.text}")
        if len(vtt_captions) > 3:
            print(f"  ... and {len(vtt_captions) - 3} more")
    except FileNotFoundError:
        print("VTT file not found, skipping VTT test")

if __name__ == "__main__":
    test_subtitle_formats()