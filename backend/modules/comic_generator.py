from typing import TypedDict, Optional
import os
import json
import mimetypes
import requests
import base64
from pathlib import Path
from datetime import datetime
from google import genai
from google.genai import types
import fal_client
import asyncio
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import threading
import time
from modules.movie_generator import generate_movie_from_panels

class ImageData(TypedDict):
    data: bytes
    mime_type: str

class PanelInstruction(TypedDict):
    prev_image: Optional[ImageData]
    instructions: str  # This will be the end state (existing behavior)
    init_state_instructions: str  # New: starting state for motion
    narrative: str  # Short narrative/dialogue for this panel
    style: str

def generate_comic(script: str, style: str) -> list[dict]:
    start_time = time.time()
    
    # Generate panel instructions first
    print("Generating panel instructions...")
    panel_instructions = generate_panel_instructions(script, style)
    
    if not panel_instructions:
        return {"error": "Failed to generate panel instructions", "panels": []}
    
    print(f"Generated {len(panel_instructions)} panel instructions")
    
    # Create output directory
    output_base = Path("./output")
    output_base.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / f"comic_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    comic = []
    prev_image = None
    movie_futures = []
    
    # Thread pool for parallel movie generation
    with ThreadPoolExecutor(max_workers=10) as executor:
        
        # SEQUENTIAL image generation to maintain continuity
        for index, instruction in enumerate(panel_instructions):
            panel_num = index + 1
            print(f"\nGenerating panel {panel_num}/{len(panel_instructions)}...")
            
            result = {
                "panel_number": panel_num,
                "instructions": instruction["instructions"],
                "init_state_instructions": instruction.get("init_state_instructions", instruction["instructions"]),
                "narrative": instruction.get("narrative", ""),
                "style": instruction["style"],
                "status": "processing"
            }
            
            # Generate INIT STATE
            print(f"  Generating init state...")
            init_instruction = dict(instruction)
            init_instruction["instructions"] = instruction["init_state_instructions"]
            
            if prev_image:
                init_instruction["prev_image"] = prev_image
                init_instruction["instructions"] = f"Continue from previous image. {init_instruction['instructions']}"
            
            init_result = None
            init_saved = False
            max_retries = 5
            
            for attempt in range(1, max_retries + 1):
                try:
                    if attempt > 1:
                        print(f"    Retrying init state (attempt {attempt}/{max_retries})...")
                    
                    init_result = generate_panels(init_instruction)
                    
                    if "image_data" in init_result and "mime_type" in init_result:
                        file_ext = mimetypes.guess_extension(init_result["mime_type"]) or ".png"
                        init_path = output_dir / f"panel_{panel_num:03d}_init{file_ext}"
                        
                        with open(init_path, "wb") as f:
                            f.write(init_result["image_data"])
                        
                        if init_path.exists() and init_path.stat().st_size > 0:
                            with open(init_path, "rb") as f:
                                saved_data = f.read()
                                if len(saved_data) == len(init_result["image_data"]):
                                    print(f"    Init state saved: {init_path.name}")
                                    result["init_state_path"] = str(init_path)
                                    init_saved = True
                                    # Update prev_image to init for end state generation
                                    prev_image = {
                                        "data": init_result["image_data"],
                                        "mime_type": init_result["mime_type"]
                                    }
                                    break
                except Exception as e:
                    print(f"    Init state error (attempt {attempt}): {str(e)}")
                    if attempt >= max_retries:
                        result["init_state_error"] = str(e)
            
            if not init_saved:
                result["status"] = "error"
                result["error"] = "Failed to generate init state"
                comic.append(result)
                continue
            
            # Generate END STATE using init as reference
            print(f"  Generating end state...")
            end_instruction = dict(instruction)
            end_instruction["prev_image"] = prev_image  # Use init state as reference
            end_instruction["instructions"] = f"Continue from this exact scene with minimal change. {end_instruction['instructions']}"
            
            end_result = None
            end_saved = False
            
            for attempt in range(1, max_retries + 1):
                try:
                    if attempt > 1:
                        print(f"    Retrying end state (attempt {attempt}/{max_retries})...")
                    
                    end_result = generate_panels(end_instruction)
                    
                    if "image_data" in end_result and "mime_type" in end_result:
                        file_ext = mimetypes.guess_extension(end_result["mime_type"]) or ".png"
                        end_path = output_dir / f"panel_{panel_num:03d}_end{file_ext}"
                        
                        with open(end_path, "wb") as f:
                            f.write(end_result["image_data"])
                        
                        if end_path.exists() and end_path.stat().st_size > 0:
                            with open(end_path, "rb") as f:
                                saved_data = f.read()
                                if len(saved_data) == len(end_result["image_data"]):
                                    print(f"    End state saved: {end_path.name}")
                                    result["end_state_path"] = str(end_path)
                                    result["file_path"] = str(end_path)
                                    end_saved = True
                                    # Update prev_image to end state for next panel
                                    prev_image = {
                                        "data": end_result["image_data"],
                                        "mime_type": end_result["mime_type"]
                                    }
                                    break
                except Exception as e:
                    print(f"    End state error (attempt {attempt}): {str(e)}")
                    if attempt >= max_retries:
                        result["end_state_error"] = str(e)
            
            if not end_saved:
                result["status"] = "partial_success"
                result["error"] = "Failed to generate end state"
            else:
                result["status"] = "success"
            
            # Submit movie generation to thread pool IMMEDIATELY after both states are ready
            if init_saved and end_saved:
                print(f"  Submitting movie generation to parallel queue...")
                movie_future = executor.submit(
                    generate_movie_from_panels,
                    result["init_state_path"],
                    result["end_state_path"],
                    instruction["instructions"],
                    str(output_dir / f"panel_{panel_num:03d}_movie.mp4")
                )
                movie_futures.append((panel_num, movie_future))
            
            comic.append(result)
            print(f"  Panel {panel_num} complete: {result['status']}")
        
        # Wait for all movies to complete in parallel
        if movie_futures:
            print(f"\nWaiting for {len(movie_futures)} movie generations (running in parallel)...")
            movies_done = 0
            
            # Process movies as they complete (not in order)
            for panel_num, future in movie_futures:
                try:
                    movie_result = future.result(timeout=300)
                    panel_idx = panel_num - 1
                    
                    if panel_idx < len(comic):
                        comic[panel_idx]["movie"] = movie_result
                    
                    movies_done += 1
                    status_icon = "✓" if movie_result.get("status") == "success" else "✗"
                    print(f"{status_icon} Movie {panel_num} completed ({movies_done}/{len(movie_futures)})")
                    
                except Exception as e:
                    print(f"✗ Movie {panel_num} error: {str(e)}")
                    panel_idx = panel_num - 1
                    if panel_idx < len(comic):
                        comic[panel_idx]["movie"] = {"status": "error", "error": str(e)}
                    movies_done += 1
    
    # Verify completion
    total_time = time.time() - start_time
    successful_panels = sum(1 for p in comic if p.get("status") == "success")
    successful_movies = sum(1 for p in comic if p.get("movie", {}).get("status") == "success")
    
    print(f"\n" + "="*50)
    print(f"Comic generation complete in {total_time:.1f} seconds")
    print(f"Panels: {successful_panels}/{len(panel_instructions)} successful")
    print(f"Movies: {successful_movies}/{len(movie_futures)} successful")  
    print(f"Output: {output_dir}")
    print("="*50)
    
    return comic
def generate_panel_instructions(script: str, style: str) -> list[PanelInstruction]:
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-pro"
    
    prompt = f"""Generate explicit storyboarding instructions for an anime-style comic. Create detailed panel-by-panel visual descriptions with both starting and ending states showing MINIMAL movement within each panel.

Each panel must have:
- init_state_instructions: DETAILED description of the panel's STARTING STATE including:
  * Exact character positions and poses at the beginning
  * Starting facial expressions and emotions
  * Objects and elements in their initial positions  
  * Background and environment details
  * Camera angle and framing
  * Lighting and atmosphere
  * This is a STATIC image - no motion blur or speed lines
  
- instructions: DETAILED description of the panel's ENDING STATE including:
  * Character positions after ONE SINGLE SMALL ACTION (e.g., turning head, raising hand, taking one step, changing expression)
  * The scene should be 90% identical to init_state
  * Only ONE primary change: a gesture, expression, small movement, or single object interaction
  * Background remains the SAME
  * Camera angle remains the SAME (no panning or zooming)
  * Lighting remains mostly the SAME
  * This is a STATIC image showing a TINY progression from init_state
  
- narrative: SHORT and CRISP text describing action or dialogue for this panel:
  * Character dialogue (e.g., "I'll never give up!")
  * Action descriptions (e.g., "He leaps through the window")
  * Brief narration when needed (e.g., "Meanwhile, across town...")
  * Keep it concise - like comic book text
  * Can combine multiple elements (e.g., "Watch out!" she yelled as the door exploded.)
  
- style: the visual style for this panel (based on provided style)

CRITICAL RULES:
1. Init and end states must be NEARLY IDENTICAL with only ONE small change
2. Think of it like two consecutive frames in an animation - minimal difference
3. Major scene changes happen BETWEEN panels, not WITHIN a panel
4. Examples of good single changes:
   - Character's mouth opens to speak
   - Hand moves to grab an object
   - Eyes widen in surprise
   - Head turns to look at something
   - Takes one step forward
   - Object falls or moves slightly
5. Both states are STATIC images - no motion blur or speed lines
6. The two states should be so similar that if overlapped, 90% would match exactly
7. Narrative must be SHORT - include dialogue, actions, or sounds as needed

Be extremely specific - these instructions will be used to generate images directly.

Script: {script}
Style: {style}"""

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    
    # Define response schema for structured output
    response_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "init_state_instructions": {
                    "type": "string",
                    "description": "Detailed description of the panel's starting state before any motion"
                },
                "instructions": {
                    "type": "string",
                    "description": "Detailed visual description of the panel's ending state after motion completes"
                },
                "narrative": {
                    "type": "string",
                    "description": "Short text describing action, dialogue, or sound effects for this panel (1-2 sentences max)"
                },
                "style": {
                    "type": "string", 
                    "description": "Visual style for this panel"
                }
            },
            "required": ["instructions", "init_state_instructions", "narrative", "style"]
        }
    }
    
    generate_content_config = types.GenerateContentConfig(
        thinking_config = types.ThinkingConfig(
            thinking_budget=-1,
        ),
        response_mime_type="application/json",
        response_schema=response_schema
    )

    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.text:
            response_text += chunk.text
    
    # Handle empty response
    if not response_text:
        return []
    
    # Parse JSON response (should be clean JSON due to structured output)
    try:
        panels_data = json.loads(response_text)
    except json.JSONDecodeError:
        # Return empty list if parsing fails
        return []
    
    # Convert to PanelInstruction format
    panels = []
    for panel_data in panels_data:
        panel = PanelInstruction(
            prev_image=None,  # Will be filled during sequential generation
            instructions=panel_data["instructions"],  # End state
            init_state_instructions=panel_data.get("init_state_instructions", panel_data["instructions"]),
            narrative=panel_data.get("narrative", ""),
            style=panel_data["style"]
        )
        panels.append(panel)
    
    return panels

def generate_panels(panel_instruction: PanelInstruction):
    try:
        # FAL_KEY should be set in environment or .env file
        if not os.environ.get("FAL_KEY"):
            raise Exception("FAL_KEY not found in environment variables")
        
        # Build prompt - using instructions as the main directive
        # The instructions field now represents either init or end state depending on context
        prompt = f"{panel_instruction['instructions']}\nStyle: {panel_instruction['style']}"
        
        # Prepare arguments for fal-ai
        arguments = {
            "prompt": prompt
        }
        
        # Determine which model to use and prepare arguments
        model_id = "fal-ai/nano-banana"
        
        if panel_instruction.get("prev_image"):
            # For edit mode, we need image URLs
            # Save prev image temporarily
            temp_path = Path("/tmp") / "prev_panel.png"
            with open(temp_path, "wb") as f:
                f.write(panel_instruction["prev_image"]["data"])
            
            # Use base64 data URL (in production, upload to CDN)
            base64_image = base64.b64encode(panel_instruction["prev_image"]["data"]).decode('utf-8')
            image_data_url = f"data:{panel_instruction['prev_image']['mime_type']};base64,{base64_image}"
            
            # Update for edit mode
            model_id = "fal-ai/nano-banana/edit"
            arguments["image_urls"] = [image_data_url]
            arguments["prompt"] = f"Continue from the previous panel. {prompt}"
        
        # Use subscribe for synchronous execution with optional logging
        def on_queue_update(update):
            if hasattr(update, 'logs'):
                for log in update.logs:
                    if isinstance(log, dict) and "message" in log:
                        print(f"FAL: {log['message']}")
        
        # Call fal-ai with subscribe
        result = fal_client.subscribe(
            model_id,
            arguments=arguments,
            with_logs=True,
            on_queue_update=on_queue_update
        )
        
        # Process the result
        response = {}
        
        # Debug: Print the result structure
        print(f"FAL Result type: {type(result)}")
        if result:
            print(f"FAL Result keys: {result.keys() if hasattr(result, 'keys') else 'Not a dict'}")
            print(f"FAL Result: {result}")
        
        # Handle different possible response structures
        image_url = None
        
        # Check for image in various possible locations
        if result:
            if isinstance(result, dict):
                # Check direct image field
                if "image" in result:
                    if isinstance(result["image"], str):
                        image_url = result["image"]
                    elif isinstance(result["image"], dict) and "url" in result["image"]:
                        image_url = result["image"]["url"]
                # Check for images array (some models return multiple)
                elif "images" in result and isinstance(result["images"], list) and len(result["images"]) > 0:
                    first_image = result["images"][0]
                    if isinstance(first_image, str):
                        image_url = first_image
                    elif isinstance(first_image, dict) and "url" in first_image:
                        image_url = first_image["url"]
                # Check for output field
                elif "output" in result:
                    if isinstance(result["output"], str):
                        image_url = result["output"]
                    elif isinstance(result["output"], dict) and "url" in result["output"]:
                        image_url = result["output"]["url"]
        
        if image_url:
            print(f"Downloading image from: {image_url}")
            
            # Retry logic for downloading image
            max_retries = 5
            retry_count = 0
            image_response = None
            
            while retry_count < max_retries:
                try:
                    image_response = requests.get(image_url, timeout=30)
                    
                    if image_response.status_code == 200:
                        response["image_data"] = image_response.content
                        # Determine mime type from content-type header or default to png
                        content_type = image_response.headers.get('content-type', 'image/png')
                        response["mime_type"] = content_type
                        print(f"Image downloaded successfully, size: {len(image_response.content)} bytes")
                        break
                    else:
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"Download failed with status {image_response.status_code}, retrying... (attempt {retry_count + 1}/{max_retries})")
                        else:
                            raise Exception(f"Failed to download image from {image_url} after {max_retries} attempts, last status: {image_response.status_code}")
                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Download error: {str(e)}, retrying... (attempt {retry_count + 1}/{max_retries})")
                    else:
                        raise Exception(f"Failed to download image from {image_url} after {max_retries} attempts, last error: {str(e)}")
        else:
            print(f"No image URL found in result: {result}")
            raise Exception("No image URL found in fal-ai response")
            
        return response
        
    except Exception as e:
        print(f"Error in generate_panels with fal-ai: {str(e)}")
        # Fallback to a simple response or raise
        raise
