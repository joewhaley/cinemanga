from typing import TypedDict, Optional
import os
import json
import mimetypes
from pathlib import Path
from datetime import datetime
from google import genai
from google.genai import types

class ImageData(TypedDict):
    data: bytes
    mime_type: str

class PanelInstruction(TypedDict):
    prev_image: Optional[ImageData]
    instructions: str
    style: str

def generate_comic(script: str, style: str) -> list[dict]:
    # Generate panel instructions first
    panel_instructions = generate_panel_instructions(script, style)
    
    if not panel_instructions:
        return {"error": "Failed to generate panel instructions", "panels": []}
    
    comic = []
    prev_image = None
    
    # Create output directory for storing images
    output_base = Path("./output")
    output_base.mkdir(exist_ok=True)
    
    # Create timestamped directory for this comic
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / f"comic_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    print(f"Saving panels to: {output_dir}")
    
    for index, instruction in enumerate(panel_instructions):
        print(f"Generating panel {index + 1}/{len(panel_instructions)}...")
        
        # Create a copy of instruction to avoid modifying original
        current_instruction = dict(instruction)
        
        # Add previous image context if available
        if prev_image:
            current_instruction["prev_image"] = prev_image
            # Add context to instructions
            current_instruction["instructions"] = f"Continue from previous panel. {current_instruction['instructions']}"
        
        try:
            # Generate panel
            panel_result = generate_panels(current_instruction)
            
            # Prepare response data
            response_data = {
                "panel_number": index + 1,
                "instructions": instruction["instructions"],
                "style": instruction["style"]
            }
            
            # Save image if generated
            if "image_data" in panel_result and "mime_type" in panel_result:
                # Get file extension from mime type
                file_extension = mimetypes.guess_extension(panel_result["mime_type"])
                if not file_extension:
                    file_extension = ".png"
                
                # Save image to output directory
                image_path = output_dir / f"panel_{index + 1:03d}{file_extension}"
                with open(image_path, "wb") as f:
                    f.write(panel_result["image_data"])
                
                # Store current panel's image for next iteration
                prev_image = {
                    "data": panel_result["image_data"],
                    "mime_type": panel_result["mime_type"]
                }
                
                # Add file path to response
                response_data["file_path"] = str(image_path)
                response_data["status"] = "success"
            else:
                response_data["status"] = "no_image_generated"
            
            # Add any text response
            if "text" in panel_result:
                response_data["generation_notes"] = panel_result["text"]
            
            comic.append(response_data)
            
        except Exception as e:
            print(f"Error generating panel {index + 1}: {str(e)}")
            comic.append({
                "panel_number": index + 1,
                "status": "error",
                "error": str(e),
                "instructions": instruction["instructions"]
            })
    
    return comic

def generate_panel_instructions(script: str, style: str) -> list[PanelInstruction]:
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-pro"
    
    prompt = f"""Generate explicit storyboarding instructions for a comic. Create direct, detailed panel-by-panel visual descriptions.

Each panel must have:
- instructions: EXPLICIT and DETAILED visual description including:
  * Exact camera angle (close-up, medium shot, wide shot, bird's eye, worm's eye, etc.)
  * Character positions, poses, and facial expressions
  * Background elements and setting details
  * Lighting and mood
  * Any visual effects or motion indicators
  * Specific objects or props visible
  * Panel composition and framing
- style: the visual style for this panel (based on provided style)

Be extremely specific - these instructions will be used to generate images directly. Describe EXACTLY what should be visible in each panel.

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
                "instructions": {
                    "type": "string",
                    "description": "Detailed visual description of what should be in this panel"
                },
                "style": {
                    "type": "string", 
                    "description": "Visual style for this panel"
                }
            },
            "required": ["instructions", "style"]
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
            instructions=panel_data["instructions"],
            style=panel_data["style"]
        )
        panels.append(panel)
    
    return panels

def generate_panels(panel_instruction: PanelInstruction):
    try:
        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )

        model = "gemini-2.5-flash-image-preview"
        
        # Build prompt with clear instructions
        if panel_instruction.get("prev_image"):
            prompt = f"""Based on the previous image provided, generate the next panel.
{panel_instruction['instructions']}
Style: {panel_instruction['style']}
Maintain visual continuity with the previous panel."""
        else:
            prompt = f"""Generate the first panel of a comic.
{panel_instruction['instructions']}
Style: {panel_instruction['style']}"""
        
        parts = []
        
        # Add previous image as multimodal input if it exists
        if panel_instruction.get("prev_image"):
            parts.append(
                types.Part.from_bytes(
                    data=panel_instruction["prev_image"]["data"],
                    mime_type=panel_instruction["prev_image"]["mime_type"]
                )
            )
        
        parts.append(types.Part.from_text(text=prompt))
        
        contents = [
            types.Content(
                role="user",
                parts=parts,
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            response_modalities=[
                "IMAGE",
                "TEXT",
            ],
        )

        result = {}
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None
                or len(chunk.candidates) == 0
                or chunk.candidates[0].content is None
                or chunk.candidates[0].content.parts is None
            ):
                continue
                
            for part in chunk.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    result["image_data"] = part.inline_data.data
                    result["mime_type"] = part.inline_data.mime_type
                elif hasattr(part, 'text') and part.text:
                    if "text" not in result:
                        result["text"] = ""
                    result["text"] += part.text
        
        if not result:
            raise Exception("No content generated from API")
            
        return result
        
    except Exception as e:
        print(f"Error in generate_panels: {str(e)}")
        raise