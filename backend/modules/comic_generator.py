from typing import TypedDict, Optional
import os
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
    panels = generate_panel_instructions(script, style)
    comic = generate_comic(panels)
    return comic

def generate_panel_instructions(script: str, style: str) -> list[PanelInstruction]:
    panels = []
    
    # Each panel dict should have: prev_image, instructions, style
    
    return panels

def generate_panels(panel_instruction: PanelInstruction):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash-image-preview"
    
    # Build prompt and parts
    prompt = f"{panel_instruction['instructions']}\nStyle: {panel_instruction['style']}"
    
    parts = []
    
    # Add previous image as multimodal input if it exists
    if panel_instruction["prev_image"]:
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
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        if chunk.candidates[0].content.parts[0].inline_data and chunk.candidates[0].content.parts[0].inline_data.data:
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            result["image_data"] = inline_data.data
            result["mime_type"] = inline_data.mime_type
        else:
            if "text" not in result:
                result["text"] = ""
            result["text"] += chunk.text if chunk.text else ""
    
    return result