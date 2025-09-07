import logging
import time
from fastapi import APIRouter
from pydantic import BaseModel, Field
from modules.comic_generator import generate_comic, generate_panel_instructions
from modules.storyboard_to_audio import generate_audio_from_panel_instructions
from modules.audio_mixer import create_audio_mixed_panels

# Set up logger
logger = logging.getLogger(__name__)

router = APIRouter()


class ComicDraftRequest(BaseModel):
    script: str = Field(
        ...,
        description="The full script/story for the comic. Can be screenplay format, narrative prose, or detailed scene descriptions.",
        example="""FADE IN:
Scene 1: A hero stands on a cliff overlooking the city at sunset.
Scene 2: Close-up of the hero's determined face.
Scene 3: The hero leaps into action."""
    )
    style: str = Field(
        ...,
        description="Visual art style for the comic panels",
        example="manga style, black and white with dramatic shading"
    )


@router.get("/generate-script")
async def upload_comic():
    return {
        "message": "Hello World",
        "status": "success"
    }


@router.post("/generate-comic-draft")
async def generate_comic_draft(request: ComicDraftRequest):
    panels = generate_comic(request.script, request.style)
    return {
        "panels": panels,
        "panel_count": len(panels),
        "status": "success"
    }

@router.post("/generate-audio-cues")
async def generate_audio_cues(request: ComicDraftRequest):
    try:
        logger.info("🎵 Starting audio generation")
        logger.info(f"📝 Script length: {len(request.script)} characters")
        
        logger.info("📋 Generating panel instructions...")
        panel_instructions = generate_panel_instructions(request.script, request.style)
        panel_count = len(panel_instructions) if panel_instructions else 0
        logger.info(f"✅ Generated {panel_count} panel instructions")
        
        logger.info("🎵 Generating audio files (music, SFX, narrative)...")
        audio_result = generate_audio_from_panel_instructions(panel_instructions)
        logger.info(f"✅ Generated audio for {audio_result['total_panels']} panels")
        logger.info(f"📁 Audio files saved to: {audio_result['output_directory']}")
        
        return {
            "audio_files": audio_result["files"], 
            "output_directory": audio_result["output_directory"],
            "total_panels": audio_result["total_panels"],
            "status": "success"
        }
    except Exception as e:
        logger.error(f"❌ Audio generation failed: {str(e)}")
        return {"error": str(e), "status": "error"}


# full multimedia comic
@router.post("/generate-multimedia-comic")
async def generate_multimedia_comic(request: ComicDraftRequest):
    try:
        start_time = time.time()
        logger.info("🎬 Starting multimedia comic generation")
        logger.info(f"📝 Script length: {len(request.script)} characters")
        logger.info(f"🎨 Style: {request.style}")
        
        # Step 1: Generate panel instructions
        step1_start = time.time()
        logger.info("📋 Step 1/3: Generating panel instructions...")
        panel_instructions = generate_panel_instructions(request.script, request.style)
        panel_count = len(panel_instructions) if panel_instructions else 0
        step1_time = time.time() - step1_start
        logger.info(f"✅ Generated {panel_count} panel instructions ({step1_time:.2f}s)")
        
        # Step 2: Generate comic panels (visual content)
        step2_start = time.time()
        logger.info("🖼️  Step 2/3: Generating comic panels (this may take a while)...")
        panels = generate_comic(request.script, request.style)
        panels_count = len(panels) if panels else 0
        step2_time = time.time() - step2_start
        logger.info(f"✅ Generated {panels_count} comic panels ({step2_time:.2f}s)")
        
        # Step 3: Generate audio (music, SFX, narrative)
        step3_start = time.time()
        logger.info("🎵 Step 3/4: Generating audio files (music, SFX, narrative)...")
        audio_result = generate_audio_from_panel_instructions(panel_instructions)
        audio_files_count = audio_result["total_panels"]
        audio_output_dir = audio_result["output_directory"]
        step3_time = time.time() - step3_start
        logger.info(f"✅ Generated audio for {audio_files_count} panels ({step3_time:.2f}s)")
        logger.info(f"📁 Audio files saved to: {audio_output_dir}")
        
        # Step 4: Mix narrative audio into panel movies
        step4_start = time.time()
        logger.info("🎬 Step 4/4: Mixing narrative audio into panel movies...")
        panels_with_audio = create_audio_mixed_panels(panels, audio_result["files"])
        step4_time = time.time() - step4_start
        logger.info(f"✅ Mixed audio into panel movies ({step4_time:.2f}s)")
        
        total_time = time.time() - start_time
        logger.info("🎉 Multimedia comic generation completed successfully!")
        logger.info(f"📊 Summary: {panels_count} panels, {audio_files_count} audio sets")
        logger.info(f"⏱️  Total time: {total_time:.2f}s (Instructions: {step1_time:.2f}s, Panels: {step2_time:.2f}s, Audio: {step3_time:.2f}s, Mixing: {step4_time:.2f}s)")
        
        return {
            "panels": panels_with_audio,
            "audio": audio_result["files"],
            "output_directory": audio_result["output_directory"],
            "panel_count": len(panels) if panels else 0,
            "audio_files_count": audio_result["total_panels"],
            "type": "multimedia_comic",
            "status": "success"
        }
    except Exception as e:
        logger.error(f"❌ Multimedia comic generation failed: {str(e)}")
        logger.error(f"🔍 Error details: {type(e).__name__}")
        return {"error": str(e), "status": "error"}

@router.get("/health")
async def health_check():
    return {"status": "healthy"}