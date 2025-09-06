from fastapi import APIRouter
from pydantic import BaseModel, Field
from modules.comic_generator import generate_comic
from modules.storyboard_to_audio import generate_audio_from_storyboard

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
        audio_files = generate_audio_from_storyboard(request.script)
        return {"audio_files": audio_files, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "error"}


# full multimedia comic
@router.post("/generate-multimedia-comic")
async def generate_multimedia_comic(request: ComicDraftRequest):
    try:
        # Parallel generation
        panels = generate_comic(request.script, request.style)
        audio = generate_audio_from_storyboard(request.script)
        
        return {
            "panels": panels,
            "audio": audio,
            "type": "multimedia_comic",
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}

@router.get("/health")
async def health_check():
    return {"status": "healthy"}