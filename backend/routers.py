from fastapi import APIRouter
from pydantic import BaseModel, Field
from modules.comic_generator import generate_comic

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


@router.get("/health")
async def health_check():
    return {"status": "healthy"}