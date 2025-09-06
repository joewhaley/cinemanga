from fastapi import APIRouter
from pydantic import BaseModel
from modules import generate_comic, generate_movie

router = APIRouter()


class ComicDraftRequest(BaseModel):
    script: str
    style: str


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
        "status": "success"
    }


@router.get("/health")
async def health_check():
    return {"status": "healthy"}