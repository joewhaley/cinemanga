#!/usr/bin/env python3
"""
FastAPI backend for video scene detection web application
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import asyncio
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Import the FAL scene detector
from modules.fal_scene_detector import FALSceneDetector

# Import the comic generation router
from routers import router as comic_router

app = FastAPI(title="Cinemanga API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(comic_router, prefix="/api")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global storage for processing results
processing_results = {}

class VideoURLRequest(BaseModel):
    url: str

class ProcessingStatus(BaseModel):
    status: str
    message: str
    result_id: Optional[str] = None

class AnalysisResult(BaseModel):
    result_id: str
    video_name: str
    video_url: str
    analysis_results: Dict[str, Any]
    extracted_scenes: list
    total_scenes: int
    processed_at: float
    status: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    return FileResponse("static/index.html")

@app.post("/api/upload", response_model=ProcessingStatus)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process a video file"""
    try:
        # Debug: Print file information
        print(f"DEBUG: File name: {file.filename}")
        print(f"DEBUG: Content type: {file.content_type}")
        print(f"DEBUG: File size: {file.size}")
        
        # Additional validation for supported formats
        supported_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.flv', '.wmv']
        file_extension = Path(file.filename).suffix.lower()
        
        # Validate file type - check both MIME type and file extension
        is_video_mime = file.content_type and file.content_type.startswith('video/')
        is_video_extension = file_extension in supported_extensions
        
        if not (is_video_mime or is_video_extension):
            print(f"DEBUG: Invalid file - MIME: {file.content_type}, Extension: {file_extension}")
            raise HTTPException(
                status_code=400, 
                detail=f"File must be a video. Detected MIME: {file.content_type}, Extension: {file_extension}. Supported formats: {', '.join(supported_extensions)}"
            )
        
        if not is_video_extension:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported formats: {', '.join(supported_extensions)}"
            )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            temp_path = tmp_file.name
        
        # Generate result ID
        result_id = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        
        # Store initial status
        processing_results[result_id] = {
            "status": "processing",
            "message": "Video uploaded, starting analysis...",
            "video_name": file.filename,
            "video_path": temp_path
        }
        
        # Start background processing
        background_tasks.add_task(process_uploaded_video, result_id, temp_path)
        
        return ProcessingStatus(
            status="processing",
            message="Video uploaded successfully, processing started",
            result_id=result_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/process-url", response_model=ProcessingStatus)
async def process_video_url(
    background_tasks: BackgroundTasks,
    request: VideoURLRequest
):
    """Process a video from URL (e.g., YouTube)"""
    try:
        # Generate result ID
        result_id = f"url_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store initial status
        processing_results[result_id] = {
            "status": "processing",
            "message": "Starting URL processing...",
            "video_url": request.url
        }
        
        # Start background processing
        background_tasks.add_task(process_video_url_task, result_id, request.url)
        
        return ProcessingStatus(
            status="processing",
            message="URL processing started",
            result_id=result_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL processing failed: {str(e)}")

@app.get("/api/status/{result_id}", response_model=ProcessingStatus)
async def get_processing_status(result_id: str):
    """Get the processing status for a result ID"""
    if result_id not in processing_results:
        raise HTTPException(status_code=404, detail="Result not found")
    
    result = processing_results[result_id]
    return ProcessingStatus(
        status=result["status"],
        message=result["message"],
        result_id=result_id
    )

@app.get("/api/results/{result_id}", response_model=AnalysisResult)
async def get_analysis_results(result_id: str):
    """Get the complete analysis results"""
    if result_id not in processing_results:
        raise HTTPException(status_code=404, detail="Result not found")
    
    result = processing_results[result_id]
    if result["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed yet")
    
    return AnalysisResult(**result)

async def process_uploaded_video(result_id: str, video_path: str):
    """Background task to process uploaded video"""
    try:
        # Update status
        processing_results[result_id]["message"] = "Initializing FAL AI detector..."
        
        # Initialize detector
        detector = FALSceneDetector()
        
        # Update status
        processing_results[result_id]["message"] = "Uploading video to FAL AI..."
        
        # Process video
        results = detector.process_video(video_path, output_dir="temp_output")
        
        # Update results
        processing_results[result_id].update({
            "result_id": result_id,
            "status": "completed",
            "message": "Analysis completed successfully",
            "video_url": results["video_url"],
            "analysis_results": results["analysis_results"],
            "extracted_scenes": results["extracted_scenes"],
            "total_scenes": results["total_scenes"],
            "processed_at": results["processed_at"]
        })
        
        # Clean up temporary file
        try:
            os.unlink(video_path)
        except:
            pass
            
    except Exception as e:
        processing_results[result_id].update({
            "status": "error",
            "message": f"Processing failed: {str(e)}"
        })

async def process_video_url_task(result_id: str, video_url: str):
    """Background task to process video from URL"""
    try:
        # Update status
        processing_results[result_id]["message"] = "Initializing FAL AI detector..."
        
        # Initialize detector
        detector = FALSceneDetector()
        
        # Update status
        processing_results[result_id]["message"] = "Analyzing video with FAL AI..."
        
        # Analyze video directly from URL
        analysis_results = detector.analyze_video_scenes(video_url, detailed_analysis=True)
        
        # Extract scenes
        scenes = detector.extract_scene_timestamps(analysis_results)
        
        # Update results
        processing_results[result_id].update({
            "result_id": result_id,
            "status": "completed",
            "message": "Analysis completed successfully",
            "video_name": "URL Video",
            "video_url": video_url,
            "analysis_results": analysis_results,
            "extracted_scenes": scenes,
            "total_scenes": len(scenes),
            "processed_at": datetime.now().timestamp()
        })
        
    except Exception as e:
        processing_results[result_id].update({
            "status": "error",
            "message": f"Processing failed: {str(e)}"
        })

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    # Create static directory if it doesn't exist
    Path("static").mkdir(exist_ok=True)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )