import os
import base64
import requests
import mimetypes
from pathlib import Path
import fal_client

def generate_movie_from_panels(init_image_path: str, end_image_path: str, instructions: str, output_path: str = None) -> dict:
    """
    Generate a movie/animation transitioning between init and end state images.
    
    Args:
        init_image_path: Path to the init state image file
        end_image_path: Path to the end state image file
        instructions: Text description of the motion/transition
        output_path: Optional path to save the movie file
    
    Returns:
        dict with movie generation results including path to saved movie
    """
    try:
        # FAL_KEY should be set in environment
        if not os.environ.get("FAL_KEY"):
            raise Exception("FAL_KEY not found in environment variables")
        
        # Read the images
        with open(init_image_path, "rb") as f:
            init_image_data = f.read()
        with open(end_image_path, "rb") as f:
            end_image_data = f.read()
        
        # Convert to base64 data URLs for Fal AI
        init_mime = mimetypes.guess_type(init_image_path)[0] or "image/png"
        end_mime = mimetypes.guess_type(end_image_path)[0] or "image/png"
        
        init_base64 = base64.b64encode(init_image_data).decode('utf-8')
        end_base64 = base64.b64encode(end_image_data).decode('utf-8')
        
        init_data_url = f"data:{init_mime};base64,{init_base64}"
        end_data_url = f"data:{end_mime};base64,{end_base64}"
        
        # Use the wan-flf2v model for first-to-last frame video generation
        model_id = "fal-ai/wan-flf2v"
        
        # Prepare arguments matching the model's expected format
        arguments = {
            "prompt": f"A smooth animation transition showing: {instructions}",
            "start_image_url": init_data_url,  # Changed from init_image to start_image_url
            "end_image_url": end_data_url       # Changed from end_image to end_image_url
        }
        
        # Generate video
        print(f"Generating movie: {instructions[:50]}...")
        
        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    print(f"  Movie Gen: {log['message']}")
        
        result = fal_client.subscribe(
            model_id,
            arguments=arguments,
            with_logs=True,
            on_queue_update=on_queue_update
        )
        
        # Extract video URL from result
        video_url = None
        if result:
            print(f"Result type: {type(result)}")
            if hasattr(result, '__dict__'):
                print(f"Result attributes: {result.__dict__}")
            elif isinstance(result, dict):
                print(f"Result keys: {result.keys()}")
            
            # Try different possible response structures
            if isinstance(result, dict):
                if "video" in result:
                    if isinstance(result["video"], dict) and "url" in result["video"]:
                        video_url = result["video"]["url"]
                    elif isinstance(result["video"], str):
                        video_url = result["video"]
                elif "output" in result:
                    if isinstance(result["output"], dict) and "url" in result["output"]:
                        video_url = result["output"]["url"]
                    elif isinstance(result["output"], str):
                        video_url = result["output"]
                elif "url" in result:
                    video_url = result["url"]
            elif hasattr(result, 'video'):
                video_url = result.video if isinstance(result.video, str) else getattr(result.video, 'url', None)
            elif hasattr(result, 'output'):
                video_url = result.output if isinstance(result.output, str) else getattr(result.output, 'url', None)
            elif hasattr(result, 'url'):
                video_url = result.url
        
        if not video_url:
            return {
                "status": "error",
                "error": "No video URL in response"
            }
        
        # Download video with retry
        print(f"Downloading movie from: {video_url}")
        max_retries = 5
        
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(video_url, timeout=60)
                if response.status_code == 200:
                    # Determine output path
                    if not output_path:
                        # Generate default path based on input
                        base_dir = Path(init_image_path).parent
                        panel_name = Path(init_image_path).stem.replace('_init', '')
                        output_path = base_dir / f"{panel_name}_movie.mp4"
                    
                    # Save movie
                    output_path = Path(output_path)
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    
                    # Verify save
                    if output_path.exists() and output_path.stat().st_size > 0:
                        print(f"Movie saved to {output_path}")
                        return {
                            "status": "success",
                            "movie_path": str(output_path),
                            "file_size": output_path.stat().st_size
                        }
                else:
                    print(f"Download failed (status {response.status_code}), attempt {attempt}/{max_retries}")
                    if attempt >= max_retries:
                        return {
                            "status": "error",
                            "error": f"Download failed with status {response.status_code}"
                        }
            except Exception as e:
                print(f"Download error on attempt {attempt}: {str(e)}")
                if attempt >= max_retries:
                    raise
        
    except Exception as e:
        print(f"Error generating movie: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }