import os
from google import genai
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    print("Please add GEMINI_API_KEY=your_api_key to your .env file")
    exit(1)

# Configure the client with your API key
client = genai.Client(api_key=API_KEY)

# The text prompt for image generation
prompt = "Create a photorealistic image of an orange cat with green eyes, sitting on a couch."

print("Generating image...")

text_prompt = "Create a side view picture of that cat, in a tropical forest, eating a nano-banana, under the stars" # @param {type:"string"}

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        text_prompt,
        PIL.Image.open('cat.png')
    ]
)

display_response(response)
save_image(response, 'cat_tropical.png')

try:
    # Call the API to generate the image
    response = client.models.generate_content(
        model="gemini-2.5-flash-image-preview",
        contents=prompt,
    )

    image_parts = [
        part.inline_data.data
        for part in response.candidates[0].content.parts
        if part.inline_data
    ]
    
    if image_parts:
        image = Image.open(BytesIO(image_parts[0]))
        image.save('cat.png')
        print("Image saved as 'cat.png'")
        print("Image generated successfully!")
    else:
        print("No image data found in the response")
        
except Exception as e:
    print(f"Error generating image: {e}")