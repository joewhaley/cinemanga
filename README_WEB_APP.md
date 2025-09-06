# Video Scene Detection Web Application

A modern web application for analyzing video content using FAL AI's video understanding capabilities. Upload video files or provide URLs to get detailed scene analysis and extraction.

## Features

- 🎥 **File Upload**: Upload video files directly from your device
- 🔗 **URL Processing**: Analyze videos from URLs (YouTube, direct links, etc.)
- 🤖 **AI-Powered Analysis**: Uses FAL AI for advanced video understanding
- 📊 **Detailed Results**: Comprehensive scene analysis with timestamps
- 🎨 **Modern UI**: Beautiful, responsive interface built with React and Tailwind CSS
- ⚡ **Real-time Processing**: Background processing with status updates

## Prerequisites

1. **Python 3.8+** installed on your system
2. **FAL AI API Key** - Get one from [fal.ai](https://fal.ai)
3. **Required Python packages** (see requirements.txt)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

Create a `.env` file in the project root:

```bash
FAL_KEY=your_fal_api_key_here
```

Replace `your_fal_api_key_here` with your actual FAL AI API key.

### 3. Start the Application

```bash
python start_web_app.py
```

The application will start on `http://localhost:8001`

## Usage

### Upload a Video File

1. Open the web application in your browser
2. Click on the "Upload File" tab
3. Select a video file from your device
4. Click "Analyze Video"
5. Wait for processing to complete
6. View the detailed analysis results

### Process a Video URL

1. Click on the "Video URL" tab
2. Enter a video URL (YouTube, direct link, etc.)
3. Click "Analyze Video"
4. Wait for processing to complete
5. View the detailed analysis results

## API Endpoints

The backend provides the following REST API endpoints:

- `POST /api/upload` - Upload and process a video file
- `POST /api/process-url` - Process a video from URL
- `GET /api/status/{result_id}` - Get processing status
- `GET /api/results/{result_id}` - Get analysis results
- `GET /api/health` - Health check

## Project Structure

```
cinemanga/
├── backend/
│   ├── main.py              # FastAPI backend server
│   ├── modules/
│   │   ├── __init__.py      # Modules package
│   │   └── fal_scene_detector.py  # FAL AI integration
│   └── static/
│       └── index.html       # React frontend
├── start_web_app.py         # Startup script
├── requirements.txt         # Python dependencies
└── README_WEB_APP.md        # This file
```

## Technical Details

### Backend (FastAPI)

- **Framework**: FastAPI with Uvicorn server
- **Processing**: Asynchronous background tasks
- **File Handling**: Temporary file storage with cleanup
- **CORS**: Enabled for frontend communication

### Frontend (React)

- **Framework**: React 18 with Babel
- **Styling**: Tailwind CSS
- **Icons**: Font Awesome
- **Features**: File upload, URL input, real-time status updates

### AI Integration

- **Service**: FAL AI Video Understanding API
- **Analysis**: Multiple prompts for comprehensive scene analysis
- **Output**: JSON format with detailed results and extracted scenes

## Supported Video Formats

- **MP4** - Most common format
- **AVI** - Windows standard
- **MOV** - Apple QuickTime
- **MKV** - Matroska Video (high quality)
- **WebM** - Web optimized
- **M4V** - iTunes video
- **FLV** - Flash video
- **WMV** - Windows Media Video

## Troubleshooting

### Common Issues

1. **"FAL_KEY must be set"**
   - Make sure your `.env` file exists and contains a valid FAL_KEY

2. **"Missing required package"**
   - Run `pip install -r requirements.txt`

3. **"Upload failed"**
   - Check file size and format
   - Ensure stable internet connection
   - For MKV files: Ensure the file is not corrupted and has proper video streams

4. **"Processing failed"**
   - Check FAL AI API key validity
   - Verify video URL accessibility

### Logs

The application logs processing status and errors to the console. Check the terminal where you started the server for detailed information.

## Development

### Running in Development Mode

The server runs with auto-reload enabled by default. Any changes to the backend code will automatically restart the server.

### Frontend Development

The frontend is a single HTML file with embedded React. For more complex frontend development, consider extracting it to a separate React project.

## Security Notes

- The application currently allows CORS from all origins (`*`)
- In production, specify your frontend domain in the CORS settings
- File uploads are stored temporarily and cleaned up after processing
- API keys should be kept secure and not exposed in client-side code

## License

This project is part of the cinemanga video analysis toolkit.
