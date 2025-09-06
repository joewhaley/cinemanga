#!/usr/bin/env python3
"""
Startup script for the Video Scene Detection Web Application
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import fal_client
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_env_file():
    """Check if .env file exists with FAL_KEY"""
    env_file = Path(".env")
    if not env_file.exists():
        print("✗ .env file not found")
        print("Please create a .env file with your FAL_KEY:")
        print("FAL_KEY=your_fal_api_key_here")
        return False
    
    # Check if FAL_KEY is set
    with open(env_file) as f:
        content = f.read()
        if "FAL_KEY" not in content or "your_fal_api_key_here" in content:
            print("✗ FAL_KEY not properly set in .env file")
            print("Please set your actual FAL API key in the .env file")
            return False
    
    print("✓ .env file found with FAL_KEY")
    return True

def start_server():
    """Start the FastAPI server"""
    print("\n🚀 Starting Video Scene Detection Web Application...")
    print("📱 Open your browser and go to: http://localhost:8001")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Change to backend directory
    backend_dir = Path(__file__).parent / "backend"
    os.chdir(backend_dir)
    
    # Start the server
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8001", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")

def main():
    """Main startup function"""
    print("🎬 Video Scene Detection Web Application")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check environment
    if not check_env_file():
        return
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()
