#!/usr/bin/env python3
"""
Storyboard → Gemini (google-genai) → ElevenLabs SDK (audio assets, with REST fallback)

Environment Variables (all optional with defaults):
  GEMINI_API_KEY         (required) Google Gemini API key
  ELEVEN_API_KEY         (required) ElevenLabs API key
  GEMINI_MODEL           (default: gemini-2.5-flash) Gemini model to use
  ELEVEN_API_URL         (default: https://api.elevenlabs.io/v1/sound-generation)
  DEFAULT_MUSIC_FORMAT   (default: mp3) Default format for music files
  DEFAULT_SFX_FORMAT     (default: wav) Default format for SFX files
  GEMINI_TEMPERATURE     (default: 0.6) Temperature for Gemini generation
  GEMINI_MAX_TOKENS      (default: 4096) Max tokens for Gemini
  RATE_LIMIT_SLEEP       (default: 0.7) Sleep between API calls
  MAX_RETRIES           (default: 4) Number of retry attempts
  RETRY_BACKOFF         (default: 1.6) Backoff multiplier for retries
  REQUEST_TIMEOUT       (default: 120) Request timeout in seconds
  DEFAULT_OUTPUT_DIR    (default: ./exports) Default output directory

Install:
  pip install google-genai elevenlabs requests python-slugify pydantic

Configuration can be customized programmatically:
  from storyboard_to_audio import create_custom_config
  config = create_custom_config(gemini_temperature=0.8, rate_limit_sleep=1.0)
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import requests
from slugify import slugify
from pydantic import BaseModel, Field, ValidationError, conint, field_validator, model_validator

# ------------------ Centralized Configuration ------------------
class AudioGenerationConfig(BaseModel):
    """Centralized configuration for audio generation pipeline using Pydantic"""
    
    # API Configuration
    gemini_api_key: str = Field("", description="Google Gemini API key (required)")
    eleven_api_key: str = Field("", description="ElevenLabs API key (required)")
    gemini_model: str = Field("gemini-2.5-flash", description="Gemini model to use")
    eleven_api_url: str = Field(
        "https://api.elevenlabs.io/v1/sound-generation", 
        description="ElevenLabs API endpoint"
    )
    
    # Audio Format Defaults
    default_music_format: str = Field("mp3", description="Default music file format")
    default_sfx_format: str = Field("wav", description="Default SFX file format")
    
    # Generation Parameters
    gemini_temperature: float = Field(
        1.0, 
        ge=0.0, 
        le=2.0, 
        description="Temperature for Gemini generation (0.0-2.0)"
    )
    gemini_max_tokens: int = Field(
        4096, 
        gt=0, 
        description="Maximum tokens for Gemini response"
    )
    rate_limit_sleep: float = Field(
        0.7, 
        ge=0.0, 
        description="Sleep duration between API calls (seconds)"
    )
    
    # Network & Retry Configuration
    max_retries: int = Field(4, gt=0, description="Number of retry attempts")
    retry_backoff: float = Field(1.6, gt=0.0, description="Backoff multiplier for retries")
    request_timeout: int = Field(120, gt=0, description="Request timeout in seconds")
    
    # Duration Constraints
    min_duration: int = Field(1, ge=1, le=60, description="Minimum audio duration")
    max_duration: int = Field(60, ge=1, le=60, description="Maximum audio duration")
    default_music_duration: int = Field(10, ge=1, le=60, description="Default music duration")
    default_sfx_duration: int = Field(5, ge=1, le=60, description="Default SFX duration")
    
    # ElevenLabs SDK Configuration
    optional_eleven_fields: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional fields for ElevenLabs API"
    )
    
    # Output Configuration
    default_output_dir: str = Field("./exports", description="Default output directory")
    
    class Config:
        """Pydantic configuration"""
        extra = "forbid"  # Prevent unknown fields
        validate_assignment = True  # Validate when fields are assigned
        
    @field_validator('optional_eleven_fields', mode='before')
    @classmethod
    def set_optional_eleven_fields(cls, v):
        """Ensure optional_eleven_fields has a default value"""
        if v is None:
            return {
                # "model_id": "eleven-sound-v1"  # Uncomment and modify as needed
            }
        return v
    
    @field_validator('default_music_format', 'default_sfx_format')
    @classmethod
    def validate_audio_formats(cls, v):
        """Validate audio format is supported"""
        valid_formats = {'mp3', 'wav', 'flac', 'aac', 'ogg'}
        if v.lower() not in valid_formats:
            raise ValueError(f"Audio format must be one of: {', '.join(valid_formats)}")
        return v.lower()
    
    @model_validator(mode='after')
    def validate_constraints(self):
        """Validate various constraints and requirements"""
        # Duration constraints validation
        if self.min_duration > self.max_duration:
            raise ValueError("min_duration must be <= max_duration")
        
        if not (self.min_duration <= self.default_music_duration <= self.max_duration):
            raise ValueError("default_music_duration must be between min_duration and max_duration")
            
        if not (self.min_duration <= self.default_sfx_duration <= self.max_duration):
            raise ValueError("default_sfx_duration must be between min_duration and max_duration")
        
        # API key validation (can be bypassed for testing or when explicitly disabled)
        skip_validation = (
            getattr(self, '_allow_empty_keys', False) or 
            os.getenv('SKIP_CONFIG_VALIDATION', '').lower() in ('true', '1', 'yes')
        )
        
        if not skip_validation:
            if not self.gemini_api_key:
                raise ValueError("gemini_api_key is required")
            if not self.eleven_api_key:
                raise ValueError("eleven_api_key is required")
        
        return self
    
    @classmethod
    def from_env(cls) -> 'AudioGenerationConfig':
        """Create configuration from environment variables with fallbacks"""
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            eleven_api_key=os.getenv("ELEVEN_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            eleven_api_url=os.getenv("ELEVEN_API_URL", "https://api.elevenlabs.io/v1/sound-generation"),
            default_music_format=os.getenv("DEFAULT_MUSIC_FORMAT", "mp3"),
            default_sfx_format=os.getenv("DEFAULT_SFX_FORMAT", "wav"),
            gemini_temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.6")),
            gemini_max_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "4096")),
            rate_limit_sleep=float(os.getenv("RATE_LIMIT_SLEEP", "0.7")),
            max_retries=int(os.getenv("MAX_RETRIES", "4")),
            retry_backoff=float(os.getenv("RETRY_BACKOFF", "1.6")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "120")),
            default_output_dir=os.getenv("DEFAULT_OUTPUT_DIR", "./exports"),
        )
    
    @classmethod
    def for_testing(cls, **overrides) -> 'AudioGenerationConfig':
        """Create a configuration suitable for testing with minimal validation"""
        defaults = {
            "gemini_api_key": "test-gemini-key",
            "eleven_api_key": "test-eleven-key",
            "rate_limit_sleep": 0.0,  # No delays in tests
            "max_retries": 1,  # Fewer retries in tests
        }
        defaults.update(overrides)
        
        # Create the instance and set the testing flag
        instance = cls(**defaults)
        # Use object.__setattr__ to bypass Pydantic validation for private field
        object.__setattr__(instance, '_allow_empty_keys', True)
        return instance

# Global configuration instance (with validation skipped for module import)
try:
    CONFIG = AudioGenerationConfig.from_env()
except ValidationError:
    # If validation fails during import (e.g., missing API keys), create a minimal config
    # This allows the module to be imported for testing or when keys will be provided later
    os.environ['SKIP_CONFIG_VALIDATION'] = 'true'
    CONFIG = AudioGenerationConfig.from_env()
    del os.environ['SKIP_CONFIG_VALIDATION']

def create_custom_config(**overrides) -> AudioGenerationConfig:
    """Create a custom configuration with specific overrides
    
    Example:
        config = create_custom_config(
            gemini_temperature=0.8,
            default_music_format="wav",
            rate_limit_sleep=1.0
        )
    """
    # Get base values from environment
    base_values = {
        "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
        "eleven_api_key": os.getenv("ELEVEN_API_KEY", ""),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "eleven_api_url": os.getenv("ELEVEN_API_URL", "https://api.elevenlabs.io/v1/sound-generation"),
        "default_music_format": os.getenv("DEFAULT_MUSIC_FORMAT", "mp3"),
        "default_sfx_format": os.getenv("DEFAULT_SFX_FORMAT", "wav"),
        "gemini_temperature": float(os.getenv("GEMINI_TEMPERATURE", "0.6")),
        "gemini_max_tokens": int(os.getenv("GEMINI_MAX_TOKENS", "4096")),
        "rate_limit_sleep": float(os.getenv("RATE_LIMIT_SLEEP", "0.7")),
        "max_retries": int(os.getenv("MAX_RETRIES", "4")),
        "retry_backoff": float(os.getenv("RETRY_BACKOFF", "1.6")),
        "request_timeout": int(os.getenv("REQUEST_TIMEOUT", "120")),
        "default_output_dir": os.getenv("DEFAULT_OUTPUT_DIR", "./exports"),
    }
    
    # Apply overrides
    base_values.update(overrides)
    
    # If API keys are empty, temporarily skip validation
    need_validation_skip = not base_values.get("gemini_api_key") or not base_values.get("eleven_api_key")
    
    if need_validation_skip:
        # Temporarily set environment variable to skip validation
        os.environ['SKIP_CONFIG_VALIDATION'] = 'true'
        try:
            config = AudioGenerationConfig(**base_values)
        finally:
            del os.environ['SKIP_CONFIG_VALIDATION']
        return config
    else:
        # Create and return new config (Pydantic will validate automatically)
        return AudioGenerationConfig(**base_values)

# ------------------ Schemas ------------------
class CueItem(BaseModel):
    prompt: str
    duration: conint(ge=1, le=60)  # Use static values to avoid evaluation issues
    
    def __init__(self, **data):
        super().__init__(**data)
        # Runtime validation using CONFIG
        if not (CONFIG.min_duration <= self.duration <= CONFIG.max_duration):
            raise ValueError(f"duration must be between {CONFIG.min_duration} and {CONFIG.max_duration}")

class PanelCue(BaseModel):
    panel_number: conint(ge=1)
    title: Optional[str] = None
    music: Optional[CueItem] = None
    sfx: Optional[CueItem] = None

class CueSheet(BaseModel):
    scene_title: Optional[str] = None
    panels: List[PanelCue]

# ------------------ Gemini prompt ------------------
GEMINI_SYSTEM = (
    "You score manga/anime storyboards into concise Audio Cue Sheets (JSON only). "
    "For each panel, optionally include 'music' and/or 'sfx' with short prompts (<=20 words) "
    "and practical durations (music ~8–12s, sfx ~3–6s). No copyrighted song refs. "
    "Silence is allowed by omitting the field."
)

GEMINI_USER = """TASK:
Convert the following storyboard into an Audio Cue Sheet.

SCHEMA (JSON ONLY):
{
  "scene_title": "string (optional)",
  "panels": [
    {
      "panel_number": 1,
      "title": "string (optional)",
      "music": { "prompt": "string", "duration": 12 },
      "sfx":   { "prompt": "string", "duration": 5 }
    }
  ]
}

STORYBOARD TEXT:
"""

# ------------------ Helpers ------------------
def read_storyboard_text(path: Optional[str]) -> str:
    return Path(path).read_text(encoding="utf-8") if path else sys.stdin.read()

def extract_json_block(s: str) -> str:
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    a = s.find("{"); b = s.rfind("}")
    return s[a:b+1] if a != -1 and b != -1 and b > a else s

def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def build_rest_payload(text: str, duration: int, fmt: str) -> Dict[str, Any]:
    payload = {"text": text, "duration": duration, "format": fmt}
    payload.update(CONFIG.optional_eleven_fields)
    return payload

def rest_request_with_retries(url: str, headers: Dict[str, str], payload: Dict[str, Any],
                              max_retries: Optional[int] = None, 
                              backoff: Optional[float] = None, 
                              timeout: Optional[int] = None) -> Tuple[bool, bytes, str]:
    max_retries = max_retries or CONFIG.max_retries
    backoff = backoff or CONFIG.retry_backoff
    timeout = timeout or CONFIG.request_timeout
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                return True, r.content, ""
            try:
                err_txt = json.dumps(r.json())
            except Exception:
                err_txt = r.text
            if r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait = backoff ** attempt
                print(f"[warn] ElevenLabs REST {r.status_code}; retrying in {wait:.1f}s ({attempt}/{max_retries})")
                time.sleep(wait); continue
            return False, b"", f"HTTP {r.status_code}: {err_txt}"
        except requests.RequestException as e:
            if attempt < max_retries:
                wait = backoff ** attempt
                print(f"[warn] network error {e}; retrying in {wait:.1f}s ({attempt}/{max_retries})")
                time.sleep(wait); continue
            return False, b"", f"Request failed: {e}"
    return False, b"", "Unknown error after retries"

def clip_filename(panel_num: int, kind: str, fmt: str, title: Optional[str]) -> str:
    base = f"panel{panel_num:02d}_{kind}"
    if title:
        base += f"_{slugify(title)[:30]}"
    return f"{base}.{fmt}"

def parse_panel_range(spec: str) -> List[int]:
    out = set()
    if not spec:
        return []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = min(int(a), int(b)), max(int(a), int(b))
            for i in range(lo, hi + 1):
                out.add(i)
        else:
            out.add(int(part))
    return sorted(out)

# ------------------ ElevenLabs SDK wrapper ------------------
class ElevenSDKWrapper:
    """
    Uses ElevenLabs Python SDK when available; otherwise falls back to REST.
    Tries multiple method names to be compatible across SDK versions.
    """
    def __init__(self, api_key: str, rest_url: str):
        self.api_key = api_key
        self.rest_url = rest_url
        self.client = None
        self.sdk_error_cls = Exception

        try:
            from elevenlabs.client import ElevenLabs  # official SDK client
            self.client = ElevenLabs(api_key=api_key)
            # best-effort: import a typed error if exposed
            try:
                from elevenlabs.api import Error as ElevenLabsError
                self.sdk_error_cls = ElevenLabsError
            except Exception:
                pass
        except Exception as e:
            self.client = None  # use REST fallback

    def generate_bytes(self, prompt: str, duration: int, fmt: str) -> Tuple[bool, bytes, str]:
        # 1) Try SDK paths if client present
        if self.client is not None:
            # Try a few likely namespaces; adjust if your SDK shows different names.
            candidates = [
                # sound effects style
                ("text_to_sound_effects", "convert"),
                ("text_to_sound_effects", "generate"),
                ("sound_effects", "generate"),
                # generic sound generation
                ("sound_generation", "generate"),
                ("sound_generation", "create"),
                # (future) music endpoints
                ("text_to_music", "generate"),
                ("music", "generate"),
            ]
            for ns, meth in candidates:
                try:
                    node = getattr(self.client, ns, None)
                    if node is None:
                        continue
                    fn = getattr(node, meth, None)
                    if fn is None:
                        continue
                    # Common parameter names vary; try the obvious ones first.
                    try_kwargs_list = [
                        dict(text=prompt, duration_seconds=duration, output_format=fmt),
                        dict(text=prompt, duration=duration, format=fmt),
                        dict(prompt=prompt, duration=duration, format=fmt),
                    ]
                    for kwargs in try_kwargs_list:
                        try:
                            data = fn(**kwargs)
                            # Some SDKs return bytes; others return objects with .content or .audio
                            if isinstance(data, (bytes, bytearray)):
                                return True, bytes(data), ""
                            for attr in ("content", "audio", "data"):
                                blob = getattr(data, attr, None)
                                if isinstance(blob, (bytes, bytearray)):
                                    return True, bytes(blob), ""
                            # If it returns a streaming iterator, join it:
                            if hasattr(data, "__iter__") and not isinstance(data, (bytes, bytearray, dict, list, str)):
                                chunks = []
                                for ch in data:
                                    if isinstance(ch, (bytes, bytearray)):
                                        chunks.append(bytes(ch))
                                    elif isinstance(ch, dict) and "audio" in ch:
                                        chunks.append(ch["audio"])
                                if chunks:
                                    return True, b"".join(chunks), ""
                            # Fallback: if it has .to_bytes()
                            if hasattr(data, "to_bytes"):
                                return True, data.to_bytes(), ""
                        except self.sdk_error_cls as e:
                            # continue to next variant; don't bail immediately
                            last_err = f"SDK error via {ns}.{meth}: {e}"
                        except TypeError:
                            # params mismatch; try next kwargs variant
                            continue
                except Exception as e:
                    last_err = f"SDK path {ns}.{meth} failed: {e}"
            # If all SDK paths failed, fall back to REST:
            # print(f"[info] falling back to REST. last SDK error: {last_err if 'last_err' in locals() else 'N/A'}")
        # 2) REST fallback
        headers = {"xi-api-key": self.api_key, "Content-Type": "application/json"}
        payload = build_rest_payload(prompt, duration, fmt)
        ok, content, err = rest_request_with_retries(self.rest_url, headers, payload)
        return ok, content, err

# ------------------ API Integration ------------------
def generate_audio_from_script(script: str, 
                              output_dir: str = "audio_assets",
                              panels_filter: Optional[str] = None,
                              audio_types: str = "music,sfx",
                              config: Optional[AudioGenerationConfig] = None) -> List[Dict[str, Any]]:
    """
    Generate audio files from a script string (API-compatible version)
    
    Args:
        script: The storyboard script text
        output_dir: Directory to save audio files (relative to backend/)
        panels_filter: Optional panel range filter like "1-6,9,12-14"
        audio_types: Types to generate: "music", "sfx", or "music,sfx"
        config: Optional custom configuration
        
    Returns:
        List of generated audio file information
    """
    if not script or not script.strip():
        raise ValueError("Script cannot be empty")
    
    # Use provided config or global CONFIG
    cfg = config or CONFIG
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    ensure_outdir(output_path)
    
    # ---- Gemini (google-genai) ----
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise RuntimeError("google-genai not installed. Run: pip install google-genai") from e

    # Create Gemini client
    client = genai.Client(api_key=cfg.gemini_api_key)
    prompt = GEMINI_USER + script.strip() + "\n\nJSON ONLY."
    
    # Generate cue sheet
    resp = client.models.generate_content(
        model=cfg.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=cfg.gemini_temperature,
            max_output_tokens=cfg.gemini_max_tokens,
            system_instruction=GEMINI_SYSTEM,
            response_mime_type="application/json"
        ),
    )
    
    raw_text = getattr(resp, "text", "") or ""
    json_text = extract_json_block(raw_text)

    try:
        cue_data = json.loads(json_text)
        cue_sheet = CueSheet(**cue_data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise RuntimeError(f"Failed to parse/validate Gemini JSON: {e}") from e

    # Filter panels if specified
    wanted = parse_panel_range(panels_filter) if panels_filter else None
    panels: List[PanelCue] = []
    seen = set()
    for p in cue_sheet.panels:
        if p.panel_number in seen:
            continue
        seen.add(p.panel_number)
        if wanted and p.panel_number not in wanted:
            continue
        panels.append(p)

    # ---- ElevenLabs generation ----
    kinds = [t.strip().lower() for t in audio_types.split(",") if t.strip() in ("music", "sfx")]
    sdk = ElevenSDKWrapper(api_key=cfg.eleven_api_key, rest_url=cfg.eleven_api_url)

    generated_files = []
    
    for p in panels:
        panel_files = {"panel_number": p.panel_number, "title": p.title, "files": []}
        
        # MUSIC
        if "music" in kinds and p.music:
            fname = clip_filename(p.panel_number, "music", cfg.default_music_format, p.title)
            fpath = output_path / fname
            
            ok, content, err = sdk.generate_bytes(
                prompt=p.music.prompt, 
                duration=p.music.duration, 
                fmt=cfg.default_music_format
            )
            
            if ok:
                fpath.write_bytes(content)
                panel_files["files"].append({
                    "type": "music",
                    "filename": fname,
                    "path": str(fpath),
                    "prompt": p.music.prompt,
                    "duration": p.music.duration,
                    "format": cfg.default_music_format
                })
            else:
                print(f"[error] panel {p.panel_number:02d} music: {err}")
            
            time.sleep(cfg.rate_limit_sleep)

        # SFX
        if "sfx" in kinds and p.sfx:
            fname = clip_filename(p.panel_number, "sfx", cfg.default_sfx_format, p.title)
            fpath = output_path / fname
            
            ok, content, err = sdk.generate_bytes(
                prompt=p.sfx.prompt, 
                duration=p.sfx.duration, 
                fmt=cfg.default_sfx_format
            )
            
            if ok:
                fpath.write_bytes(content)
                panel_files["files"].append({
                    "type": "sfx",
                    "filename": fname,
                    "path": str(fpath),
                    "prompt": p.sfx.prompt,
                    "duration": p.sfx.duration,
                    "format": cfg.default_sfx_format
                })
            else:
                print(f"[error] panel {p.panel_number:02d} sfx: {err}")
            
            time.sleep(cfg.rate_limit_sleep)
        
        if panel_files["files"]:  # Only add if files were generated
            generated_files.append(panel_files)

    return generated_files

# ------------------ Main CLI ------------------
def generate_audio_from_storyboard():
    parser = argparse.ArgumentParser(description="Storyboard → Gemini (google-genai) → ElevenLabs SDK/REST")
    parser.add_argument("--storyboard", type=str, default=None, help="Path to storyboard .txt (or pass via stdin).")
    parser.add_argument("--out", type=str, default=CONFIG.default_output_dir, help="Output directory.")
    parser.add_argument("--panels", type=str, default=None, help="Subset like '1-6,9,12-14'.")
    parser.add_argument("--only", type=str, default="music,sfx", help="'music', 'sfx', or 'music,sfx'.")

    parser.add_argument("--format-music", type=str, default=CONFIG.default_music_format)
    parser.add_argument("--format-sfx", type=str, default=CONFIG.default_sfx_format)
    parser.add_argument("--override-duration-music", type=int, default=None)
    parser.add_argument("--override-duration-sfx", type=int, default=None)

    parser.add_argument("--gemini-temp", type=float, default=CONFIG.gemini_temperature)
    parser.add_argument("--gemini-max-tokens", type=int, default=CONFIG.gemini_max_tokens)
    parser.add_argument("--rate-limit-sleep", type=float, default=CONFIG.rate_limit_sleep)

    args = parser.parse_args()

    # Pydantic validates automatically during instantiation, but we can catch any issues here
    try:
        # Just access the CONFIG to trigger any lazy validation
        _ = CONFIG.gemini_api_key, CONFIG.eleven_api_key
    except ValidationError as e:
        print(f"ERROR: Configuration validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    storyboard = read_storyboard_text(args.storyboard).strip()
    if not storyboard:
        print("ERROR: no storyboard text provided.", file=sys.stderr); sys.exit(1)

    # ---- Gemini (google-genai) ----
    try:
        from google import genai
        from google.genai import types
    except Exception:
        print("ERROR: google-genai not installed. Run: pip install google-genai", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=CONFIG.gemini_api_key)
    prompt = GEMINI_USER + storyboard + "\n\nJSON ONLY."
    resp = client.models.generate_content(
        model=CONFIG.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=args.gemini_temp,
            max_output_tokens=args.gemini_max_tokens,
            system_instruction=GEMINI_SYSTEM,
            response_mime_type="application/json"
        ),
    )
    raw_text = getattr(resp, "text", "") or ""
    json_text = extract_json_block(raw_text)

    try:
        cue_data = json.loads(json_text)
        cue_sheet = CueSheet(**cue_data)
    except (json.JSONDecodeError, ValidationError) as e:
        print("\n[Gemini raw output]\n", raw_text[:1500], "\n", file=sys.stderr)
        raise RuntimeError(f"Failed to parse/validate Gemini JSON: {e}")

    wanted = parse_panel_range(args.panels) if args.panels else None
    panels: List[PanelCue] = []
    seen = set()
    for p in cue_sheet.panels:
        if p.panel_number in seen:
            continue
        seen.add(p.panel_number)
        if wanted and p.panel_number not in wanted:
            continue
        panels.append(p)

    # ---- ElevenLabs generation (SDK w/ REST fallback) ----
    outdir = Path(args.out); ensure_outdir(outdir)
    kinds = [t.strip().lower() for t in args.only.split(",") if t.strip() in ("music", "sfx")]

    sdk = ElevenSDKWrapper(api_key=CONFIG.eleven_api_key, rest_url=CONFIG.eleven_api_url)

    total, failures = 0, []
    for p in panels:
        # MUSIC
        if "music" in kinds and p.music:
            dur = args.override_duration_music or p.music.duration
            fname = clip_filename(p.panel_number, "music", args.format_music, p.title)
            fpath = outdir / fname
            print(f"[gen] panel {p.panel_number:02d} music ({dur}s, {args.format_music}) → {fname}")
            ok, content, err = sdk.generate_bytes(prompt=p.music.prompt, duration=dur, fmt=args.format_music)
            if ok:
                fpath.write_bytes(content); total += 1
            else:
                print(f"[error] panel {p.panel_number:02d} music: {err}")
                failures.append({"panel": p.panel_number, "kind": "music", "error": err})
            time.sleep(args.rate_limit_sleep)

        # SFX
        if "sfx" in kinds and p.sfx:
            dur = args.override_duration_sfx or p.sfx.duration
            fname = clip_filename(p.panel_number, "sfx", args.format_sfx, p.title)
            fpath = outdir / fname
            print(f"[gen] panel {p.panel_number:02d} sfx ({dur}s, {args.format_sfx}) → {fname}")
            ok, content, err = sdk.generate_bytes(prompt=p.sfx.prompt, duration=dur, fmt=args.format_sfx)
            if ok:
                fpath.write_bytes(content); total += 1
            else:
                print(f"[error] panel {p.panel_number:02d} sfx: {err}")
                failures.append({"panel": p.panel_number, "kind": "sfx", "error": err})
            time.sleep(args.rate_limit_sleep)

    print(f"\nDone. Wrote {total} files to: {outdir.resolve()}")
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f" - Panel {f['panel']:02d} {f['kind']}: {f['error']}")

if __name__ == "__main__":
    generate_audio_from_storyboard()