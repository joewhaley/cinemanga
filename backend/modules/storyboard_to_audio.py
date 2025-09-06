#!/usr/bin/env python3
"""
Storyboard → Gemini (google-genai) → ElevenLabs SDK (audio assets, with REST fallback)

Env:
  GEMINI_API_KEY   (Google Gemini API key)
  ELEVEN_API_KEY   (ElevenLabs API key)

Install:
  pip install google-genai elevenlabs requests python-slugify pydantic
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
from pydantic import BaseModel, Field, ValidationError, conint

# ------------------ Config ------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
ELEVEN_API_URL = os.getenv("ELEVEN_API_URL", "https://api.elevenlabs.io/v1/sound-generation")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "")

DEFAULT_MUSIC_FORMAT = "mp3"   # e.g., "mp3"
DEFAULT_SFX_FORMAT   = "wav"   # e.g., "wav"

# If your ElevenLabs SDK/tenant needs a specific model id, add it here:
OPTIONAL_ELEVEN_FIELDS: Dict[str, Any] = {
    # "model_id": "eleven-sound-v1"
}

# ------------------ Schemas ------------------
class CueItem(BaseModel):
    prompt: str
    duration: conint(ge=1, le=60)

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
    payload.update(OPTIONAL_ELEVEN_FIELDS)
    return payload

def rest_request_with_retries(url: str, headers: Dict[str, str], payload: Dict[str, Any],
                              max_retries=4, backoff=1.6, timeout=120) -> Tuple[bool, bytes, str]:
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

# ------------------ Main ------------------
def main():
    parser = argparse.ArgumentParser(description="Storyboard → Gemini (google-genai) → ElevenLabs SDK/REST")
    parser.add_argument("--storyboard", type=str, default=None, help="Path to storyboard .txt (or pass via stdin).")
    parser.add_argument("--out", type=str, default="./exports", help="Output directory.")
    parser.add_argument("--panels", type=str, default=None, help="Subset like '1-6,9,12-14'.")
    parser.add_argument("--only", type=str, default="music,sfx", help="'music', 'sfx', or 'music,sfx'.")

    parser.add_argument("--format-music", type=str, default=DEFAULT_MUSIC_FORMAT)
    parser.add_argument("--format-sfx", type=str, default=DEFAULT_SFX_FORMAT)
    parser.add_argument("--override-duration-music", type=int, default=None)
    parser.add_argument("--override-duration-sfx", type=int, default=None)

    parser.add_argument("--gemini-temp", type=float, default=0.6)
    parser.add_argument("--gemini-max-tokens", type=int, default=4096)
    parser.add_argument("--rate-limit-sleep", type=float, default=0.7)

    args = parser.parse_args()

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set.", file=sys.stderr); sys.exit(1)
    if not ELEVEN_API_KEY:
        print("ERROR: ELEVEN_API_KEY not set.", file=sys.stderr); sys.exit(1)

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

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = GEMINI_USER + storyboard + "\n\nJSON ONLY."
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
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

    sdk = ElevenSDKWrapper(api_key=ELEVEN_API_KEY, rest_url=ELEVEN_API_URL)

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
    main()