import json
import base64
import os
import uuid
import shutil
import asyncio
import subprocess
from pathlib import Path
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips, ImageClip
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import cv2
import time
import httpx
from typing import Annotated, Dict
from elevenlabs import ElevenLabs
import numpy as np

load_dotenv()

MODEL_NAME = 'gemini-2.5-flash'  # gemini-2.5-flash, gemini-2.0-flash-lite, gemini-3-flash-preview
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)
CENTRAL_BASE_URL = "http://127.0.0.1:8002"  # Base URL for all central API calls
CENTRAL_API_URL = f"{CENTRAL_BASE_URL}/api/kits/simple"  # Kit upload endpoint

VOICE_MAP = {
    "Sarah": "EXAVITQu4vr4xnSDxMaL",
    "Enniah": "WHaUUVTDq47Yqc9aDbkH",
    "Patrick": "9Ft9sm9dzvprPILZmLJl",
    "Allison": "xctasy8XvGp2cVO9HL9k",
}

PROMPT = """You are an expert tutorial creator analyzing a screen recording to generate professional voiceover narrations.

CONTEXT:
You have a screen recording video showing a user interacting with a web application. Click events with precise timestamps have been detected and are listed below.

YOUR TASK:
1. Watch the video carefully and observe what happens at each click timestamp
2. For each click event, create a narration that:
   - Describes what UI element is being clicked (button, link, dropdown, input field, etc.)
   - Explains what action or result occurs after the click
   - Uses natural, conversational language suitable for text-to-speech voiceover
   - Guides the viewer as if you're teaching them step-by-step

NARRATION REQUIREMENTS:
- Language: {language}
- Tone: {style}
- Length: 1-2 sentences per click (concise but informative)
- Style: Use active voice and action-oriented language
  - Good: "Click the Save button to store your changes"
  - Bad: "The Save button is clicked and changes are stored"
- Focus on WHAT is clicked and WHY (the outcome/purpose)

DETECTED CLICK EVENTS:
{click_summary}

OUTPUT FORMAT:
- One narration per click in chronological order
- Include an overall summary of what the user accomplished in this tutorial

EXAMPLE NARRATION:
Instead of: "User clicks on the button"
Write: "Click the Submit button to send your form data to the server"

Instead of: "The dropdown menu is opened"
Write: "Open the dropdown menu to select your preferred language"

Now analyze the video and create professional tutorial narrations for each click event."""

# Job tracking system
jobs: Dict[str, dict] = {}  # {job_id: {status, progress, session_id, error, result}}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schemas ---

class Click(BaseModel):
    ui_element: Annotated[str, Field(..., description="The specific UI element that the user interacted with (e.g., button, icon, input field, calendar date).")]
    action_result: Annotated[str, Field(..., description="The observable result or outcome of the interaction triggered by the user action.")]
    narration: Annotated[str, Field(..., description="Natural, spoken narration describing the action in clear, conversational language suitable for text-to-speech voiceover (e.g., 'Click on Save to store your changes').")]

class Analysis(BaseModel):
    detected_clicks: Annotated[list[Click], Field(..., description="Chronologically ordered list of all significant user interactions detected in the screen recording.")]
    summary: Annotated[str, Field(..., description="High-level summary describing the user's overall activity and intent during the session.")]


# --- LLM Setup ---

llm = ChatGoogleGenerativeAI(model=MODEL_NAME)
structured_llm = llm.with_structured_output(Analysis)
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None


# --- Helper: TTS ---

def generate_tts_audio(text: str, voice_name: str, output_path: str) -> float:
    """
    Generate TTS audio via ElevenLabs API.

    Args:
        text: The narration text to convert to speech.
        voice_name: Voice identifier (e.g., "Sarah", "Enniah", "Patrick", "Allison").
                    Falls back to "Sarah" if not found in VOICE_MAP.
        output_path: Absolute path where the MP3 file will be saved.

    Returns:
        Duration of the generated audio in seconds.

    Raises:
        RuntimeError: If ELEVENLABS_API_KEY is not configured.
    """
    from mutagen.mp3 import MP3

    if not eleven_client:
        raise RuntimeError("ElevenLabs API key not configured")

    voice_id = VOICE_MAP.get(voice_name, VOICE_MAP["Sarah"])
    audio_generator = eleven_client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    with open(output_path, "wb") as f:
        for chunk in audio_generator:
            f.write(chunk)

    # Get exact duration using mutagen (pure Python)
    audio = MP3(output_path)
    return audio.info.length

# --- Zoom helpers (from zoom_testing.py) ---

ZOOM_SCALE = 1.15
ZOOM_IN_DURATION = 0.4
ZOOM_OUT_DURATION = 0.5
EASING_POWER = 1.5


def ease_in_out(x):
    """Smooth easing curve for natural zoom animation."""
    return x ** EASING_POWER


def zoom_at_point(frame, scale, cx, cy):
    """
    Apply zoom centered around a specific (x, y) point.
    Crops a smaller area based on scale and resizes back to original resolution.
    """
    h, w = frame.shape[:2]
    new_w = int(w / scale)
    new_h = int(h / scale)
    
    x1 = int(cx - new_w / 2)
    y1 = int(cy - new_h / 2)
    
    # Clamp to frame boundaries
    x1 = max(0, min(x1, w - new_w))
    y1 = max(0, min(y1, h - new_h))
    
    cropped = frame[y1:y1 + new_h, x1:x1 + new_w]
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized


def make_zoom_clip(frame_rgb, cx, cy, narration_duration, fps, annotate=False):
    """
    Create an animated clip that zooms in on (cx, cy), holds during narration, then zooms out.
    
    Structure:
        [zoom_in: 0.25s] -> [hold_zoomed: narration_duration] -> [zoom_out: 0.35s]
    """
    total_duration = ZOOM_IN_DURATION + narration_duration + ZOOM_OUT_DURATION
    
    # Optionally annotate the frame
    if annotate:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        rect_size = 60
        x1 = int(cx - rect_size // 2)
        y1 = int(cy - rect_size // 2)
        x2 = int(cx + rect_size // 2)
        y2 = int(cy + rect_size // 2)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    def make_frame(t):
        # Phase 1: Zoom in
        if t <= ZOOM_IN_DURATION:
            progress = t / ZOOM_IN_DURATION if ZOOM_IN_DURATION > 0 else 1.0
            scale = 1.0 + (ZOOM_SCALE - 1.0) * ease_in_out(progress)
        # Phase 2: Hold zoomed
        elif t <= ZOOM_IN_DURATION + narration_duration:
            scale = ZOOM_SCALE
        # Phase 3: Zoom out
        else:
            elapsed = t - ZOOM_IN_DURATION - narration_duration
            progress = elapsed / ZOOM_OUT_DURATION if ZOOM_OUT_DURATION > 0 else 1.0
            progress = min(progress, 1.0)
            scale = ZOOM_SCALE - (ZOOM_SCALE - 1.0) * ease_in_out(progress)
        
        if scale > 1.001:
            return zoom_at_point(frame_rgb, scale, cx, cy)
        return frame_rgb
    
    from moviepy.editor import VideoClip
    clip = VideoClip(make_frame, duration=total_duration)
    clip = clip.set_fps(fps)
    return clip


def render_voice_only(session_dir: Path, video_filename: str, video_name: str, narrations: list, click_data: list) -> str:
    """
    Render a 'voice-only' video with zoom animation at each click point.
    
    Logic:
    1. Play video normally until a click happens
    2. Zoom in on cursor coordinates when narration starts
    3. Hold zoomed while narration plays
    4. Zoom out when narration ends
    5. Resume normal video playback
    6. Repeat for each click
    """
    
    video_path = session_dir / video_filename
    output_filename = f"{video_name}_voice_only.mp4"
    output_path = session_dir / output_filename
    
    # Get zoom level from click_data for y_offset
    zoom_level = click_data[0].get('zoomLevel', 1.0) if click_data else 1.0
    y_offset = 114 * zoom_level
    
    print(f"[RENDER] Starting voice-only render for {video_name}")
    print(f"[RENDER] Input: {video_path}")
    print(f"[RENDER] Output: {output_path}")
    print(f"[RENDER] Narrations: {len(narrations)}")
    
    # Fix WebM file metadata
    fixed_video_path = session_dir / f"{video_name}_fixed.webm"
    print(f"[RENDER] Fixing WebM metadata...")
    try:
        from moviepy.config import get_setting
        ffmpeg_binary = get_setting("FFMPEG_BINARY")
        
        subprocess.run([
            ffmpeg_binary, '-i', str(video_path),
            '-c', 'copy',
            '-y',
            str(fixed_video_path)
        ], check=True, capture_output=True)
        
        video_path = fixed_video_path
        print(f"[RENDER] Fixed video: {fixed_video_path}")
    except Exception as e:
        print(f"[RENDER] Warning: Could not fix WebM metadata: {e}")
        print(f"[RENDER] Continuing with original file...")
    
    # Load video
    video = VideoFileClip(str(video_path), audio=True, fps_source='fps')
    video_duration = video.duration
    
    print(f"[RENDER] Video duration: {video_duration}s")
    
    # Build segments
    segments = []
    current_time = 0
    
    for i, narr in enumerate(narrations):
        click_time = narr['click_time']
        narration_duration = narr['audio_duration']
        
        # Get cursor coordinates
        click = click_data[i] if i < len(click_data) else {}
        cx = int(click.get('clientX', 0))
        cy = int(click.get('clientY', 0)) + int(y_offset)
        
        print(f"[RENDER] Processing click {i+1}/{len(narrations)} at {click_time:.2f}s, narration: {narration_duration:.2f}s, cursor: ({cx}, {cy})")
        
        # 1. Add normal video segment from current_time to click_time
        if click_time > current_time:
            segment = video.subclip(current_time, click_time)
            segments.append(segment)
            print(f"  → Segment {len(segments)}: Normal video {current_time:.2f}s → {click_time:.2f}s")
        
        # 2. Create zoom animation clip (zoom in → hold → zoom out)
        frame = video.get_frame(click_time)
        zoom_clip = make_zoom_clip(frame, cx, cy, narration_duration, video.fps, annotate=False)
        
        # 3. Add narration audio — offset to start after zoom-in completes
        audio_file = session_dir / f"{video_name}_narration_{i}.mp3"
        if audio_file.exists():
            narration_audio = AudioFileClip(str(audio_file))
            narration_audio = narration_audio.set_start(ZOOM_IN_DURATION)
            zoom_clip = zoom_clip.set_audio(CompositeAudioClip([narration_audio]))
            print(f"  → Segment {len(segments)+1}: Zoom clip {ZOOM_IN_DURATION:.2f}s + {narration_duration:.2f}s + {ZOOM_OUT_DURATION:.2f}s with narration")
        else:
            print(f"  → WARNING: Audio file not found: {audio_file}")
        
        segments.append(zoom_clip)
        
        # 4. Update current_time to resume after the click point
        current_time = click_time
    
    # 5. Add remaining video after last click
    if current_time < video_duration:
        segment = video.subclip(current_time, video_duration)
        segments.append(segment)
        print(f"[RENDER] Segment {len(segments)}: Final video {current_time:.2f}s → {video_duration:.2f}s")
    
    # Concatenate all segments
    print(f"[RENDER] Concatenating {len(segments)} segments...")
    final_video = concatenate_videoclips(segments, method="compose")
    
    # Write output
    print(f"[RENDER] Writing output video...")
    final_video.write_videofile(
        str(output_path),
        codec='libx264',
        audio_codec='aac',
        temp_audiofile=str(session_dir / 'temp-audio.m4a'),
        remove_temp=True,
        logger=None
    )
    
    # Close clips
    video.close()
    final_video.close()
    for seg in segments:
        if hasattr(seg, 'close'):
            seg.close()
    
    print(f"[RENDER] Voice-only render complete: {output_filename}")
    return output_filename


def render_voice_with_annotations(session_dir: Path, video_filename: str, video_name: str, narrations: list, click_data: list) -> str:
    
    video_path = session_dir / video_filename
    output_filename = f"{video_name}_with_annotations.mp4"
    output_path = session_dir / output_filename

    # Get zoom level from click_data
    zoom_level = click_data[0].get('zoomLevel', 1.0) if click_data else 1.0
    
    # Calculate y_offset based on zoom level
    y_offset = 114 * zoom_level    
    
    print(f"[RENDER] Starting voice with annotation render for {video_name}")
    print(f"[RENDER] Input: {video_path}")
    print(f"[RENDER] Output: {output_path}")
    print(f"[RENDER] Narrations: {len(narrations)}")
    
    # Fix WebM file metadata
    fixed_video_path = session_dir / f"{video_name}_fixed.webm"
    print(f"[RENDER] Fixing WebM metadata...")
    try:
        from moviepy.config import get_setting
        ffmpeg_binary = get_setting("FFMPEG_BINARY")
        
        subprocess.run([
            ffmpeg_binary, '-i', str(video_path),
            '-c', 'copy',
            '-y',
            str(fixed_video_path)
        ], check=True, capture_output=True)
        
        video_path = fixed_video_path
        print(f"[RENDER] Fixed video: {fixed_video_path}")
    except Exception as e:
        print(f"[RENDER] Warning: Could not fix WebM metadata: {e}")
        print(f"[RENDER] Continuing with original file...")
    
    # Load video
    video = VideoFileClip(str(video_path), audio=True, fps_source='fps')
    video_duration = video.duration
    
    print(f"[RENDER] Video duration: {video_duration}s")
    
    # Build segments
    segments = []
    current_time = 0
    
    for i, narr in enumerate(narrations):
        click_time = narr['click_time']
        narration_duration = narr['audio_duration']
        
        # Get click coordinates from click_data
        click = click_data[i] if i < len(click_data) else {}
        cx = int(click.get('clientX', 0))
        cy = int(click.get('clientY', 0)) + int(y_offset)
        
        print(f"[RENDER] Processing click {i+1}/{len(narrations)} at {click_time:.2f}s, narration: {narration_duration:.2f}s, cursor: ({cx}, {cy})")
        
        # 1. Add normal video segment from current_time to click_time
        if click_time > current_time:
            segment = video.subclip(current_time, click_time)
            segments.append(segment)
            print(f"  → Segment {len(segments)}: Normal video {current_time:.2f}s → {click_time:.2f}s")
        
        # 2. Create zoom animation clip with annotation (zoom in → hold → zoom out)
        frame = video.get_frame(click_time)
        zoom_clip = make_zoom_clip(frame, cx, cy, narration_duration, video.fps, annotate=True)
        
        # 3. Add narration audio — offset to start after zoom-in completes
        audio_file = session_dir / f"{video_name}_narration_{i}.mp3"
        if audio_file.exists():
            narration_audio = AudioFileClip(str(audio_file))
            narration_audio = narration_audio.set_start(ZOOM_IN_DURATION)
            zoom_clip = zoom_clip.set_audio(CompositeAudioClip([narration_audio]))
            print(f"  → Segment {len(segments)+1}: Zoom clip {ZOOM_IN_DURATION:.2f}s + {narration_duration:.2f}s + {ZOOM_OUT_DURATION:.2f}s with annotation + narration")
        else:
            print(f"  → WARNING: Audio file not found: {audio_file}")
        
        segments.append(zoom_clip)
        
        # 4. Update current_time to resume after the click point
        current_time = click_time
    
    # 5. Add remaining video after last click
    if current_time < video_duration:
        segment = video.subclip(current_time, video_duration)
        segments.append(segment)
        print(f"[RENDER] Segment {len(segments)}: Final video {current_time:.2f}s → {video_duration:.2f}s")
    
    # Concatenate all segments
    print(f"[RENDER] Concatenating {len(segments)} segments...")
    final_video = concatenate_videoclips(segments, method="compose")
    
    # Write output
    print(f"[RENDER] Writing output video...")
    final_video.write_videofile(
        str(output_path),
        codec='libx264',
        audio_codec='aac',
        temp_audiofile=str(session_dir / 'temp-audio.m4a'),
        remove_temp=True,
        logger=None
    )
    
    # Close clips
    video.close()
    final_video.close()
    for seg in segments:
        if hasattr(seg, 'close'):
            seg.close()
    
    print(f"[RENDER] Voice-only render complete: {output_filename}")
    return output_filename


@app.get('/')
async def home():
    """
    Home endpoint for server health check.

    Returns:
        JSON with status and title.
    """
    return {
        'status': 'OK',
        'title': 'Testing Server'
    }


# Background task for processing analysis
async def process_analysis_job(
    job_id: str,
    session_id: str,
    video_bytes: bytes,
    video_filename: str,
    click_data: list,
    voice: str,
    style: str,
    language: str,
    video_content_type: str
):
    """
    Background task that processes the video analysis.

    Args:
        job_id: Unique job identifier.
        session_id: Unique session identifier.
        video_bytes: Raw bytes of the uploaded video.
        video_filename: Name of the uploaded video file.
        click_data: List of click event dicts.
        voice: Voice name for TTS.
        style: Narration style.
        language: Narration language.
        video_content_type: MIME type of the video.

    Returns:
        None. Updates the jobs dict with progress and results.
    """
    session_dir = SESSIONS_DIR / session_id
    
    try:
        # Update: Saving video
        jobs[job_id].update({"status": "saving", "progress": 10, "message": "Saving video file..."})
        
        video_path = session_dir / video_filename
        video_path.write_bytes(video_bytes)
        print(f"[JOB {job_id}] Video saved to: {video_path}")
        
        video_name = Path(video_filename).stem.replace(" ", "_")
        print(f"[JOB {job_id}] Video name for narrations: {video_name}")
        
        # Calculate/normalize click times
        from datetime import datetime

        def parse_iso(ts: str | None):
            if not ts:
                return None
            try:
                return datetime.fromisoformat(ts.replace('Z', '+00:00'))
            except Exception:
                return None

        if click_data and len(click_data) > 0:
            # Establish the earliest timestamp (if any) for fallback calculations
            parsed_ts = [parse_iso(c.get('timestamp')) for c in click_data if c.get('timestamp')]
            first_ts = min(parsed_ts) if parsed_ts else None

            click_time_from_recorder = 0
            click_time_from_ts = 0
            click_time_missing = 0

            for click in click_data:
                ct = click.get('clickTime')
                if isinstance(ct, (int, float)):
                    click['clickTime'] = round(max(0.0, float(ct)), 3)
                    click_time_from_recorder += 1
                    continue

                ts = parse_iso(click.get('timestamp'))
                if ts and first_ts:
                    click['clickTime'] = round((ts - first_ts).total_seconds(), 3)
                    click_time_from_ts += 1
                else:
                    click['clickTime'] = 0.0
                    click_time_missing += 1

            # Sort by clickTime to ensure chronological order
            click_data.sort(key=lambda c: c.get('clickTime', 0.0))

            print(
                f"[JOB {job_id}] Normalized clickTime for {len(click_data)} clicks "
                f"(recorder={click_time_from_recorder}, timestamp_fallback={click_time_from_ts}, missing={click_time_missing})"
            )
        else:
            print(f"[JOB {job_id}] No click data provided")

        # Update: Analyzing with Gemini
        jobs[job_id].update({"status": "analyzing", "progress": 20, "message": "AI analyzing video..."})
        
        print(f"[JOB {job_id}] Encoding video for Gemini...")
        video_b64 = base64.standard_b64encode(video_bytes).decode("utf-8")
        print(f"[JOB {job_id}] Base64 encoded: {len(video_b64)} chars")

        click_summary = "\n".join(
            f"  Click {i+1}: at t={c.get('clickTime', 'unknown')}s, "
            f"screen=({c.get('screenX', '?')}, {c.get('screenY', '?')}), "
            f"client=({c.get('clientX', '?')}, {c.get('clientY', '?')}), "
            f"target=<{c.get('target', '?')}>, url={c.get('url', '?')}"
            for i, c in enumerate(click_data)
        )

        prompt = PROMPT.format(
            language=language,
            style=style,
            click_summary=click_summary
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "media",
                    "mime_type": video_content_type or "video/webm",
                    "data": video_b64,
                },
            ]
        )

        print(f"[JOB {job_id}] Sending to Gemini for analysis...")
        analysis: Analysis = await structured_llm.ainvoke([message])
        print(f"[JOB {job_id}] Gemini analysis complete. Detected {len(analysis.detected_clicks)} clicks")

        # Update: Generating TTS
        jobs[job_id].update({"status": "generating_tts", "progress": 50, "message": f"Generating voice narrations (0/{len(analysis.detected_clicks)})..."})
        
        print(f"[JOB {job_id}] Starting TTS generation for {len(analysis.detected_clicks)} narrations...")
        narrations = []
        num = min(len(analysis.detected_clicks), len(click_data))

        for i in range(num):
            click = click_data[i]
            detected = analysis.detected_clicks[i]
            audio_path = str(session_dir / f"{video_name}_narration_{i}.mp3")
            
            # Get click time from calculated clickTime in click_data
            click_time_seconds = float(click.get("clickTime", 0.0))
            
            # Update progress for each TTS generation
            progress = 50 + int((i / num) * 40)  # 50% to 90%
            jobs[job_id].update({"progress": progress, "message": f"Generating voice narrations ({i+1}/{num})..."})
            
            print(f"[JOB {job_id}] Generating TTS {i+1}/{num} at {click_time_seconds}s: {detected.narration[:50]}...")
            duration = generate_tts_audio(detected.narration, voice, audio_path)
            print(f"[JOB {job_id}] TTS {i+1} complete. Duration: {duration}s")

            narrations.append({
                "index": i,
                "click_time": click_time_seconds,
                "ui_element": detected.ui_element,
                "action_result": detected.action_result,
                "narration_text": detected.narration,
                "audio_duration": duration,
            })

        # Update: Saving metadata
        jobs[job_id].update({"status": "finalizing", "progress": 95, "message": "Finalizing analysis..."})
        
        # Save session metadata
        meta = {
            "session_id": session_id,
            "video_filename": video_filename,
            "video_name": video_name,
            "voice": voice,
            "style": style,
            "language": language,
            "summary": analysis.summary,
            "narrations": narrations,
            "click_data": click_data,
        }
        (session_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[JOB {job_id}] Session metadata saved")
        print(f"[JOB {job_id}] Analysis complete! Session: {session_id}")

        # Mark as complete
        jobs[job_id].update({
            "status": "complete",
            "progress": 100,
            "message": "Analysis complete!",
            "result": {
                "session_id": session_id,
                "summary": analysis.summary,
                "narrations": narrations,
            }
        })

    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"[JOB {job_id}] ERROR: {error_msg}")
        print(f"[JOB {job_id}] Traceback:\n{error_trace}")
        
        jobs[job_id].update({
            "status": "error",
            "progress": 0,
            "message": f"Error: {error_msg}",
            "error": error_msg,
            "traceback": error_trace
        })


@app.post('/analyze')
async def analyze(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    clicks: str = Form(...),
    voice: str = Form("Aria"),
    style: str = Form("Professional"),
    language: str = Form("English"),
):
    """
    Start video analysis job and return job_id immediately.

    Args:
        background_tasks: FastAPI background task manager.
        video: Uploaded video file (form-data).
        clicks: JSON string of click data (form-data).
        voice: Voice name for TTS (form-data).
        style: Narration style (form-data).
        language: Narration language (form-data).

    Returns:
        JSON with job_id, session_id, and status message.
    """
    
    # Generate IDs
    job_id = uuid.uuid4().hex
    session_id = uuid.uuid4().hex
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True)
    
    print(f"[ANALYZE] Created job: {job_id}, session: {session_id}")
    
    # Read video data before starting background task
    video_bytes = await video.read()
    video_filename = video.filename or "input.webm"
    click_data = json.loads(clicks)
    
    print(f"[ANALYZE] Video size: {len(video_bytes)} bytes, {len(click_data)} clicks")
    
    # Initialize job status
    jobs[job_id] = {
        "job_id": job_id,
        "session_id": session_id,
        "status": "queued",
        "progress": 0,
        "message": "Job queued...",
        "created_at": str(Path(session_dir).stat().st_ctime) if session_dir.exists() else "",
    }
    
    # Start background processing
    background_tasks.add_task(
        process_analysis_job,
        job_id=job_id,
        session_id=session_id,
        video_bytes=video_bytes,
        video_filename=video_filename,
        click_data=click_data,
        voice=voice,
        style=style,
        language=language,
        video_content_type=video.content_type
    )
    
    print(f"[ANALYZE] Job {job_id} started in background")
    
    return {
        "status": "ok",
        "job_id": job_id,
        "session_id": session_id,
        "message": "Analysis job started. Use /status/{job_id} to check progress."
    }


@app.get('/status/{job_id}')
async def get_job_status(job_id: str):
    """
    Check the status of an analysis job.

    Args:
        job_id: The job identifier to check.

    Returns:
        JSON with status, progress, message, result (if complete), error (if failed).
    """
    if job_id not in jobs:
        return {"status": "not_found", "message": "Job ID not found"}
    
    return jobs[job_id]


@app.post('/render/{session_id}')
async def render(
    session_id: str, 
    mode: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """
    Start video rendering job and return job_id immediately.

    Args:
        session_id: The session identifier to render.
        mode: Render mode ('voice-only' or 'full').
        background_tasks: FastAPI background task manager.

    Returns:
        JSON with job_id, session_id, mode, and status message.
    """
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        return {"status": "error", "message": "Session not found. Run /analyze first."}

    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        return {"status": "error", "message": "Session metadata not found."}
    
    meta = json.loads(meta_path.read_text())
    
    # Generate job ID
    job_id = uuid.uuid4().hex
    
    print(f"[RENDER] Created render job: {job_id} for session: {session_id}, mode: {mode}")
    
    # Initialize job status
    jobs[job_id] = {
        "job_id": job_id,
        "session_id": session_id,
        "status": "queued",
        "progress": 0,
        "message": f"Render job queued ({mode})...",
        "mode": mode,
    }
    
    # Start background rendering
    if background_tasks:
        background_tasks.add_task(
            process_render_job,
            job_id=job_id,
            session_id=session_id,
            session_dir=session_dir,
            meta=meta,
            mode=mode
        )
    
    print(f"[RENDER] Job {job_id} started in background")
    
    return {
        "status": "ok",
        "job_id": job_id,
        "session_id": session_id,
        "mode": mode,
        "message": f"Render job started. Use /status/{job_id} to check progress."
    }


# Background task for processing render job
async def process_render_job(
    job_id: str,
    session_id: str,
    session_dir: Path,
    meta: dict,
    mode: str
):
    """
    Background task for rendering video.

    Args:
        job_id: Unique job identifier.
        session_id: Unique session identifier.
        session_dir: Path to the session directory.
        meta: Metadata dict for the session.
        mode: Render mode ('voice-only' or 'full').

    Returns:
        None. Updates the jobs dict with progress and results.
    """
    try:
        jobs[job_id].update({"status": "rendering", "progress": 10, "message": f"Starting {mode} render..."})
        
        if mode == "voice-only":
            jobs[job_id].update({"progress": 30, "message": "Processing video segments..."})
            
            output_filename = render_voice_only(
                session_dir=session_dir,
                video_filename=meta["video_filename"],
                video_name=meta["video_name"],
                narrations=meta["narrations"],
                click_data=meta.get("click_data", [])
            )
            
            jobs[job_id].update({
                "status": "complete",
                "progress": 100,
                "message": "Render complete!",
                "result": {
                    "session_id": session_id,
                    "mode": mode,
                    "output_file": output_filename
                }
            })
            print(f"[RENDER JOB {job_id}] Voice-only render complete: {output_filename}")
            
        elif mode == "full":
            jobs[job_id].update({"progress": 30, "message": "Processing video segments with annotations..."})
            output_filename = render_voice_with_annotations(
                session_dir=session_dir,
                video_filename=meta["video_filename"],
                video_name=meta["video_name"],
                narrations=meta["narrations"],
                click_data=meta["click_data"]
            )
            
            jobs[job_id].update({
                "status": "complete",
                "progress": 100,
                "message": "Render complete!",
                "result": {
                    "session_id": session_id,
                    "mode": mode,
                    "output_file": output_filename
                }
            })
            print(f"[RENDER JOB {job_id}] Full annotations render complete: {output_filename}")
            
        else:
            jobs[job_id].update({
                "status": "error",
                "progress": 0,
                "message": f"Unknown render mode: {mode}"
            })
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"[RENDER JOB {job_id}] Error: {error_msg}")
        print(traceback.format_exc())
        jobs[job_id].update({
            "status": "error",
            "progress": 0,
            "message": f"Render failed: {error_msg}",
            "error": error_msg
        })


class UploadRequest(BaseModel):
    frames: list = []  # Pre-captured frames from the extension
    videoUrl: str = ""   # Optional: pre-hosted video URL (unused for now; video is uploaded from session dir)


class UploadRawRequest(BaseModel):
    frames: list = []       # Pre-captured frames [{image, clickTime, clickX, clickY, clientX, clientY}]
    clicks: list = []       # Raw click event objects from the extension
    videoBlob: str = ""     # Base64 data-URL of the raw WebM recording
    name: str = ""          # User-provided recording name


@app.post('/upload-raw')
async def upload_raw_to_central(body: UploadRawRequest):
    """
    Upload a raw (non-analyzed) recording as a kit to the central website.

    Accepts the video blob, pre-captured frames, and click data directly from the
    extension — no prior AI analysis required. Builds a kit with clean + annotated
    frames and posts it to the central API using the same schema as /upload/{session_id}.

    Args:
        body: JSON body with frames, clicks, videoBlob (base64), and recording name.

    Returns:
        JSON with status and the response from the central API.
    """
    frames = body.frames or []
    clicks = body.clicks or []
    video_blob_b64 = body.videoBlob or ""
    recording_name = body.name or "Untitled Recording"

    if not frames:
        return {"status": "error", "message": "No frames provided."}

    # Generate a unique kit ID for this raw upload
    kit_id = uuid.uuid4().hex
    kit_title = recording_name.replace("_", " ").title()

    print(f"[UPLOAD-RAW] Starting raw upload: kit_id={kit_id}, frames={len(frames)}, clicks={len(clicks)}, name={recording_name!r}")

    # Compute y_offset from zoomLevel — same formula as the analyzed upload
    # zoomLevel is stored on each click event object by the content script
    zoom_level = clicks[0].get('zoomLevel', 1.0) if clicks else 1.0
    y_offset = 114 * zoom_level
    print(f"[UPLOAD-RAW] zoomLevel={zoom_level}, y_offset={y_offset}")

    # Determine frame dimensions from the first frame
    first_frame = frames[0]
    frame_img_data = first_frame.get("image", "")
    frame_width, frame_height = 1920, 1080
    if frame_img_data.startswith("data:"):
        try:
            _, b64data = frame_img_data.split(",", 1)
            img_bytes = base64.b64decode(b64data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                frame_height, frame_width = img.shape[:2]
        except Exception as e:
            print(f"[UPLOAD-RAW] Could not decode first frame for dimensions: {e}")

    print(f"[UPLOAD-RAW] Frame dimensions: {frame_width}x{frame_height}")

    # Upload the clean video to the central API (if blob provided)
    kit_video_url = ""
    if video_blob_b64 and video_blob_b64.startswith("data:"):
        try:
            _, vid_b64 = video_blob_b64.split(",", 1)
            vid_bytes = base64.b64decode(vid_b64)
            print(f"[UPLOAD-RAW] Uploading clean video ({len(vid_bytes)} bytes) to central API...")
            async with httpx.AsyncClient(timeout=120.0) as http_client:
                video_response = await http_client.post(
                    f"{CENTRAL_BASE_URL}/api/screens/video",
                    data={"kit_id": kit_id},
                    files={"video": (f"{kit_id}.webm", vid_bytes, "video/webm")}
                )
            if video_response.status_code in (200, 201):
                kit_video_url = video_response.json().get("video_url", "")
                print(f"[UPLOAD-RAW] Clean video uploaded → {kit_video_url}")
            else:
                print(f"[UPLOAD-RAW] WARNING: Video upload failed: {video_response.status_code} - {video_response.text}")
        except Exception as e:
            print(f"[UPLOAD-RAW] WARNING: Video upload error: {e}")

    # Build screens — one per frame
    now_ms = int(time.time() * 1000)
    screens = []

    for i, frame in enumerate(frames):
        frame_img = frame.get("image", "")
        click_time = frame.get("clickTime", 0.0)

        # Use clientX/clientY from the matching click event object (has zoomLevel context),
        # falling back to the frame's own coords if the click list is shorter.
        if i < len(clicks):
            cursor_x = int(clicks[i].get('clientX', 0))
            cursor_y = int(clicks[i].get('clientY', 0)) + int(y_offset)
        else:
            cursor_x = int(frame.get('clientX') or frame.get('clickX') or 0)
            cursor_y = int(frame.get('clientY') or frame.get('clickY') or 0) + int(y_offset)

        if not frame_img.startswith("data:"):
            print(f"[UPLOAD-RAW] WARNING: Frame {i} has no valid image data, skipping")
            continue

        # Decode frame
        try:
            _, b64data = frame_img.split(",", 1)
            img_bytes = base64.b64decode(b64data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise ValueError("cv2.imdecode returned None")
        except Exception as e:
            print(f"[UPLOAD-RAW] WARNING: Could not decode frame {i}: {e}")
            continue

        # Encode clean (unannotated) frame
        _, clean_buffer = cv2.imencode(".png", frame_bgr)
        clean_b64 = base64.b64encode(clean_buffer).decode("utf-8")
        clean_data_url = f"data:image/png;base64,{clean_b64}"

        # Draw green rectangle annotation at click point
        rect_size = 60
        x1 = int(cursor_x - rect_size // 2)
        y1 = int(cursor_y - rect_size // 2)
        x2 = int(cursor_x + rect_size // 2)
        y2 = int(cursor_y + rect_size // 2)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Encode annotated frame
        _, ann_buffer = cv2.imencode(".png", frame_bgr)
        ann_b64 = base64.b64encode(ann_buffer).decode("utf-8")
        ann_data_url = f"data:image/png;base64,{ann_b64}"

        screen_id = f"screen_{kit_id}_{i}"
        screen = {
            "id": screen_id,
            "name": f"Step {i + 1}",
            "width": frame_width,
            "height": frame_height,
            "backgroundColor": "#000000",
            "backgroundImage": ann_data_url,
            "cleanBackgroundImage": clean_data_url,
            "audioUrl": "",
            "audioText": "",
            "audioLanguage": "en",
            "audioDuration": 0,
            "screenDuration": 5000,
            "autoNext": True,
            "elements": [
                {
                    "id": f"cursor_{kit_id}_{i}",
                    "type": "cursor",
                    "x": cursor_x,
                    "y": cursor_y,
                    "width": 60,
                    "height": 60,
                    "backgroundColor": "transparent",
                    "opacity": 1,
                    "borderWidth": 0,
                    "borderColor": "transparent",
                    "action": "click",
                    "content": json.dumps({"clickTime": click_time}),  # seconds into session video
                }
            ],
        }
        screens.append(screen)
        print(f"[UPLOAD-RAW] Screen {i+1}/{len(frames)}: click at ({cursor_x}, {cursor_y}), t={click_time:.2f}s")

    if not screens:
        return {"status": "error", "message": "No valid screens could be created from the provided frames."}

    # Build kit payload
    thumbnail = screens[0]["backgroundImage"] if screens else ""
    kit_payload = {
        "id": kit_id,
        "title": kit_title,
        "thumbnail": thumbnail,
        "videoUrl": kit_video_url,  # Full session video URL at kit level (for Preview button)
        "createdAt": now_ms,
        "updatedAt": now_ms,
        "userId": 3,
        "published": False,
        "publishedKitId": "",
        "screens": screens,
    }

    print(f"[UPLOAD-RAW] Kit payload built: {len(screens)} screens, title: {kit_title!r}")
    print(f"[UPLOAD-RAW] Kit-level videoUrl: '{kit_video_url}' (empty = video upload failed)")

    # POST to central API
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                CENTRAL_API_URL,
                json=kit_payload,
                headers={"Content-Type": "application/json"},
            )

        print(f"[UPLOAD-RAW] Central API response: {response.status_code}")

        if response.status_code in (200, 201):
            result = response.json()
            print(f"[UPLOAD-RAW] Upload successful: kit_id={result.get('id')} title={result.get('title')!r} videoUrl={result.get('videoUrl')} screens={len(result.get('screens', []))}")
            return {
                "status": "ok",
                "message": "Kit uploaded successfully!",
                "kit_id": kit_id,
                "central_response": result,
            }
        else:
            error_text = response.text
            print(f"[UPLOAD-RAW] Upload failed: {response.status_code} - {error_text}")
            return {
                "status": "error",
                "message": f"Central API returned {response.status_code}: {error_text}",
            }
    except httpx.ConnectError:
        print(f"[UPLOAD-RAW] Could not connect to central API at {CENTRAL_API_URL}")
        return {
            "status": "error",
            "message": f"Could not connect to central API at {CENTRAL_API_URL}. Is the server running?",
        }
    except Exception as e:
        print(f"[UPLOAD-RAW] Error uploading to central API: {e}")
        return {"status": "error", "message": f"Upload failed: {str(e)}"}


@app.post('/upload/{session_id}')
async def upload_to_central(session_id: str, body: UploadRequest = UploadRequest()):
    """
    Upload an analyzed session as a kit to the central website.
    
    Uses pre-captured frames from the extension (taken at the exact click moment)
    for pixel-perfect timing. Falls back to video extraction if no frames provided.

    Args:
        session_id: The session identifier to upload.
        body: Optional JSON body with pre-captured frames from IndexedDB.

    Returns:
        JSON with status and the response from the central API.
    """
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        return {"status": "error", "message": "Session not found."}

    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        return {"status": "error", "message": "Session metadata not found. Run /analyze first."}

    meta = json.loads(meta_path.read_text())
    
    video_filename = meta["video_filename"]
    video_name = meta.get("video_name", "Untitled Kit")
    narrations = meta.get("narrations", [])
    click_data = meta.get("click_data", [])
    
    if not narrations or not click_data:
        return {"status": "error", "message": "No narrations or click data found in session."}
    
    # Get zoom level and compute y_offset (same logic as render_voice_with_annotations)
    zoom_level = click_data[0].get('zoomLevel', 1.0) if click_data else 1.0
    y_offset = 114 * zoom_level
    
    # Pre-captured frames from the extension (taken at exact click instant)
    precaptured_frames = body.frames or []
    use_precaptured = len(precaptured_frames) > 0
    
    print(f"[UPLOAD] Starting upload for session {session_id}")
    print(f"[UPLOAD] Narrations: {len(narrations)}, Clicks: {len(click_data)}, Pre-captured frames: {len(precaptured_frames)}")
    
    # Determine frame dimensions
    video = None
    if use_precaptured:
        # Decode first frame to get dimensions
        first_frame_data = precaptured_frames[0].get('image', '')
        if first_frame_data.startswith('data:'):
            header, b64data = first_frame_data.split(',', 1)
            img_bytes = base64.b64decode(b64data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame_width = img.shape[1]
            frame_height = img.shape[0]
        else:
            frame_width = 1920
            frame_height = 1080
        print(f"[UPLOAD] Using pre-captured frames: {frame_width}x{frame_height}")
    else:
        # Fallback: open video for frame extraction
        video_path = session_dir / video_filename
        fixed_video_path = session_dir / f"{video_name}_fixed.webm"
        if not fixed_video_path.exists():
            try:
                from moviepy.config import get_setting
                ffmpeg_binary = get_setting("FFMPEG_BINARY")
                subprocess.run([
                    ffmpeg_binary, '-i', str(video_path),
                    '-c', 'copy', '-y',
                    str(fixed_video_path)
                ], check=True, capture_output=True)
            except Exception:
                fixed_video_path = video_path
        
        video = VideoFileClip(str(fixed_video_path), audio=False, fps_source='fps')
        frame_width = video.size[0]
        frame_height = video.size[1]
        print(f"[UPLOAD] Fallback: extracting frames from video: {frame_width}x{frame_height}")
    
    # Upload the clean (raw) session video to the central API to get a hosted URL
    kit_video_url = ""
    video_path_for_upload = session_dir / video_filename
    # Prefer the fixed WebM if it was already created during a prior render
    fixed_video_path_for_upload = session_dir / f"{video_name}_fixed.webm"
    if fixed_video_path_for_upload.exists():
        video_path_for_upload = fixed_video_path_for_upload

    print(f"[UPLOAD] Video file check:")
    print(f"[UPLOAD]   session_dir = {session_dir}")
    print(f"[UPLOAD]   video_filename = {video_filename}")
    print(f"[UPLOAD]   video_path_for_upload = {video_path_for_upload}")
    print(f"[UPLOAD]   exists = {video_path_for_upload.exists()}")
    # List all files in session dir for debugging
    if session_dir.exists():
        files = list(session_dir.iterdir())
        print(f"[UPLOAD]   files in session dir: {[f.name for f in files]}")

    if video_path_for_upload.exists():
        try:
            print(f"[UPLOAD] Uploading clean video to central API: {video_path_for_upload.name}")
            async with httpx.AsyncClient(timeout=120.0) as http_client:
                with open(video_path_for_upload, "rb") as vf:
                    video_response = await http_client.post(
                        f"{CENTRAL_BASE_URL}/api/screens/video",
                        data={"kit_id": session_id},
                        files={"video": (video_path_for_upload.name, vf, "video/webm")}
                    )
            if video_response.status_code in (200, 201):
                video_result = video_response.json()
                kit_video_url = video_result.get("video_url", "")
                print(f"[UPLOAD] Clean video uploaded → {kit_video_url}")
            else:
                print(f"[UPLOAD] WARNING: Video upload failed: {video_response.status_code} - {video_response.text}")
        except Exception as e:
            print(f"[UPLOAD] WARNING: Video upload error: {e}")
    else:
        print(f"[UPLOAD] WARNING: Video file not found for upload: {video_path_for_upload}")

    # Build screens - one per click/narration
    num = min(len(narrations), len(click_data))
    screens = []
    now_ms = int(time.time() * 1000)
    
    for i in range(num):
        narr = narrations[i]
        click = click_data[i]
        
        # Compute cursor coordinates with offset and zoom level
        cursor_x = int(click.get('clientX', 0))
        cursor_y = int(click.get('clientY', 0)) + int(y_offset)
        
        if use_precaptured and i < len(precaptured_frames):
            # Use the pre-captured frame (taken at exact click instant)
            frame_img = precaptured_frames[i].get('image', '')
            
            # Decode, annotate, re-encode
            if frame_img.startswith('data:'):
                header, b64data = frame_img.split(',', 1)
                img_bytes = base64.b64decode(b64data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                print(f"[UPLOAD] WARNING: Frame {i} has no valid image data, skipping")
                continue
            
            print(f"[UPLOAD] Frame {i+1}/{num}: using pre-captured frame, cursor at ({cursor_x}, {cursor_y})")
        else:
            # Fallback: extract from video
            click_time = narr.get("click_time", 0.0)
            click_time = max(0, min(click_time, video.duration - 0.01))
            frame_rgb = video.get_frame(click_time)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            print(f"[UPLOAD] Frame {i+1}/{num}: extracted from video at {click_time:.2f}s, cursor at ({cursor_x}, {cursor_y})")
        
        # Encode clean (unannotated) frame as base64 PNG
        _, clean_buffer = cv2.imencode('.png', frame_bgr)
        clean_b64 = base64.b64encode(clean_buffer).decode('utf-8')
        clean_data_url = f"data:image/png;base64,{clean_b64}"
        
        # Draw rectangle annotation (same as render_voice_with_annotations)
        rect_size = 60
        rect_color = (0, 255, 0)  # Green in BGR
        rect_thickness = 3
        
        x1 = int(cursor_x - rect_size // 2)
        y1 = int(cursor_y - rect_size // 2)
        x2 = int(cursor_x + rect_size // 2)
        y2 = int(cursor_y + rect_size // 2)
        
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), rect_color, rect_thickness)
        
        # Encode annotated frame as base64 PNG
        _, buffer = cv2.imencode('.png', frame_bgr)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        frame_data_url = f"data:image/png;base64,{frame_b64}"
        
        # Upload narration audio to central API
        audio_file = session_dir / f"{video_name}_narration_{i}.mp3"
        audio_url = ""
        audio_duration_ms = 0
        screen_id = f"screen_{session_id}_{i}"
        
        if audio_file.exists():
            audio_duration_ms = int(narr.get("audio_duration", 0) * 1000)
            try:
                async with httpx.AsyncClient(timeout=60.0) as http_client:
                    with open(audio_file, "rb") as f:
                        response = await http_client.post(
                            f"{CENTRAL_BASE_URL}/api/screens/audio",
                            data={"screen_id": screen_id},
                            files={"audio": (audio_file.name, f, "audio/mpeg")}
                        )
                
                if response.status_code in (200, 201):
                    audio_result = response.json()
                    audio_url = audio_result.get("audio_url", "")
                    print(f"[UPLOAD] Audio {i+1}/{num}: uploaded \u2192 {audio_url}")
                else:
                    print(f"[UPLOAD] WARNING: Audio upload failed for screen {i}: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"[UPLOAD] WARNING: Audio upload error for screen {i}: {e}")
        
        screen = {
            "id": f"screen_{session_id}_{i}",
            "name": narr.get("ui_element", f"Step {i + 1}"),
            "width": frame_width,
            "height": frame_height,
            "backgroundColor": "#000000",
            "backgroundImage": frame_data_url,
            "cleanBackgroundImage": clean_data_url,
            "audioUrl": audio_url,
            "audioText": narr.get("narration_text", ""),
            "audioLanguage": meta.get("language", "en"),
            "audioDuration": audio_duration_ms,
            "screenDuration": max(audio_duration_ms, 5000),
            "autoNext": True,
            "elements": [
                {
                    "id": f"cursor_{session_id}_{i}",
                    "type": "cursor",
                    "x": cursor_x,
                    "y": cursor_y,
                    "width": 60,
                    "height": 60,
                    "backgroundColor": "transparent",
                    "opacity": 1,
                    "borderWidth": 0,
                    "borderColor": "transparent",
                    "action": "click",
                }
            ]
        }
        screens.append(screen)
        print(f"[UPLOAD] Screen {i+1}/{num}: click at ({cursor_x}, {cursor_y}), audio: {audio_duration_ms}ms, video: {kit_video_url[:60] if kit_video_url else 'none'}")
    
    if video:
        video.close()
    
    if not screens:
        return {"status": "error", "message": "No screens could be created from the session data."}
    
    # Generate thumbnail from the first screen's frame
    thumbnail = screens[0]["backgroundImage"] if screens else ""
    
    # Build the kit payload
    kit_payload = {
        "id": session_id,
        "title": meta.get("video_name", "Untitled Kit").replace("_", " ").title(),
        "thumbnail": thumbnail,
        "videoUrl": kit_video_url,  # Full session video URL at kit level (for Preview button)
        "createdAt": now_ms,
        "updatedAt": now_ms,
        "userId": 3,
        "published": False,
        "publishedKitId": "",
        "screens": screens,
    }
    
    print(f"[UPLOAD] Kit payload built: {len(screens)} screens, title: {kit_payload['title']}")
    print(f"[UPLOAD] Kit-level videoUrl: '{kit_payload['videoUrl']}' (empty = video upload failed)")
    
    # POST to central API
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                CENTRAL_API_URL,
                json=kit_payload,
                headers={"Content-Type": "application/json"}
            )
        
        print(f"[UPLOAD] Central API response: {response.status_code}")
        
        if response.status_code in (200, 201):
            result = response.json()
            print(f"[UPLOAD] Upload successful: kit_id={result.get('id')} title={result.get('title')!r} videoUrl={result.get('videoUrl')} screens={len(result.get('screens', []))}")
            return {
                "status": "ok",
                "message": "Kit uploaded successfully!",
                "central_response": result
            }
        else:
            error_text = response.text
            print(f"[UPLOAD] Upload failed: {response.status_code} - {error_text}")
            return {
                "status": "error",
                "message": f"Central API returned {response.status_code}: {error_text}"
            }
    except httpx.ConnectError:
        print(f"[UPLOAD] Could not connect to central API at {CENTRAL_API_URL}")
        return {
            "status": "error",
            "message": f"Could not connect to central API at {CENTRAL_API_URL}. Is the server running?"
        }
    except Exception as e:
        print(f"[UPLOAD] Error uploading to central API: {e}")
        return {
            "status": "error",
            "message": f"Upload failed: {str(e)}"
        }


@app.get('/download/{session_id}/{filename}')
async def download_video(session_id: str, filename: str):
    """
    Download rendered video file.

    Args:
        session_id: The session identifier.
        filename: The video filename to download.

    Returns:
        FileResponse for the requested video file, or error JSON if not found.
    """
    session_dir = SESSIONS_DIR / session_id
    file_path = session_dir / filename
    
    if not file_path.exists():
        return {"status": "error", "message": "File not found."}
    
    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=filename
    )


@app.delete('/session/{session_id}')
async def cleanup_session(session_id: str):
    """
    Clean up session files after the user has downloaded the video.

    Args:
        session_id: The session identifier to clean up.

    Returns:
        JSON with status and message.
    """
    session_dir = SESSIONS_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
        return {"status": "ok", "message": "Session cleaned up."}
    return {"status": "error", "message": "Session not found."}
