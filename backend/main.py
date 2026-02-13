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
from typing import Annotated, Dict
from elevenlabs import ElevenLabs

load_dotenv()

MODEL_NAME = 'gemini-2.5-flash'  # gemini-2.5-flash, gemini-2.0-flash-lite, gemini-3-flash-preview
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

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

def render_voice_only(session_dir: Path, video_filename: str, video_name: str, narrations: list) -> str:
    """
    Render a 'voice-only' video where the video freezes at each click point while narration plays.
    
    Logic:
    1. Play video normally until a click happens
    2. Freeze the frame at the click point
    3. Play the narration over the frozen frame
    4. Resume video playback
    5. Repeat for each click
    """
    
    video_path = session_dir / video_filename
    output_filename = f"{video_name}_voice_only.mp4"
    output_path = session_dir / output_filename
    
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
        
        print(f"[RENDER] Processing click {i+1}/{len(narrations)} at {click_time:.2f}s, narration: {narration_duration:.2f}s")
        
        # 1. Add normal video segment from current_time to click_time
        if click_time > current_time:
            segment = video.subclip(current_time, click_time)
            segments.append(segment)
            print(f"  → Segment {len(segments)}: Normal video {current_time:.2f}s → {click_time:.2f}s")
        
        # 2. Freeze frame at click_time and play narration
        frame = video.get_frame(click_time)
        frozen_clip = ImageClip(frame, duration=narration_duration)
        frozen_clip = frozen_clip.set_fps(video.fps)
        
        # 3. Add narration audio to frozen frame
        audio_file = session_dir / f"{video_name}_narration_{i}.mp3"
        if audio_file.exists():
            narration_audio = AudioFileClip(str(audio_file))
            frozen_clip = frozen_clip.set_audio(narration_audio)
            print(f"  → Segment {len(segments)+1}: Frozen frame for {narration_duration:.2f}s with narration")
        else:
            print(f"  → WARNING: Audio file not found: {audio_file}")
        
        segments.append(frozen_clip)
        
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
                narrations=meta["narrations"]
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
            jobs[job_id].update({
                "status": "error",
                "progress": 0,
                "message": "Full annotations mode not yet implemented"
            })
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
