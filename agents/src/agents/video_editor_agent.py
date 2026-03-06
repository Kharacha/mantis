# src/agents/video_editor_agent.py
from datetime import datetime, UTC
from uuid import uuid4
from urllib.parse import quote
from pathlib import Path
from typing import List, Optional, Tuple
import os, asyncio, tempfile, shutil

import httpx
import cv2
import numpy as np

from uagents import Agent, Context, Protocol, Model
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement, ChatMessage, TextContent, chat_protocol_spec
)

# ---------- Identity / config ----------
NAME = "video-editor"
SEED = os.getenv("VIDEO_EDIT_AGENT_SEED", NAME)
PORT = int(os.getenv("VIDEO_EDIT_AGENT_PORT", os.getenv("PORT", "8110")))

# Detection config (optional external model)
DETECTOR_URL = os.getenv("DETECTOR_URL", "").rstrip("/")
THREAT_THRESHOLD = float(os.getenv("THREAT_THRESHOLD", "0.5"))
MIN_THREAT_DURATION = float(os.getenv("MIN_THREAT_DURATION", "0.5"))  # seconds
MERGE_NEARBY_THREATS = float(os.getenv("MERGE_NEARBY_THREATS", "1.0"))  # seconds
ANALYZE_FPS = float(os.getenv("ANALYZE_FPS", "0"))  # 0 = full fps; else sample rate

# Fallback windows if nothing crosses threshold
FALLBACK_WINDOW_SEC = float(os.getenv("FALLBACK_WINDOW_SEC", "2.0"))
FALLBACK_MAX_KEEP_SEC = float(os.getenv("FALLBACK_MAX_KEEP_SEC", "16.0"))

# ---- Supabase via raw HTTP (no SDK conflicts) ----
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")  # service_role key
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "detectBucket")
SUPABASE_PATH_PREFIX = os.getenv("SUPABASE_PATH_PREFIX", "").strip().strip("/")
SUPABASE_SIGNED_URL_TTL = os.getenv("SUPABASE_SIGNED_URL_TTL", "")  # optional
KEEP_LOCAL_OUTPUTS = os.getenv("KEEP_LOCAL_OUTPUTS", "0") == "1"

# ---------- Agent ----------
agent = Agent(
    name=NAME,
    seed=SEED,
    port=PORT,
    mailbox=True,               # AgentVerse/ASI messaging
    publish_agent_details=True,
)

# ---------- Chat protocol (Inspector/ASI) ----------
chat_proto = Protocol(spec=chat_protocol_spec)

@chat_proto.on_message(ChatMessage)
async def on_chat(ctx: Context, sender: str, msg: ChatMessage):
    """
    Command:
      edit <VIDEO_URL>

    Downloads the video, detects threats, keeps only threat segments,
    uploads the result to Supabase, and replies with the URL.
    """
    text = next((c.text for c in msg.content if isinstance(c, TextContent)), "").strip()

    if text.lower().startswith("edit "):
        video_url = text[5:].strip()
        if not video_url:
            await _reply(ctx, sender, "❌ Missing video URL")
            return

        await _reply(ctx, sender, f"🛠️ Started analysis for {video_url}. I’ll send the result when it’s ready.")

        async def _run_and_reply():
            try:
                req = VideoEditRequest(video_url=video_url)
                result = await edit_video(ctx, req)
                if result.ok:
                    kept = f"{result.duration_kept_seconds:.2f}s" if result.duration_kept_seconds else "unknown"
                    outp = result.download_url or "(upload missing)"
                    msg_text = (
                        f"✅ {result.message}\n"
                        f"Threat Duration: {kept}\n"
                        f"Threat Segments: {result.threat_count or 0}\n"
                        f"Output: {outp}"
                    )
                else:
                    msg_text = f"❌ {result.message}"
            except Exception as e:
                msg_text = f"❌ Failed during processing: {e}"
            await _reply(ctx, sender, msg_text)

        asyncio.create_task(_run_and_reply())
        return

    await _reply(ctx, sender, f"[{NAME}] pong: {text or 'ping'}")

async def _reply(ctx: Context, sender: str, text: str):
    await ctx.send(sender, ChatMessage(
        timestamp=datetime.now(UTC), msg_id=str(uuid4()),
        content=[TextContent(type="text", text=text)],
    ))

@chat_proto.on_message(ChatAcknowledgement)
async def on_chat_ack(_: Context, __: str, ___: ChatAcknowledgement):
    return

agent.include(chat_proto, publish_manifest=True)

# ---------- Models ----------
class HealthStatus(Model):
    ok: bool
    name: str
    addr: str

class TimeSegment(Model):
    start: float  # seconds
    end: float    # seconds

class VideoEditRequest(Model):
    video_url: str

class VideoEditResult(Model):
    ok: bool
    message: str
    output_path: Optional[str] = None  # None unless KEEP_LOCAL_OUTPUTS=1
    download_url: Optional[str] = None
    duration_kept_seconds: Optional[float] = None
    threat_count: Optional[int] = None
    analysis_log: Optional[str] = None

# ---------- Health ----------
@agent.on_rest_get("/health", HealthStatus)
async def health(_: Context) -> HealthStatus:
    return HealthStatus(ok=True, name=NAME, addr=agent.address)

# ---------- Threat Detection ----------
async def detect_threat_in_frame(frame_bytes: bytes) -> float:
    if not DETECTOR_URL:
        return 0.0
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            files = {"file": ("frame.jpg", frame_bytes, "image/jpeg")}
            r = await client.post(DETECTOR_URL, files=files)
            r.raise_for_status()
            data = r.json()
            score = data.get("score") or data.get("confidence") or data.get("threat") or 0.0
            return float(np.clip(float(score), 0.0, 1.0))
        except Exception as e:
            print(f"[DETECTOR ERROR] {e}")
            return 0.0

def _motion_and_scene_score(frame: np.ndarray, prev_frame: Optional[np.ndarray]) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / (gray.shape[0] * gray.shape[1])
    motion_score = 0.0
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        motion_score = float(diff.mean()) / 255.0
    return float(np.clip(motion_score * 0.6 + edge_density * 0.4, 0.0, 1.0))

def _select_top_windows(
    scored_samples: List[Tuple[float, float]],
    window_sec: float = FALLBACK_WINDOW_SEC,
    max_keep_sec: float = FALLBACK_MAX_KEEP_SEC,
) -> List[TimeSegment]:
    if not scored_samples:
        return []
    scored = sorted(scored_samples, key=lambda x: x[1], reverse=True)
    kept: List[TimeSegment] = []
    used: List[Tuple[float, float]] = []
    def overlaps(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        return not (a[1] <= b[0] or b[1] <= a[0])
    total = 0.0
    half = window_sec / 2.0
    for t, _ in scored:
        seg = (max(t - half, 0.0), t + half)
        if any(overlaps(seg, u) for u in used):
            continue
        used.append(seg)
        kept.append(TimeSegment(start=seg[0], end=seg[1]))
        total += window_sec
        if total >= max_keep_sec:
            break
    merged: List[TimeSegment] = []
    for seg in sorted(kept, key=lambda s: s.start):
        if merged and seg.start - merged[-1].end <= 0.25:
            merged[-1].end = max(merged[-1].end, seg.end)
        else:
            merged.append(seg)
    return merged

def _open_video_any(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        return cap
    cap.release()
    for backend in [getattr(cv2, "CAP_FFMPEG", None),
                    getattr(cv2, "CAP_MSMF", None),
                    getattr(cv2, "CAP_GSTREAMER", None)]:
        if backend is None:
            continue
        cap2 = cv2.VideoCapture(path, backend)
        if cap2.isOpened():
            return cap2
        cap2.release()
    return cv2.VideoCapture(path)

# ---------- Writer helpers (H.264-first) ----------
def _pick_video_writer(output_path: str, fps: float, size: Tuple[int, int]) -> Optional[cv2.VideoWriter]:
    """
    Try web-playable codecs first (H.264/avc1). Fall back to mp4v as last resort.
    """
    candidates = [
        ("avc1", "H.264 (avc1)"),
        ("H264", "H.264 (H264)"),
        ("mp4v", "MPEG-4 Part 2"),
    ]
    for fourcc_str, label in candidates:
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            wr = cv2.VideoWriter(output_path, fourcc, fps, size)
            if wr.isOpened():
                print(f"[WRITER] Using codec {label} ({fourcc_str})")
                return wr
        except Exception:
            pass
    return None

# ---------- Threat analysis ----------
async def analyze_video_for_threats(video_path: str) -> Tuple[List[TimeSegment], str]:
    vid = _open_video_any(video_path)
    if not vid.isOpened():
        raise RuntimeError("Could not open video file")

    fps = vid.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        vid.release()
        raise RuntimeError("Invalid video properties")

    total_duration = total_frames / fps
    frame_duration = 1.0 / fps
    step = max(1, int(round(fps / ANALYZE_FPS))) if (ANALYZE_FPS and ANALYZE_FPS > 0) else 1

    threat_frames: List[Tuple[int, float, float]] = []
    score_series: List[Tuple[int, float, float]] = []

    frame_idx = 0
    prev_for_motion = None
    log_lines = [f"Analyzing: {total_frames} frames @ {fps:.2f} fps, {total_duration:.2f}s total (step={step})"]
    use_detector = bool(DETECTOR_URL)

    while True:
        ok, frame = vid.read()
        if not ok:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx * frame_duration
        if use_detector:
            h, w = frame.shape[:2]
            max_dim = 480
            scale = min(1.0, float(max_dim) / max(h, w))
            frame_small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
            ok_enc, jpg = cv2.imencode(".jpg", frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            score = await detect_threat_in_frame(jpg.tobytes()) if ok_enc else 0.0
        else:
            score = _motion_and_scene_score(frame, prev_for_motion)

        score = float(score)
        score_series.append((frame_idx, timestamp, score))
        if score >= THREAT_THRESHOLD:
            threat_frames.append((frame_idx, timestamp, score))
            log_lines.append(f"THREAT @ {timestamp:.2f}s (frame {frame_idx}): {score:.3f}")
        if frame_idx % (5 * step) == 0:
            prev_for_motion = frame
        frame_idx += 1

    vid.release()

    segments = _group_threats(threat_frames, frame_duration * step, MIN_THREAT_DURATION, MERGE_NEARBY_THREATS)
    extra_log = []
    if not segments and score_series:
        ts_scores = [(ts, sc) for _, ts, sc in score_series]
        segments = _select_top_windows(ts_scores, window_sec=FALLBACK_WINDOW_SEC, max_keep_sec=FALLBACK_MAX_KEEP_SEC)
        if segments:
            extra_log.append(
                f"No scores >= threshold; using top windows (win={FALLBACK_WINDOW_SEC}s, keep≈{FALLBACK_MAX_KEEP_SEC}s)"
            )
        else:
            extra_log.append("No usable activity found")

    summary = [f"Threat frames: {len(threat_frames)} → segments: {len(segments)}"]
    for seg in segments:
        summary.append(f"  {seg.start:.2f}s - {seg.end:.2f}s ({seg.end - seg.start:.2f}s)")
    return segments, "\n".join(summary + extra_log + log_lines)

def _group_threats(threat_frames: List[Tuple[int, float, float]],
                   sample_step_seconds: float,
                   min_duration: float,
                   merge_gap: float) -> List[TimeSegment]:
    if not threat_frames:
        return []
    segments: List[TimeSegment] = []
    cur_start = threat_frames[0][1]
    cur_end = cur_start
    for i in range(1, len(threat_frames)):
        _, ts, _ = threat_frames[i]
        prev_ts = threat_frames[i - 1][1]
        if ts - prev_ts > sample_step_seconds * 1.5:
            if cur_end - cur_start >= min_duration:
                segments.append(TimeSegment(start=max(cur_start - sample_step_seconds * 0.5, 0.0),
                                            end=cur_end + sample_step_seconds * 0.5))
            cur_start, cur_end = ts, ts
        else:
            cur_end = ts
    if cur_end - cur_start >= min_duration:
        segments.append(TimeSegment(start=max(cur_start - sample_step_seconds * 0.5, 0.0),
                                    end=cur_end + sample_step_seconds * 0.5))
    if not segments:
        return []
    merged: List[TimeSegment] = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg.start - last.end <= merge_gap:
            last.end = max(last.end, seg.end)
        else:
            merged.append(seg)
    return merged

# ---------- Cut/concat with browser-safe codec ----------
async def cut_and_concatenate_video(input_path: str,
                                    output_path: str,
                                    segments: List[TimeSegment]) -> Tuple[bool, str, float]:
    """
    Keep only frames within threat segments and write a single mp4.
    We try H.264 first for browser playback and validate the result.
    """
    if not segments:
        return False, "No threat segments to extract", 0.0

    vid = _open_video_any(input_path)
    if not vid.isOpened():
        return False, "Cannot open input video", 0.0

    fps = vid.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = _pick_video_writer(output_path, fps, (width, height))
    if writer is None:
        vid.release()
        return False, "Cannot create output video writer with any supported codec", 0.0

    total_kept = 0.0
    frame_idx = 0
    try:
        while True:
            ok, frame = vid.read()
            if not ok:
                break
            t = frame_idx / fps
            for seg in segments:
                if seg.start <= t < seg.end:
                    writer.write(frame)
                    total_kept += 1.0 / fps
                    break
            frame_idx += 1
    finally:
        writer.release()
        vid.release()

    # validate non-zero duration
    try:
        cap = cv2.VideoCapture(output_path)
        if not cap.isOpened():
            cap.release()
            return False, "Output file could not be opened after writing", total_kept
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        out_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        cap.release()
        duration = (frames / out_fps) if (frames > 0 and out_fps > 0) else 0.0
        if duration < 0.1:
            return False, (
                "Output appears to have 0 duration. Your OpenCV build may lack H.264 encoding. "
                "Install FFmpeg-enabled OpenCV or try a different machine."
            ), total_kept
    except Exception:
        pass

    return True, "Video extracted successfully", total_kept

# ---------- Supabase upload (async raw HTTP) ----------
async def _upload_to_supabase(local_path: str) -> Optional[str]:
    """
    Upload the file to Supabase Storage and return a PUBLIC (or signed) URL.
    Uses Service Role key; bucket must exist. Public buckets return immediate URL.
    """
    if not (SUPABASE_URL and SUPABASE_SERVICE_KEY and SUPABASE_BUCKET):
        print("[SUPABASE] Missing SUPABASE_URL / SUPABASE_SERVICE_KEY / SUPABASE_BUCKET")
        return None

    filename = Path(local_path).name
    key = f"{SUPABASE_PATH_PREFIX}/{filename}" if SUPABASE_PATH_PREFIX else filename
    object_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{key}"

    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "apikey": SUPABASE_SERVICE_KEY,      # accepted header
        "x-upsert": "true",
        "Content-Type": "video/mp4",
    }

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            with open(local_path, "rb") as f:
                data = f.read()
            r = await client.post(object_url, content=data, headers=headers)
            if r.status_code not in (200, 201):
                print(f"[SUPABASE UPLOAD ERROR] {r.status_code} {r.text}")
                return None
    except Exception as e:
        print(f"[SUPABASE UPLOAD EXCEPTION] {e}")
        return None

    # Public URL (works if bucket is public)
    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{key}"

    # For private bucket, optionally mint a signed URL
    if SUPABASE_SIGNED_URL_TTL:
        try:
            sign_url = f"{SUPABASE_URL}/storage/v1/object/sign/{SUPABASE_BUCKET}/{key}"
            async with httpx.AsyncClient(timeout=30) as client:
                r2 = await client.post(
                    sign_url,
                    headers={
                        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                        "apikey": SUPABASE_SERVICE_KEY,
                        "Content-Type": "application/json",
                    },
                    json={"expiresIn": int(SUPABASE_SIGNED_URL_TTL)},
                )
                if r2.status_code in (200, 201):
                    data = r2.json()
                    signed = data.get("signedURL") or data.get("signed_url")
                    if signed:
                        return f"{SUPABASE_URL}{signed}" if signed.startswith("/") else signed
        except Exception as e:
            print(f"[SUPABASE SIGNED URL ERROR] {e}")

    return public_url

# ---------- REST: /video/edit ----------
@agent.on_rest_post("/video/edit", VideoEditRequest, VideoEditResult)
async def edit_video(_: Context, body: VideoEditRequest) -> VideoEditResult:
    if not body.video_url:
        return VideoEditResult(ok=False, message="Missing video_url")

    temp_dir = Path(tempfile.mkdtemp(prefix="videdit_")).resolve()
    input_path = str(temp_dir / "input_video")                 # any ext; OpenCV probes
    temp_output_path = str(temp_dir / f"threats_{uuid4().hex}.mp4")
    local_copy_path = str(Path.cwd() / f"threats_{uuid4().hex}.mp4") if KEEP_LOCAL_OUTPUTS else None

    try:
        # Download source
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", body.video_url, follow_redirects=True) as r:
                r.raise_for_status()
                with open(input_path, "wb") as f:
                    async for chunk in r.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)

        # Analyze
        segments, analysis_log = await analyze_video_for_threats(input_path)
        if not segments:
            return VideoEditResult(ok=False, message="No threats or activity windows found", analysis_log=analysis_log)

        # Produce edited video to TEMP
        ok, msg, total_kept = await cut_and_concatenate_video(input_path, temp_output_path, segments)
        if not ok:
            return VideoEditResult(ok=False, message=msg, analysis_log=analysis_log)

        if not os.path.exists(temp_output_path) or os.path.getsize(temp_output_path) == 0:
            return VideoEditResult(ok=False, message="Output video is empty or missing", analysis_log=analysis_log)

        # Upload to Supabase
        sb_url = await _upload_to_supabase(temp_output_path)

        # Optional local copy for debugging
        out_path_to_report = None
        if KEEP_LOCAL_OUTPUTS and local_copy_path:
            try:
                shutil.copyfile(temp_output_path, local_copy_path)
                out_path_to_report = local_copy_path
            except Exception:
                out_path_to_report = None

        return VideoEditResult(
            ok=True,
            message="Threats detected and extracted successfully",
            output_path=out_path_to_report,
            download_url=sb_url,
            duration_kept_seconds=total_kept,
            threat_count=len(segments),
            analysis_log=analysis_log,
        )

    except Exception as e:
        return VideoEditResult(ok=False, message=f"Processing failed: {e}")

    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

# ---------- Optional: typed protocol ----------
edit_proto = Protocol(name="video-edit-protocol", version="1.0.0")

@edit_proto.on_message(VideoEditRequest)
async def on_edit_msg(ctx: Context, sender: str, req: VideoEditRequest):
    res = await edit_video(ctx, req)
    await ctx.send(sender, res)

agent.include(edit_proto)

# ---------- Inspector link ----------
def print_inspector_link():
    addr = quote(agent.address, safe="")
    print(f"[{NAME}] Inspector: https://agentverse.ai/inspect/?uri=http://127.0.0.1:{PORT}&address={addr}")

if __name__ == "__main__":
    print(f"[{NAME}] Address: {agent.address}")
    print_inspector_link()
    agent.run()

