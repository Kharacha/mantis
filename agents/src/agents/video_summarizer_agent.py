# src/agents/video_summarizer_agent.py
from datetime import datetime, UTC
from uuid import uuid4
from urllib.parse import quote
from pathlib import Path
from typing import List, Optional, Tuple
import os, asyncio, tempfile, shutil, socket
from urllib.parse import urlparse

import httpx
import cv2
import numpy as np

from uagents import Agent, Context, Protocol, Model
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement, ChatMessage, TextContent, chat_protocol_spec
)

# ───────────────────────── Identity / config ─────────────────────────
NAME = "video-summarizer"
SEED = os.getenv("VIDEO_SUM_AGENT_SEED", NAME)
PORT = int(os.getenv("VIDEO_SUM_AGENT_PORT", os.getenv("PORT", "8120")))

# Heuristic analysis knobs
ANALYZE_FPS = float(os.getenv("ANALYZE_FPS", "0"))            # 0 = use full fps; else sample rate
EVENT_THRESHOLD = float(os.getenv("EVENT_THRESHOLD", "0.35")) # activity score to mark an “event”
MIN_EVENT_DURATION = float(os.getenv("MIN_EVENT_DURATION", "1.0"))
MERGE_GAP = float(os.getenv("MERGE_GAP", "1.5"))
TOP_EVENTS_LIMIT = int(os.getenv("TOP_EVENTS_LIMIT", "8"))    # cap how many segments we summarize

# ───────────────────────── Agent setup ─────────────────────────
agent = Agent(
    name=NAME,
    seed=SEED,
    port=PORT,
    mailbox=True,                # required for AgentVerse / ASI chat
    publish_agent_details=True,
)

# ───────────────────────── Chat protocol (ASI/Inspector) ─────────────────────────
chat_proto = Protocol(spec=chat_protocol_spec)

@chat_proto.on_message(ChatMessage)
async def on_chat(ctx: Context, sender: str, msg: ChatMessage):
    """
    Commands:
      summarize <VIDEO_URL>
      ping  (or any text — returns pong)

    The agent downloads the video, detects the most eventful parts (motion/scene complexity),
    and replies with a natural-language summary + timestamped bullet points.
    """
    text = next((c.text for c in msg.content if isinstance(c, TextContent)), "").strip()

    if text.lower().startswith(("summarize ", "summarise ")):
        video_url = text.split(" ", 1)[1].strip() if " " in text else ""
        if not video_url:
            await _reply(ctx, sender, "❌ Missing video URL")
            return

        await _reply(ctx, sender, f"🛠️ Analyzing video… {video_url}")

        async def _run_and_reply():
            try:
                req = VideoSummarizeRequest(video_url=video_url)
                result = await summarize_video(ctx, req)
                if result.ok:
                    await _reply(ctx, sender, result.summary or "✅ Done (no text)")
                else:
                    await _reply(ctx, sender, f"❌ {result.message}")
            except Exception as e:
                await _reply(ctx, sender, f"❌ Failed during processing: {e}")

        asyncio.create_task(_run_and_reply())
        return

    # default ping/pong
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

# ───────────────────────── Models ─────────────────────────
class HealthStatus(Model):
    ok: bool
    name: str
    addr: str

class TimeSegment(Model):
    start: float  # seconds
    end: float    # seconds
    peak_score: Optional[float] = None

class VideoSummarizeRequest(Model):
    video_url: str

class VideoSummarizeResult(Model):
    ok: bool
    message: str
    summary: Optional[str] = None
    events: Optional[List[TimeSegment]] = None
    analysis_log: Optional[str] = None
    duration_seconds: Optional[float] = None

# ───────────────────────── Health ─────────────────────────
@agent.on_rest_get("/health", HealthStatus)
async def health(_: Context) -> HealthStatus:
    return HealthStatus(ok=True, name=NAME, addr=agent.address)

# ───────────────────────── Networking helpers ─────────────────────────
def _preflight_dns(url: str) -> tuple[bool, str]:
    try:
        host = urlparse(url).hostname or url.split("//",1)[-1].split("/",1)[0]
        socket.getaddrinfo(host, None)
        return True, host
    except Exception as e:
        return False, f"{e}"

async def _download_file(url: str, dest_path: str) -> None:
    ok_dns, info = _preflight_dns(url)
    if not ok_dns:
        raise RuntimeError(f"DNS resolve failed for {url}: {info}")

    max_attempts = 4
    backoff = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                async with client.stream("GET", url, follow_redirects=True) as r:
                    r.raise_for_status()
                    with open(dest_path, "wb") as f:
                        async for chunk in r.aiter_bytes(chunk_size=1024 * 1024):
                            f.write(chunk)
            return
        except Exception as e:
            if attempt == max_attempts:
                raise
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 8.0)

# ───────────────────────── Video I/O helpers ─────────────────────────
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

def _fmt_ts(seconds: float) -> str:
    s = max(0.0, float(seconds))
    h = int(s // 3600); s -= h * 3600
    m = int(s // 60);   s -= m * 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:04.1f}"
    return f"{m:02d}:{s:04.1f}"

# ───────────────────────── Event scoring ─────────────────────────
def _activity_score(frame: np.ndarray, prev_frame: Optional[np.ndarray]) -> float:
    """
    Combine motion (frame difference) + edge density as a light heuristic.
    0..1 scale (rough).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / (gray.shape[0] * gray.shape[1])
    motion = 0.0
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        motion = float(diff.mean()) / 255.0
    return float(np.clip(motion * 0.65 + edge_density * 0.35, 0.0, 1.0))

def _group_events(samples: List[Tuple[int, float, float]],
                  sample_step_seconds: float,
                  min_duration: float,
                  merge_gap: float,
                  top_limit: int) -> List[TimeSegment]:
    """
    samples = [(frame_idx, timestamp, score)]
    Build segments where score >= threshold, merge close ones, keep top N by peak.
    """
    if not samples:
        return []
    # initial runs
    segments: List[TimeSegment] = []
    cur_start = samples[0][1]
    cur_end = cur_start
    cur_peak = samples[0][2]

    runs: List[Tuple[float, float, float]] = []  # (start, end, peak)
    for i in range(1, len(samples)):
        _, ts, sc = samples[i]
        _, prev_ts, _ = samples[i - 1]
        if ts - prev_ts > sample_step_seconds * 1.5:
            if cur_end - cur_start >= min_duration:
                runs.append((cur_start, cur_end, cur_peak))
            cur_start, cur_end, cur_peak = ts, ts, sc
        else:
            cur_end = ts
            if sc > cur_peak:
                cur_peak = sc
    # last run
    if cur_end - cur_start >= min_duration:
        runs.append((cur_start, cur_end, cur_peak))

    if not runs:
        return []

    # merge runs that are near each other
    merged: List[TimeSegment] = [TimeSegment(start=runs[0][0], end=runs[0][1], peak_score=runs[0][2])]
    for s, e, p in runs[1:]:
        last = merged[-1]
        if s - last.end <= merge_gap:
            last.end = max(last.end, e)
            if p is not None and (last.peak_score is None or p > last.peak_score):
                last.peak_score = p
        else:
            merged.append(TimeSegment(start=s, end=e, peak_score=p))

    # sort by peak score desc and limit
    merged.sort(key=lambda seg: seg.peak_score or 0.0, reverse=True)
    merged = merged[:max(1, top_limit)]
    # sort back by time for a readable summary
    merged.sort(key=lambda seg: seg.start)
    return merged

# ───────────────────────── Analysis core ─────────────────────────
async def analyze_video_for_events(video_path: str) -> Tuple[List[TimeSegment], str, float]:
    """
    Returns:
      events (segments), analysis_log (string), total_duration_seconds
    """
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

    hits: List[Tuple[int, float, float]] = []      # samples above EVENT_THRESHOLD
    all_scores: List[Tuple[int, float, float]] = []  # all sampled scores

    prev_for_motion = None
    frame_idx = 0
    log = [f"Frames={total_frames}, fps={fps:.2f}, duration={total_duration:.2f}s, step={step}"]

    while True:
        ok, frame = vid.read()
        if not ok:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        ts = frame_idx * frame_duration
        score = _activity_score(frame, prev_for_motion)
        all_scores.append((frame_idx, ts, float(score)))

        if score >= EVENT_THRESHOLD:
            hits.append((frame_idx, ts, float(score)))
            log.append(f"EVENT @ {ts:.2f}s (f{frame_idx}) score={score:.3f}")

        if frame_idx % (5 * step) == 0:
            prev_for_motion = frame

        frame_idx += 1

    vid.release()

    events = _group_events(
        hits,
        sample_step_seconds=frame_duration * step,
        min_duration=MIN_EVENT_DURATION,
        merge_gap=MERGE_GAP,
        top_limit=TOP_EVENTS_LIMIT,
    )

    log.append(f"Events above threshold: {len(hits)} → segments: {len(events)}")
    for seg in events:
        log.append(f"  {seg.start:.2f}-{seg.end:.2f} (peak={seg.peak_score:.3f if seg.peak_score else 0.0})")

    return events, "\n".join(log), total_duration

# ───────────────────────── Summarization text ─────────────────────────
def _summarize_events_text(events: List[TimeSegment], duration: float) -> str:
    if not events:
        return "No notably eventful moments were detected."

    # brief overview
    total_evt_time = sum(max(0.0, seg.end - seg.start) for seg in events)
    pct = (total_evt_time / max(1e-6, duration)) * 100.0
    header = (
        f"🎬 Video summary\n"
        f"- Duration: {_fmt_ts(duration)}\n"
        f"- Eventful time kept: {_fmt_ts(total_evt_time)} (~{pct:.1f}%)\n"
        f"- Segments detected: {len(events)}\n"
        f"\nKey moments:\n"
    )

    lines = []
    for i, seg in enumerate(events, 1):
        start = _fmt_ts(seg.start)
        end = _fmt_ts(seg.end)
        peak = f"{seg.peak_score:.2f}" if seg.peak_score is not None else "—"
        # Generic labels based on peak score
        if seg.peak_score is not None:
            if seg.peak_score >= 0.75:
                label = "High-intensity activity"
            elif seg.peak_score >= 0.55:
                label = "Elevated activity"
            else:
                label = "Notable motion"
        else:
            label = "Notable motion"

        lines.append(f"{i}. {start} → {end}  ({label}; peak={peak})")

    return header + "\n".join(lines)

# ───────────────────────── REST: /video/summarize ─────────────────────────
@agent.on_rest_post("/video/summarize", VideoSummarizeRequest, VideoSummarizeResult)
async def summarize_video(_: Context, body: VideoSummarizeRequest) -> VideoSummarizeResult:
    if not body.video_url:
        return VideoSummarizeResult(ok=False, message="Missing video_url")

    temp_dir = Path(tempfile.mkdtemp(prefix="vidsum_")).resolve()
    input_path = str(temp_dir / "input_video")  # any extension; OpenCV will probe

    try:
        await _download_file(body.video_url, input_path)

        events, analysis_log, duration = await analyze_video_for_events(input_path)
        if not events:
            return VideoSummarizeResult(
                ok=True,
                message="No high-activity segments detected.",
                summary="No notably eventful moments were detected.",
                events=[],
                analysis_log=analysis_log,
                duration_seconds=duration,
            )

        summary_text = _summarize_events_text(events, duration)
        return VideoSummarizeResult(
            ok=True,
            message="Summary generated.",
            summary=summary_text,
            events=events,
            analysis_log=analysis_log,
            duration_seconds=duration,
        )

    except Exception as e:
        return VideoSummarizeResult(ok=False, message=f"Processing failed: {e}")

    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

# ───────────────────────── Optional: typed protocol ─────────────────────────
sum_proto = Protocol(name="video-summarize-protocol", version="1.0.0")

@sum_proto.on_message(VideoSummarizeRequest)
async def on_sum_msg(ctx: Context, sender: str, req: VideoSummarizeRequest):
    res = await summarize_video(ctx, req)
    await ctx.send(sender, res)

agent.include(sum_proto)

# ───────────────────────── Inspector link ─────────────────────────
def print_inspector_link():
    addr = quote(agent.address, safe="")
    print(f"[{NAME}] Inspector: https://agentverse.ai/inspect/?uri=http://127.0.0.1:{PORT}&address={addr}")

if __name__ == "__main__":
    print(f"[{NAME}] Address: {agent.address}")
    print_inspector_link()
    agent.run()
