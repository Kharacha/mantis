from typing import Optional, List, Dict, Any
from uagents import Model

# ---------- Generic ----------
class HealthStatus(Model):
    ok: bool
    name: str
    addr: str

class ProcessResponse(Model):
    ok: bool
    message: str

class DetectionScore(Model):
    label: str
    score: float  # 0..1

# ---------- Core video/clip ----------
class ClipReady(Model):
    clip_url: str
    investigation_id: str
    start_utc: Optional[str] = None
    end_utc: Optional[str] = None

class FrameScored(Model):
    frame_url: str
    score: float            # 0..1
    label: str
    timestamp_utc: str      # ISO8601
    investigation_id: str
    face_id: Optional[str] = None

# ---------- Face search ----------
class FaceQuery(Model):
    image_url: str
    top_k: int = 5

class FaceMatch(Model):
    face_id: str
    similarity: float
    first_seen_utc: Optional[str] = None
    last_seen_utc: Optional[str] = None

class FaceMatches(Model):
    matches: List[FaceMatch] = []

# ---------- Transcription ----------
class TranscriptRequest(Model):
    audio_url: str
    lang: Optional[str] = None

class TranscriptReady(Model):
    text: str
    confidence: float
    start_utc: Optional[str] = None
    end_utc: Optional[str] = None

# ---------- Summarization ----------
class SummaryRequest(Model):
    clip_url: str
    transcript: Optional[str] = None

class SummaryReady(Model):
    summary: str
    keywords: List[str] = []

# ---------- Chunking ----------
class ChunkRequest(Model):
    video_url: str
    max_seconds: int = 30

class ChunkResult(Model):
    chunks: List[str]

# ---------- Timeline / Events ----------
class EventLog(Model):
    event: str
    frame_id: Optional[int] = None
    object: Optional[str] = None
    face_id: Optional[str] = None
    timestamp_utc: str
    confidence: Optional[float] = None
    extra: Dict[str, Any] = {}

class TimelineWindow(Model):
    label: str
    start_utc: str
    end_utc: str
    confidence: float

class TimelineBatch(Model):
    investigation_id: str
    windows: List[TimelineWindow]

# ---------- Search ----------
class SearchQuery(Model):
    query: str
    filters: Dict[str, Any] = {}

class SearchResults(Model):
    query: str
    results: List[Dict[str, Any]] = []

# ---------- Alerts ----------
class Alert(Model):
    alert_type: str
    priority: str
    video_clip_url: Optional[str] = None
    recipients: List[str] = []
    meta: Dict[str, Any] = {}

# ---------- Timestamping ----------
class TimestampRequest(Model):
    video_url: str
    prompt: str = "Find key events with times."

class TimestampResult(Model):
    events: List[Dict[str, Any]] = []

# --- Voice / TTS ---
class TTSRequest(Model):
    text: str
