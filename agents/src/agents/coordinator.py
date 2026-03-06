import os
from uagents import Agent, Context
from src.common.schemas import HealthStatus, ClipReady, FrameScored, Alert, ProcessResponse

NAME = "coordinator"
PORT = int(os.getenv("PORT", os.getenv("AGENT_PORT", "8000")))

agent = Agent(
    name=NAME,
    seed=NAME,
    port=PORT,
    mailbox=True,
    publish_agent_details=True,
)

# Example wiring placeholders
WEAPON_ADDR = os.getenv("WEAPON_ADDR")
ROBBERY_ADDR = os.getenv("ROBBERY_ADDR")
FACE_ADDR = os.getenv("FACE_ADDR")
TS_ADDR = os.getenv("TS_ADDR")
SEARCH_ADDR = os.getenv("SEARCH_ADDR")
NOTIFY_ADDR = os.getenv("NOTIFY_ADDR")
TIMELINE_ADDR = os.getenv("TIMELINE_ADDR")

@agent.on_rest_get("/health", HealthStatus)
async def health(_: Context) -> HealthStatus:
    return HealthStatus(ok=True, name=NAME, addr=agent.address)

@agent.on_rest_post("/ingest/clip", ClipReady, ProcessResponse)
async def ingest_clip(ctx: Context, body: ClipReady) -> ProcessResponse:
    sent = 0
    for target in [WEAPON_ADDR, ROBBERY_ADDR, FACE_ADDR]:
        if target:
            await ctx.send(target, body)
            sent += 1
    return ProcessResponse(ok=True, message=f"Dispatched to {sent} detectors")

@agent.on_message(model=FrameScored)
async def on_score(ctx: Context, sender: str, msg: FrameScored):
    if msg.label == "weapon" and msg.score >= 0.9 and NOTIFY_ADDR:
        await ctx.send(NOTIFY_ADDR, Alert(
            alert_type="weapon",
            priority="high",
            video_clip_url=msg.frame_url,
            recipients=["owner"],
            meta={"investigation_id": msg.investigation_id, "score": msg.score},
        ))
