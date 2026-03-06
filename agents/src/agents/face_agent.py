from datetime import datetime
from uuid import uuid4
from urllib.parse import quote
import os

from uagents import Agent, Context, Protocol, Model
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement, ChatMessage, TextContent, chat_protocol_spec
)
from src.common.schemas import HealthStatus, ClipReady, FrameScored, FaceQuery, FaceMatch, FaceMatches

NAME = "face-detector"
SEED = os.getenv("FACE_AGENT_SEED", NAME)
PORT = int(os.getenv("FACE_AGENT_PORT", os.getenv("PORT", os.getenv("AGENT_PORT", "8103"))))

agent = Agent(name=NAME, seed=SEED, port=PORT, mailbox=True, publish_agent_details=True)

# Chat protocol (Inspector callable)
chat_proto = Protocol(spec=chat_protocol_spec)

@chat_proto.on_message(ChatMessage)
async def on_chat(ctx: Context, sender: str, msg: ChatMessage):
    text = next((c.text for c in msg.content if isinstance(c, TextContent)), "")
    await ctx.send(sender, ChatMessage(
        timestamp=datetime.utcnow(), msg_id=str(uuid4()),
        content=[TextContent(type="text", text=f"[face] pong: {text or 'ping'}")]
    ))

@chat_proto.on_message(ChatAcknowledgement)
async def on_chat_ack(_: Context, __: str, ___: ChatAcknowledgement):
    return

agent.include(chat_proto, publish_manifest=True)

# Optional tiny health proto (kept from original)
health_proto = Protocol(name="health-protocol", version="1.0.0")

class Ping(Model):
    msg: str = "ping"

class Pong(Model):
    msg: str
    at: str

@health_proto.on_message(Ping)
async def on_ping(ctx: Context, sender: str, _: Ping):
    await ctx.send(sender, Pong(msg="pong", at=datetime.utcnow().isoformat()))

agent.include(health_proto)

@agent.on_rest_get("/health", HealthStatus)
async def health(_: Context) -> HealthStatus:
    return HealthStatus(ok=True, name=NAME, addr=agent.address)

@agent.on_message(model=ClipReady)
async def on_clip(ctx: Context, sender: str, msg: ClipReady):
    await ctx.send(sender, FrameScored(
        frame_url=msg.clip_url, score=0.9, label="face",
        timestamp_utc=msg.start_utc or "", investigation_id=msg.investigation_id
    ))

@agent.on_rest_post("/faces/match", FaceQuery, FaceMatches)
async def rest_match(_: Context, body: FaceQuery) -> FaceMatches:
    return FaceMatches(matches=[FaceMatch(face_id="demo", similarity=0.99)])

def print_inspector_link():
    addr = quote(agent.address, safe="")
    print(f"[{NAME}] Inspector: https://agentverse.ai/inspect/?uri=http://127.0.0.1:{PORT}&address={addr}")

if __name__ == "__main__":
    print(f"[{NAME}] Address: {agent.address}")
    print_inspector_link()
    agent.run()
