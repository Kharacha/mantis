from datetime import datetime
from uuid import uuid4
from urllib.parse import quote
from typing import List
import os

from uagents import Agent, Context, Protocol, Model
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement, ChatMessage, TextContent, chat_protocol_spec
)

NAME = "video-process-agent"
SEED = os.getenv("VIDEO_AGENT_SEED", NAME)
PORT = int(os.getenv("VIDEO_AGENT_PORT", os.getenv("PORT", "8009")))

agent = Agent(name=NAME, seed=SEED, port=PORT, mailbox=True, publish_agent_details=True)

# Chat protocol (for inspector)
chat_proto = Protocol(spec=chat_protocol_spec)

@chat_proto.on_message(ChatMessage)
async def on_chat(ctx: Context, sender: str, msg: ChatMessage):
    text = next((c.text for c in msg.content if isinstance(c, TextContent)), "")
    await ctx.send(sender, ChatMessage(
        timestamp=datetime.utcnow(), msg_id=str(uuid4()),
        content=[TextContent(type="text", text=f"[video] pong: {text or 'ping'}")]
    ))

@chat_proto.on_message(ChatAcknowledgement)
async def on_chat_ack(_: Context, __: str, ___: ChatAcknowledgement):
    return

agent.include(chat_proto, publish_manifest=True)

# Tiny health proto (optional)
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

# Models (kept from original)
class Health(Model):
    ok: bool
    name: str

class VideoBatch(Model):
    urls: List[str]
    investigation_id: str

class ProcessResponse(Model):
    ok: bool
    message: str

@agent.on_rest_get("/health", Health)
async def health(_: Context) -> dict:
    return {"ok": True, "name": NAME}

@agent.on_rest_post("/process-videos", VideoBatch, ProcessResponse)
async def process_videos(ctx: Context, req: VideoBatch) -> ProcessResponse:
    ctx.logger.info(f"[video] Received {len(req.urls)} videos for {req.investigation_id}")
    return ProcessResponse(ok=True, message=f"{len(req.urls)} videos received")

def print_inspector_link():
    from urllib.parse import quote
    addr = quote(agent.address, safe="")
    print(f"[{NAME}] Inspector: https://agentverse.ai/inspect/?uri=http://127.0.0.1:{PORT}&address={addr}")

if __name__ == "__main__":
    print(f"[{NAME}] Address: {agent.address}")
    print_inspector_link()
    agent.run()
