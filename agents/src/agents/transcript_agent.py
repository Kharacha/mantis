from datetime import datetime
from uuid import uuid4
from urllib.parse import quote
import os, httpx

from uagents import Agent, Context, Protocol, Model
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement, ChatMessage, TextContent, chat_protocol_spec
)
from src.common.schemas import HealthStatus, TranscriptRequest, TranscriptReady

NAME = "transcript-maker"
SEED = os.getenv("TRANSCRIPT_AGENT_SEED", NAME)
PORT = int(os.getenv("TRANSCRIPT_AGENT_PORT", os.getenv("PORT", os.getenv("AGENT_PORT", "8104"))))
ASR_ENDPOINT = os.getenv("ASR_ENDPOINT")

agent = Agent(name=NAME, seed=SEED, port=PORT, mailbox=True, publish_agent_details=True)

chat_proto = Protocol(spec=chat_protocol_spec)

@chat_proto.on_message(ChatMessage)
async def on_chat(ctx: Context, sender: str, msg: ChatMessage):
    text = next((c.text for c in msg.content if isinstance(c, TextContent)), "")
    await ctx.send(sender, ChatMessage(
        timestamp=datetime.utcnow(), msg_id=str(uuid4()),
        content=[TextContent(type="text", text=f"[transcript] pong: {text or 'ping'}")]
    ))

@chat_proto.on_message(ChatAcknowledgement)
async def on_chat_ack(_: Context, __: str, ___: ChatAcknowledgement):
    return

agent.include(chat_proto, publish_manifest=True)

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

@agent.on_rest_post("/asr/transcribe", TranscriptRequest, TranscriptReady)
async def transcribe(_: Context, body: TranscriptRequest) -> TranscriptReady:
    if ASR_ENDPOINT:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(ASR_ENDPOINT, json=body.model_dump())
            r.raise_for_status()
            return TranscriptReady(**r.json())
    return TranscriptReady(text="", confidence=0.0)

def print_inspector_link():
    addr = quote(agent.address, safe="")
    print(f"[{NAME}] Inspector: https://agentverse.ai/inspect/?uri=http://127.0.0.1:{PORT}&address={addr}")

if __name__ == "__main__":
    print(f"[{NAME}] Address: {agent.address}")
    print_inspector_link()
    agent.run()
