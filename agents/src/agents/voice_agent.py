from datetime import datetime
from uuid import uuid4
from urllib.parse import quote
import os, httpx

from uagents import Agent, Context, Protocol, Model
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement, ChatMessage, TextContent, chat_protocol_spec
)
from src.common.schemas import HealthStatus, TTSRequest, ProcessResponse

NAME = "voice-agent"
SEED = os.getenv("VOICE_AGENT_SEED", NAME)
PORT = int(os.getenv("VOICE_AGENT_PORT", os.getenv("PORT", os.getenv("AGENT_PORT", "8105"))))
TTS_ENDPOINT = os.getenv("VOICE_TTS_ENDPOINT")

agent = Agent(name=NAME, seed=SEED, port=PORT, mailbox=True, publish_agent_details=True)

chat_proto = Protocol(spec=chat_protocol_spec)

@chat_proto.on_message(ChatMessage)
async def on_chat(ctx: Context, sender: str, msg: ChatMessage):
    text = next((c.text for c in msg.content if isinstance(c, TextContent)), "")
    await ctx.send(sender, ChatMessage(
        timestamp=datetime.utcnow(), msg_id=str(uuid4()),
        content=[TextContent(type="text", text=f"[voice] pong: {text or 'ping'}")]
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

@agent.on_rest_post("/voice/tts", TTSRequest, ProcessResponse)
async def tts(_: Context, body: TTSRequest) -> ProcessResponse:
    if not body.text:
        return ProcessResponse(ok=False, message="Missing 'text'")
    if TTS_ENDPOINT:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(TTS_ENDPOINT, json={"text": body.text})
            r.raise_for_status()
            return ProcessResponse(ok=True, message=r.text)
    return ProcessResponse(ok=True, message="no-op (VOICE_TTS_ENDPOINT not set)")

def print_inspector_link():
    addr = quote(agent.address, safe="")
    print(f"[{NAME}] Inspector: https://agentverse.ai/inspect/?uri=http://127.0.0.1:{PORT}&address={addr}")

if __name__ == "__main__":
    print(f"[{NAME}] Address: {agent.address}")
    print_inspector_link()
    agent.run()
