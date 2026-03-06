from datetime import datetime
from uuid import uuid4
from urllib.parse import quote
import os

from uagents import Agent, Context, Protocol, Model
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

from src.common.schemas import HealthStatus, EventLog

NAME = "timeline-ingest"

# Give this agent its own stable DID; override with env if you want a custom one
SEED = os.getenv("TIMELINE_INGEST_SEED", NAME)

# Only used if run standalone (not via Bureau)
PORT = int(os.getenv("AGENT_PORT", "8203"))

agent = Agent(
    name=NAME,
    seed=SEED,
    port=PORT,
    mailbox=True,
    publish_agent_details=True,
)

# --- Publish a callable protocol (chat) so Inspector can talk to it ---
chat_proto = Protocol(spec=chat_protocol_spec)

@chat_proto.on_message(ChatMessage)
async def on_chat(ctx: Context, sender: str, msg: ChatMessage):
    text = next((c.text for c in msg.content if isinstance(c, TextContent)), "")
    await ctx.send(
        sender,
        ChatMessage(
            timestamp=datetime.utcnow(),
            msg_id=str(uuid4()),
            content=[TextContent(type="text", text=f"[timeline] pong: {text or 'ping'}")],
        ),
    )

@chat_proto.on_message(ChatAcknowledgement)
async def on_chat_ack(_: Context, __: str, ___: ChatAcknowledgement):
    return

agent.include(chat_proto, publish_manifest=True)

# --- Optional: a tiny custom Ping/Pong protocol (kept for parity with your original) ---
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

# --- REST health (handy for local curl checks if run standalone) ---
@agent.on_rest_get("/health", HealthStatus)
async def health(_: Context) -> HealthStatus:
    return HealthStatus(ok=True, name=NAME, addr=agent.address)

# --- Domain handler: EventLog ---
@agent.on_message(model=EventLog)
async def on_event(ctx: Context, sender: str, evt: EventLog):
    ctx.logger.info(f"[{NAME}] event={evt.event} at {evt.timestamp_utc}")

# --- Inspector link helper (uses Bureau URI when in Bureau mode) ---
def print_inspector_link(bureau_uri: str = "http://127.0.0.1:8000"):
    addr = quote(agent.address, safe="")
    print(f"[{NAME}] Inspector: https://agentverse.ai/inspect/?uri={bureau_uri}&address={addr}")

if __name__ == "__main__":
    print(f"Address: {agent.address}")
    # Standalone Inspector link (uses this agent's own port)
    print_inspector_link(bureau_uri=f"http://127.0.0.1:{PORT}")
    agent.run()
