from datetime import datetime
from uuid import uuid4
import os
from urllib.parse import quote

from uagents import Agent, Context, Protocol, Model
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

from src.common.schemas import HealthStatus, SearchQuery, SearchResults

# ---- Identity / addressing ----
NAME = "search-query"
# Give this agent its own stable DID (change the env var value per agent)
SEED = os.getenv("SEARCH_AGENT_SEED", NAME)

# Port only matters when you run this agent standalone.
# In Bureau mode the Bureau hosts on its own port (e.g., 8000).
PORT = int(os.getenv("AGENT_PORT", "8201"))

agent = Agent(
    name=NAME,
    seed=SEED,
    port=PORT,
    mailbox=True,
    publish_agent_details=True,
)

# ---- Publish a known, callable protocol (chat) ----
chat_proto = Protocol(spec=chat_protocol_spec)

@chat_proto.on_message(ChatMessage)
async def on_chat(ctx: Context, sender: str, msg: ChatMessage):
    text = next((c.text for c in msg.content if isinstance(c, TextContent)), "")
    await ctx.send(
        sender,
        ChatMessage(
            timestamp=datetime.utcnow(),
            msg_id=str(uuid4()),
            content=[TextContent(type="text", text=f"pong: {text or 'ping'}")],
        ),
    )

# Some versions require acknowledging acks to pass spec verification
@chat_proto.on_message(ChatAcknowledgement)
async def on_chat_ack(_: Context, __: str, ___: ChatAcknowledgement):
    return

agent.include(chat_proto, publish_manifest=True)

# ---- REST & custom handlers (optional but handy) ----
@agent.on_rest_get("/health", HealthStatus)
async def health(_: Context) -> HealthStatus:
    return HealthStatus(ok=True, name=NAME, addr=agent.address)

@agent.on_message(model=SearchQuery)
async def on_search(ctx: Context, sender: str, q: SearchQuery):
    # TODO: do your search; stub returns empty list
    await ctx.send(sender, SearchResults(query=q.query, results=[]))

# ---- Helper to print the Inspector link for Bureau mode ----
def print_inspector_link(bureau_uri: str = "http://127.0.0.1:8000"):
    # When running inside a Bureau, ALWAYS use the Bureau URI in the inspector link
    addr = quote(agent.address, safe="")
    print(f"[search-query] Inspector: https://agentverse.ai/inspect/?uri={bureau_uri}&address={addr}")

if __name__ == "__main__":
    # Standalone mode (not Bureau): bind to PORT and serve
    print(f"Address: {agent.address}")
    print_inspector_link(bureau_uri=f"http://127.0.0.1:{PORT}")
    agent.run()
