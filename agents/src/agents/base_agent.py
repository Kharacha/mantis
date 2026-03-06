from uagents import Agent, Context
from uagents.setup import fund_agent_if_low
from src.common.settings import settings
from uagents import Model
class HealthStatus(Model):
    ok: bool
    name: str
    addr: str
class BaseAgent(Agent):
    def __init__(
        self,
        name: str,
        *,
        port: int | None = None,
        mailbox: bool = False,
        publish_agent_details: bool = True,
    ):
        super().__init__(
            name=name,
            port=port if port is not None else settings.port,
            mailbox=mailbox,
            publish_agent_details=publish_agent_details,
        )
        # testnet funds so registration/mailbox works
        fund_agent_if_low(self.wallet.address())

    def add_health(self):
        @self.on_rest_get("/health",response=HealthStatus)
        async def _health(_: Context):
            return HealthStatus(ok=True, name=self.name, addr=self.address)
