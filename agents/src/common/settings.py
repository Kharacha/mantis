import os
from dataclasses import dataclass

@dataclass
class Settings:
    # default port for helpers unless overridden by each agent
    port: int = int(os.getenv("PORT", "8000"))

settings = Settings()
