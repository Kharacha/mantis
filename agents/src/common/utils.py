from typing import Optional
from fastapi import HTTPException

def auth_guard(token_header: Optional[str], expected: Optional[str]):
    if expected and token_header != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Unauthorized")
