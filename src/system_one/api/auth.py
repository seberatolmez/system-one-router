"""Authentication dependencies for the HTTP API."""

from secrets import compare_digest
from typing import Annotated

from fastapi import Header, HTTPException, status

from system_one.core.config import get_settings


async def require_api_key(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Require a configured Bearer token for protected API routes."""
    expected_key = get_settings().api_key
    if expected_key is None:
        return

    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token or not compare_digest(token, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
