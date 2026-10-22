"""Identity supplied by the trusted Authelia forward-auth proxy."""

import hmac

from fastapi import Depends, HTTPException, Request, status

from .config import Settings, get_settings


def sso_user(request: Request, settings: Settings) -> str | None:
    """Return the identity vouched for by Caddy, or None.

    Caddy overwrites client-supplied identity headers and adds the shared
    proxy secret only after Authelia authorizes the request.
    """
    secret = settings.sso_proxy_secret
    if not secret:
        return None
    sent = request.headers.get("x-sso-proxy", "")
    if not hmac.compare_digest(sent.encode(), secret.encode()):
        return None
    name = request.headers.get("remote-user", "").strip()
    if not name:
        return None
    return settings.sso_names.get(name.lower(), name)


def current_user(
    request: Request, settings: Settings = Depends(get_settings)
) -> str:
    """FastAPI dependency returning the authenticated username, or raising 401."""
    name = sso_user(request, settings)
    if not name:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not signed in"
        )
    return name


