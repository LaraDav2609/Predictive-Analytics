"""Shared outbound HTTP/SSL setup for data clients.

Builds httpx clients whose TLS verification uses the **operating-system
certificate store** instead of httpx's bundled ``certifi`` roots.

Why: on managed/corporate networks that terminate TLS with an internal root CA
(common on Windows work machines), ``certifi`` does not contain that root, so
every outbound request fails ``[SSL: CERTIFICATE_VERIFY_FAILED] unable to get
local issuer certificate``. The OS trust store *does* contain the corporate root.
``ssl.create_default_context()`` still performs full certificate verification — it
simply also trusts the roots installed in the OS store (which on Windows includes
the system ROOT/CA stores), so genuine cert failures are still rejected.
"""
from __future__ import annotations

import ssl
from functools import lru_cache

import httpx


@lru_cache(maxsize=1)
def os_trust_ssl_context() -> ssl.SSLContext:
    """A verifying SSL context backed by the OS certificate store (cached)."""
    return ssl.create_default_context()


def make_async_client(**kwargs) -> httpx.AsyncClient:
    """``httpx.AsyncClient`` that trusts the OS certificate store by default.

    Accepts the same keyword arguments as ``httpx.AsyncClient``; callers may still
    override ``verify`` explicitly if they need different behaviour.
    """
    kwargs.setdefault("verify", os_trust_ssl_context())
    return httpx.AsyncClient(**kwargs)
