"""
schemas/common.py
=================
Response models shared across every router — primarily the error
envelope every non-2xx response uses, registered once in `api/errors.py`.

Every future phase (Documents, Chat, Search, ...) reuses `ErrorResponse`
in its route `responses={...}` declarations rather than inventing a
per-resource error shape, so the frontend only ever has to parse one
error format regardless of which endpoint failed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    type: str = Field(..., description="The DocIntelError subclass name, e.g. 'ValidationError'.")
    message: str = Field(..., description="Human-readable, safe-to-display error message.")


class ErrorResponse(BaseModel):
    error: ErrorDetail
