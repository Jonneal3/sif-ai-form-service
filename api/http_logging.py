from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("api.http")


_SENSITIVE_KEYS = {
    "authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "token",
    "secret",
    "password",
    "openai_api_key",
    "groq_api_key",
    "supabase_service_role_key",
}


def _env_bool(name: str, default: bool = False) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            key = str(k).lower()
            if key in _SENSITIVE_KEYS:
                out[k] = "***"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _decode_headers(headers: Optional[Iterable[Tuple[bytes, bytes]]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not headers:
        return out
    for k, v in headers:
        try:
            ks = k.decode("latin-1").lower()
        except Exception:
            continue
        try:
            vs = v.decode("latin-1")
        except Exception:
            vs = "<binary>"
        if ks in _SENSITIVE_KEYS:
            vs = "***"
        out[ks] = vs
    return out


def _content_type(headers: Optional[Iterable[Tuple[bytes, bytes]]]) -> str:
    if not headers:
        return ""
    for k, v in headers:
        if k.lower() == b"content-type":
            try:
                return v.decode("latin-1")
            except Exception:
                return ""
    return ""


def _parse_body(content_type: str, body: bytes) -> Any:
    ct = (content_type or "").lower()
    if "application/json" in ct:
        try:
            return _redact(json.loads(body.decode("utf-8", errors="replace")))
        except Exception:
            return body.decode("utf-8", errors="replace")
    if "application/x-www-form-urlencoded" in ct:
        return body.decode("utf-8", errors="replace")
    if "multipart/form-data" in ct:
        return "<multipart>"
    if ct.startswith("text/"):
        return body.decode("utf-8", errors="replace")
    if not body:
        return ""
    return "<binary>"


class HttpLoggingMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        log_headers: bool,
        max_body_bytes: int,
    ) -> None:
        self.app = app
        self.log_headers = log_headers
        self.max_body_bytes = max(0, max_body_bytes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        started_at = time.perf_counter()
        request_id = _get_request_id(scope) or uuid.uuid4().hex[:12]

        req_headers_list: List[Tuple[bytes, bytes]] = list(scope.get("headers") or [])
        req_headers = _decode_headers(req_headers_list) if self.log_headers else {}
        req_ct = _content_type(req_headers_list)

        req_body_buf = bytearray()
        req_body_truncated = False

        res_headers_list: List[Tuple[bytes, bytes]] = []
        res_status: Optional[int] = None
        res_ct = ""
        res_body_buf = bytearray()
        res_body_truncated = False

        async def receive_wrapped() -> Message:
            nonlocal req_body_truncated
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body") or b""
                if body and self.max_body_bytes > 0 and not req_body_truncated:
                    remaining = self.max_body_bytes - len(req_body_buf)
                    if remaining > 0:
                        req_body_buf.extend(body[:remaining])
                    if len(body) > remaining:
                        req_body_truncated = True
            return message

        async def send_wrapped(message: Message) -> None:
            nonlocal res_status, res_headers_list, res_ct, res_body_truncated
            if message.get("type") == "http.response.start":
                res_status = int(message.get("status") or 0)
                res_headers_list = list(message.get("headers") or [])
                res_ct = _content_type(res_headers_list)
            elif message.get("type") == "http.response.body":
                body = message.get("body") or b""
                if body and self.max_body_bytes > 0 and not res_body_truncated:
                    remaining = self.max_body_bytes - len(res_body_buf)
                    if remaining > 0:
                        res_body_buf.extend(body[:remaining])
                    if len(body) > remaining:
                        res_body_truncated = True
            await send(message)

        err: Optional[BaseException] = None
        try:
            await self.app(scope, receive_wrapped, send_wrapped)
        except BaseException as e:  # noqa: BLE001 - we want to log then re-raise
            err = e
            raise
        finally:
            dur_ms = int((time.perf_counter() - started_at) * 1000)
            path = str(scope.get("path") or "")
            query_string = scope.get("query_string") or b""
            query = query_string.decode("latin-1", errors="ignore")
            method = str(scope.get("method") or "").upper()

            req_body = _parse_body(req_ct, bytes(req_body_buf)) if self.max_body_bytes else ""
            res_body = _parse_body(res_ct, bytes(res_body_buf)) if self.max_body_bytes else ""
            res_headers = _decode_headers(res_headers_list) if self.log_headers else {}

            record = {
                "id": request_id,
                "method": method,
                "path": path,
                "query": query,
                "status": res_status,
                "dur_ms": dur_ms,
                "request": {
                    "content_type": req_ct,
                    "headers": req_headers,
                    "body": req_body,
                    "body_truncated": req_body_truncated,
                },
                "response": {
                    "content_type": res_ct,
                    "headers": res_headers,
                    "body": res_body,
                    "body_truncated": res_body_truncated,
                },
            }
            if err is not None:
                record["error"] = {"type": type(err).__name__, "message": str(err)}

            # One-line JSON for easy grepping in server logs.
            try:
                logger.info(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            except Exception:
                logger.info("%s %s %s status=%s dur_ms=%s", request_id, method, path, res_status, dur_ms)


def _get_request_id(scope: Scope) -> Optional[str]:
    headers: Iterable[Tuple[bytes, bytes]] = scope.get("headers") or []
    for k, v in headers:
        if k.lower() == b"x-request-id":
            try:
                return v.decode("latin-1")
            except Exception:
                return None
    return None


def install_http_logging(app: Any) -> None:
    """
    Enable request/response logging via env vars.

    - `AI_FORM_HTTP_LOG=1` enables middleware
    - `AI_FORM_HTTP_LOG_HEADERS=1` logs request/response headers (redacted)
    - `AI_FORM_HTTP_LOG_BODY_MAX_BYTES=4096` caps body bytes captured per request/response
    """
    if not _env_bool("AI_FORM_HTTP_LOG", default=False):
        return
    log_headers = _env_bool("AI_FORM_HTTP_LOG_HEADERS", default=False)
    max_body_bytes = _env_int("AI_FORM_HTTP_LOG_BODY_MAX_BYTES", default=4096)
    app.add_middleware(HttpLoggingMiddleware, log_headers=log_headers, max_body_bytes=max_body_bytes)
