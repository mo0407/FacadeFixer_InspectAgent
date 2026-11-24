"""P0-v2 Responses gateway with durable, exclusive raw response capture.

The original gateway still constructs the request, validates model identity and
parses the result. This adapter records response bytes before that parsing.
No request payload, authorization header, or response text is logged here.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

import httpx

from .model_gateway import GatewayError, ModelGateway


class _CaptureFailure(Exception):
    pass


class _CaptureTransport(httpx.AsyncBaseTransport):
    def __init__(self, underlying, fd):
        self.underlying = underlying
        self.fd = fd
        self.capture_failed = False
        self.capture_info = None

    async def handle_async_request(self, request):
        response = await self.underlying.handle_async_request(request)
        try:
            # aiter_raw yields encoded body bytes, prior to content decoding.
            if response.is_stream_consumed:
                if response.headers.get('content-encoding') not in (None, 'identity'):
                    raise _CaptureFailure('Encoded response was already consumed')
                raw = response.content
                if len(raw) > 16 * 1024 * 1024:
                    raise _CaptureFailure('Response exceeds the capture bound')
            else:
                chunks=[]; size=0
                async for chunk in response.aiter_raw():
                    size += len(chunk)
                    if size > 16 * 1024 * 1024:
                        raise _CaptureFailure('Response exceeds the capture bound')
                    chunks.append(chunk)
                raw = b''.join(chunks)
            remaining = memoryview(raw)
            while remaining:
                written = os.write(self.fd, remaining)
                if written <= 0:
                    raise OSError('Incomplete capture write')
                remaining = remaining[written:]
            os.fsync(self.fd)
            self.capture_info = {'response_body_sha256':hashlib.sha256(raw).hexdigest(),
                                 'response_body_size_bytes':len(raw),
                                 'response_content_encoding':response.headers.get('content-encoding')}
        except (OSError, _CaptureFailure):
            self.capture_failed = True
            await response.aclose()
            raise _CaptureFailure('Response capture failed') from None
        # Give the original gateway an independent response from those same
        # bytes, letting HTTPX handle content decoding exactly as before.
        rebuilt = httpx.Response(response.status_code, headers=response.headers,
                                 content=raw, request=request,
                                 extensions=response.extensions)
        await response.aclose()
        return rebuilt

    async def aclose(self):
        await self.underlying.aclose()


class RawCaptureGateway(ModelGateway):
    """A serialized Responses-only gateway with one exclusive file per call."""

    def __init__(self, config, *, allow_api=False, transport=None):
        if transport is None and any(os.environ.get(name) for name in (
                'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy')):
            raise GatewayError('Configured proxy is not supported by P0-v2 capture transport')
        super().__init__(config, allow_api=allow_api, transport=transport)
        self._capture_lock = asyncio.Lock()

    async def invoke(self, role, task, schema, *, capture_path, mock_output=None, image_paths=None):
        if role not in self.config.roles:
            raise GatewayError('Unknown role')
        if self.config.connections[self.config.roles[role]].adapter != 'responses':
            raise GatewayError('P0-v2 capture requires the Responses adapter')
        path = Path(capture_path)
        async with self._capture_lock:
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
                try:
                    os.fchmod(fd, 0o600)
                    # Persist the filename before a request can be sent.
                    dirfd = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
                    try:
                        os.fsync(dirfd)
                    finally:
                        os.close(dirfd)
                except Exception:
                    os.close(fd)
                    raise
            except (OSError, TypeError, ValueError):
                raise GatewayError('Cannot reserve exclusive response capture') from None
            underlying = self.transport or httpx.AsyncHTTPTransport()
            capture = _CaptureTransport(underlying, fd)
            worker = ModelGateway(self.config, allow_api=self.allow_api, transport=capture)
            worker._pinned_models = self._pinned_models
            try:
                return await worker.invoke(role, task, schema, mock_output=mock_output, image_paths=image_paths)
            except GatewayError:
                if capture.capture_failed:
                    if worker.trace:
                        worker.trace[-1]['status'] = 'request_outcome_unknown'
                        worker.trace[-1]['error_code'] = 'response_capture_failed'
                    raise GatewayError('Response capture failed; request outcome requires reconciliation') from None
                raise
            finally:
                if worker.trace and capture.capture_info is not None:
                    worker.trace[-1].update(capture.capture_info)
                self.trace.extend(worker.trace)
                os.close(fd)
