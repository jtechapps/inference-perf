# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native llm-d-async Redis SortedSet client for inference-perf.

Implements the atomic SortedSet Lua submission contract from
``llm-d-async/producer/redis_sortedset_producer.go`` and consumes completion
``ResultMessage`` envelopes via ``BRPOP`` using a zero-dependency asyncio RESP2
transport. Picklable across ``forkserver`` worker processes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import secrets
import socket
import ssl
import time
import urllib.parse
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from aiohttp import ClientResponse
from multidict import CIMultiDict, CIMultiDictProxy

from inference_perf.apis import (
    ErrorResponseInfo,
    InferenceAPIData,
    InferenceInfo,
    RequestLifecycleMetric,
)
from inference_perf.apis.base import is_stage_teardown_cancelled, stage_teardown_cancelled_error
from inference_perf.apis.response_errors import InvalidResponseError
from inference_perf.apis.tool_schema import clean_request_tool_schemas
from inference_perf.client.modelserver.base import ModelServerClient, ModelServerClientSession
from inference_perf.client.modelserver.metrics import CounterMetric, GaugeMetric, HistogramMetric
from inference_perf.client.modelserver.openai_client import (
    OpenAIMetrics,
    _strip_embeddings_vectors,
    _update_headers_case_insensitive,
)
from inference_perf.config import APIConfig, APIType, CustomTokenizerConfig, LLMDAsyncRedisConfig, MultiLoRAConfig
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.utils import CustomTokenizer

logger = logging.getLogger(__name__)

# Atomic SortedSet enqueue script matching llm-d-async/producer/redis_sortedset_producer.go
SUBMIT_REQUEST_LUA = """
local seq = redis.call("INCR", KEYS[1])
redis.call("EXPIREAT", KEYS[1], ARGV[3])
redis.call("DEL", KEYS[2])
redis.call("SET", KEYS[3], ARGV[1], "PX", ARGV[2])
local score = tonumber(ARGV[6]) + math.min(seq, 2097151) / 2097152
redis.call("ZADD", KEYS[4], string.format("%.17g", score), ARGV[4] .. seq .. ARGV[5])
if KEYS[5] then
  redis.call("SET", KEYS[5], ARGV[7], "PX", ARGV[8])
end
return seq
""".strip()

_ENQUEUE_SEQ_FIELD_JSON = '"enqueue_seq":'
_ENQUEUE_SEQ_PLACEHOLDER_JSON = _ENQUEUE_SEQ_FIELD_JSON + "-1"
_ENQUEUE_SEQ_GRACE_SECONDS = 3600
_WORKER_RELAY_TTL_SECONDS = 300
_REQ_ID_PREFIX = "ip-"


class RedisProtocolError(RuntimeError):
    """Raised when a Redis server returns an error reply or malformed RESP2 frame."""


class AsyncResp2RedisClient:
    """Minimal asyncio RESP2 TCP client requiring no external Redis package."""

    def __init__(
        self,
        redis_url: str,
        password_override: Optional[str] = None,
        connect_timeout: float = 10.0,
    ) -> None:
        normalized = redis_url if "://" in redis_url else f"redis://{redis_url}"
        parsed = urllib.parse.urlparse(normalized)
        self.scheme = parsed.scheme.lower()
        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or 6379
        self.username = urllib.parse.unquote(parsed.username) if parsed.username else None
        url_password = urllib.parse.unquote(parsed.password) if parsed.password else None
        self.password = password_override if password_override is not None else url_password
        db_str = (parsed.path or "").lstrip("/")
        self.db = int(db_str) if db_str.isdigit() else 0
        self.connect_timeout = connect_timeout

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        if self._writer is not None and not self._writer.is_closing():
            return
        ssl_ctx: Optional[ssl.SSLContext] = None
        if self.scheme == "rediss":
            ssl_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port, ssl=ssl_ctx),
            timeout=self.connect_timeout,
        )
        sock = writer.get_extra_info("socket")
        if sock is not None:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
        self._reader = reader
        self._writer = writer
        if self.password is not None:
            if self.username:
                await self._send_unlocked("AUTH", self.username, self.password)
            else:
                await self._send_unlocked("AUTH", self.password)
        if self.db:
            await self._send_unlocked("SELECT", str(self.db))

    async def close(self) -> None:
        writer = self._writer
        self._writer = None
        self._reader = None
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except OSError:
                pass

    @staticmethod
    def encode_command(*args: str | bytes | int | float) -> bytes:
        parts = [f"*{len(args)}\r\n".encode("ascii")]
        for arg in args:
            if isinstance(arg, bytes):
                raw = arg
            elif isinstance(arg, str):
                raw = arg.encode("utf-8")
            else:
                raw = str(arg).encode("ascii")
            parts.append(f"${len(raw)}\r\n".encode("ascii"))
            parts.append(raw)
            parts.append(b"\r\n")
        return b"".join(parts)

    async def _read_line(self) -> bytes:
        assert self._reader is not None
        line = await self._reader.readline()
        if not line:
            raise ConnectionError("Redis connection closed unexpectedly")
        if not line.endswith(b"\r\n"):
            raise RedisProtocolError(f"Malformed RESP2 line: {line!r}")
        return line[:-2]

    async def _read_reply(self) -> Any:
        assert self._reader is not None
        line = await self._read_line()
        if not line:
            raise RedisProtocolError("Empty RESP2 frame")
        prefix = line[:1]
        payload = line[1:]
        if prefix == b"+":
            return payload.decode("utf-8", errors="replace")
        if prefix == b"-":
            raise RedisProtocolError(payload.decode("utf-8", errors="replace"))
        if prefix == b":":
            return int(payload)
        if prefix == b"$":
            length = int(payload)
            if length < 0:
                return None
            data = await self._reader.readexactly(length + 2)
            return data[:-2].decode("utf-8", errors="replace")
        if prefix == b"*":
            count = int(payload)
            if count < 0:
                return None
            return [await self._read_reply() for _ in range(count)]
        raise RedisProtocolError(f"Unknown RESP2 prefix: {prefix!r}")

    async def _send_unlocked(self, *args: str | bytes | int | float) -> Any:
        if self._writer is None or self._writer.is_closing():
            await self.connect()
        assert self._writer is not None
        self._writer.write(self.encode_command(*args))
        await self._writer.drain()
        return await self._read_reply()

    async def execute(self, *args: str | bytes | int | float) -> Any:
        async with self._lock:
            return await self._send_unlocked(*args)

    async def ping(self) -> bool:
        return bool(await self.execute("PING") in ("PONG", True))

    async def eval_script(self, script: str, keys: Sequence[str], argv: Sequence[str | int | float]) -> Any:
        return await self.execute("EVAL", script, len(keys), *keys, *argv)

    async def brpop(self, queue_names: Sequence[str], timeout_seconds: int = 1) -> Optional[Tuple[str, str]]:
        res = await self.execute("BRPOP", *queue_names, int(max(timeout_seconds, 1)))
        if not isinstance(res, list) or len(res) != 2:
            return None
        return str(res[0]), str(res[1])

    async def rpush_with_ttl(self, key: str, value: str, ttl_seconds: int) -> None:
        await self.execute("RPUSH", key, value)
        await self.execute("EXPIRE", key, ttl_seconds)

    async def delete(self, *keys: str) -> int:
        if not keys:
            return 0
        return int(await self.execute("DEL", *keys) or 0)


def compute_queue_score(deadline_unix: int, seq: int) -> float:
    """Mirror api.InternalRequest.QueueScore() from llm-d-async/api/internal_api.go."""
    clamped = min(max(int(seq), 0), (1 << 21) - 1)
    return float(deadline_unix) + float(clamped) / float(1 << 21)


def _resolve_deadline_offset_seconds(
    api_config: APIConfig,
    combined_headers: Mapping[str, str],
    default_deadline_seconds: int,
) -> int:
    """Resolve per-request deadline offset in seconds from headers or default_deadline_seconds."""
    lowered = {k.lower(): v for k, v in combined_headers.items()}
    raw_unit = getattr(api_config, "slo_unit", None)
    configured_unit = raw_unit if isinstance(raw_unit, str) and raw_unit else None
    unit_to_s = {"s": 1.0, "ms": 0.001, "us": 0.000001}

    custom_lat_header = getattr(api_config, "slo_request_latency_header", None)
    if isinstance(custom_lat_header, str) and custom_lat_header:
        val_str = lowered.get(custom_lat_header.lower())
        if val_str is not None and val_str != "default":
            unit = (
                "s"
                if configured_unit is None and custom_lat_header.lower().endswith(("-seconds", "-s", "_seconds", "_sec"))
                else (configured_unit or "ms")
            )
            try:
                parsed_sec = float(val_str) * unit_to_s.get(unit.lower(), 1.0)
                return max(int(math.ceil(parsed_sec)), 1)
            except ValueError:
                logger.warning(f"Invalid request deadline header value: {val_str}")

    timeout_hdr = lowered.get("x-llm-d-request-timeout-seconds")
    if timeout_hdr is not None and timeout_hdr != "default":
        try:
            return max(int(math.ceil(float(timeout_hdr))), 1)
        except ValueError:
            logger.warning(f"Invalid x-llm-d-request-timeout-seconds value: {timeout_hdr}")

    return int(default_deadline_seconds)


def build_sortedset_enqueue_command(
    *,
    req_id: str,
    created_unix: int,
    deadline_unix: int,
    payload: Mapping[str, Any],
    request_queue_name: str,
    result_queue_name: str,
    endpoint: str,
    model_name: str,
    metadata: Optional[Mapping[str, str]] = None,
    headers: Optional[Mapping[str, str]] = None,
    request_token: Optional[str] = None,
    now_epoch: Optional[float] = None,
) -> Tuple[List[str], List[str], str]:
    """Build exact KEYS and ARGV for SUBMIT_REQUEST_LUA matching RedisSortedSetProducer."""
    if deadline_unix <= 0:
        raise ValueError("deadline_unix must be positive")
    now_s = float(now_epoch) if now_epoch is not None else time.time()
    active_ttl_ms = max(int(round((deadline_unix - now_s) * 1000.0)), 1)
    seq_expire_at = int(deadline_unix + _ENQUEUE_SEQ_GRACE_SECONDS)
    token = request_token or secrets.token_hex(16)

    envelope_obj: Dict[str, Any] = {
        "internal": {
            "request_token": token,
            "request_queue_name": request_queue_name,
            "result_queue_name": result_queue_name,
            "enqueue_seq": -1,
        },
        "request_kind": "plain",
        "data": {
            "id": req_id,
            "created": int(created_unix),
            "deadline": int(deadline_unix),
            "payload": dict(payload),
            "metadata": dict(metadata or {}),
            "headers": dict(headers or {}),
            "endpoint": endpoint,
            "model": model_name,
        },
    }
    serialized = json.dumps(envelope_obj, separators=(",", ":"))
    at = serialized.find(_ENQUEUE_SEQ_PLACEHOLDER_JSON)
    if at < 0:
        raise RuntimeError("Marshaled envelope missing enqueue_seq placeholder")

    json_before = serialized[: at + len(_ENQUEUE_SEQ_FIELD_JSON)]
    json_after = serialized[at + len(_ENQUEUE_SEQ_PLACEHOLDER_JSON) :]

    keys = [
        f"request-seq:{request_queue_name}:{int(deadline_unix)}",
        f"request-cancel:{req_id}",
        f"request-active:{req_id}",
        request_queue_name,
    ]
    argv = [
        token,
        str(active_ttl_ms),
        str(seq_expire_at),
        json_before,
        json_after,
        str(int(deadline_unix)),
    ]
    return keys, argv, token


class _BufferedJsonResponse:
    """Minimal aiohttp.ClientResponse-compatible adapter for non-streaming JSON payloads."""

    def __init__(self, status: int, body_text: str) -> None:
        self.status = status
        self._body_text = body_text
        self.headers: CIMultiDictProxy[str] = CIMultiDictProxy(CIMultiDict({"Content-Type": "application/json"}))

    async def text(self) -> str:
        return self._body_text

    async def read(self) -> bytes:
        return self._body_text.encode("utf-8")

    async def json(self) -> Any:
        return json.loads(self._body_text)


def _extract_worker_tag(req_id: str) -> Optional[str]:
    """Extract the session tag from 'ip-<tag>-<seq>-<nonce>'."""
    if not req_id.startswith(_REQ_ID_PREFIX):
        return None
    rest = req_id[len(_REQ_ID_PREFIX) :]
    sep = rest.find("-")
    if sep <= 0:
        return None
    return rest[:sep]


class LLMDAsyncRedisModelServerClient(ModelServerClient):
    """ModelServerClient submitting requests directly to llm-d-async's Redis SortedSet."""

    def __init__(
        self,
        metrics_collector: Any,
        api_config: APIConfig,
        uri: str,
        model_name: Optional[str],
        tokenizer_config: Optional[CustomTokenizerConfig],
        additional_filters: List[str],
        redis_config: Optional[LLMDAsyncRedisConfig] = None,
        redis_password: Optional[str] = None,
        ignore_eos: bool = True,
        timeout: Optional[float] = None,
        request_retries: int = 0,
        request_retry_backoff_sec: float = 0.5,
        lora_config: Optional[List[MultiLoRAConfig]] = None,
    ) -> None:
        if api_config.streaming:
            raise ValueError("streaming is not supported for server.type 'llmd_async_redis'")
        if not model_name:
            raise ValueError("model_name is required for server.type 'llmd_async_redis'")

        super().__init__(
            api_config,
            timeout,
            request_retries=request_retries,
            request_retry_backoff_sec=request_retry_backoff_sec,
        )
        self.metrics_collector = metrics_collector
        self.uri = uri
        self.model_name: str = model_name
        self.max_completion_tokens = 30
        self.ignore_eos = ignore_eos
        self.additional_filters = additional_filters
        self.lora_config = lora_config
        self.redis_config = redis_config or LLMDAsyncRedisConfig()
        self.redis_password = redis_password

        resolved_tok_cfg = tokenizer_config if tokenizer_config is not None else CustomTokenizerConfig()
        if not resolved_tok_cfg.pretrained_model_name_or_path:
            resolved_tok_cfg = resolved_tok_cfg.model_copy(update={"pretrained_model_name_or_path": self.model_name})
        self.tokenizer_config = resolved_tok_cfg
        self.tokenizer = CustomTokenizer(self.tokenizer_config)
        self.metric_filters = [f"model_name='{self.model_name}'", *additional_filters]

        self._session: Optional[LLMDAsyncRedisClientSession] = None
        self._session_lock: Optional[asyncio.Lock] = None

        if self.redis_config.clear_queues_before_run:
            asyncio.run(self._clear_queues_once())

    async def _clear_queues_once(self) -> None:
        admin = AsyncResp2RedisClient(self.uri, password_override=self.redis_password)
        try:
            await admin.delete(self.redis_config.request_queue_name, self.redis_config.result_queue_name)
        finally:
            await admin.close()

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_session"] = None
        state["_session_lock"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._session = None
        self._session_lock = None

    def new_session(self) -> "ModelServerClientSession":
        return LLMDAsyncRedisClientSession(self)

    async def process_request(
        self,
        data: InferenceAPIData,
        stage_id: int,
        scheduled_time: float,
        lora_adapter: Optional[str] = None,
    ) -> None:
        if self._session_lock is None:
            self._session_lock = asyncio.Lock()
        async with self._session_lock:
            if self._session is None:
                self._session = LLMDAsyncRedisClientSession(self)
            session = self._session
        await session.process_request(data, stage_id, scheduled_time, lora_adapter)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat, APIType.AnthropicMessages, APIType.Embeddings, APIType.Rerank]

    def get_prometheus_metric_metadata(self) -> OpenAIMetrics:
        return OpenAIMetrics(
            filters=self.metric_filters,
            prompt_tokens=CounterMetric("vllm:prompt_tokens"),
            output_tokens=CounterMetric("vllm:generation_tokens"),
            requests=CounterMetric("vllm:request_success"),
            request_latency=HistogramMetric("vllm:e2e_request_latency_seconds"),
            queue_length=GaugeMetric("llm_d_async_async_queue_depth"),
            time_per_output_token=HistogramMetric("vllm:request_time_per_output_token_seconds"),
        )


class LLMDAsyncRedisClientSession(ModelServerClientSession):
    """Per-worker async Redis SortedSet producer + BRPOP result demultiplexer."""

    client: LLMDAsyncRedisModelServerClient

    def __init__(self, client: LLMDAsyncRedisModelServerClient) -> None:
        super().__init__(client)
        self.client = client
        self._session_tag = secrets.token_hex(4)
        self._seq = 0
        self._producer = AsyncResp2RedisClient(client.uri, password_override=client.redis_password)
        self._consumer = AsyncResp2RedisClient(client.uri, password_override=client.redis_password)
        self._inflight: Dict[str, asyncio.Future[Dict[str, Any]]] = {}
        self._consumer_task: Optional[asyncio.Task[None]] = None
        self._closed = False
        self._start_lock: Optional[asyncio.Lock] = None

    @property
    def private_result_queue(self) -> str:
        return f"{self.client.redis_config.result_queue_name}:{self._session_tag}"

    async def _ensure_consumer_started(self) -> None:
        if self._consumer_task is not None and not self._consumer_task.done():
            return
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
        async with self._start_lock:
            if self._consumer_task is None or self._consumer_task.done():
                self._consumer_task = asyncio.create_task(self._consumer_loop())

    async def _consumer_loop(self) -> None:
        base_queue = self.client.redis_config.result_queue_name
        queues = [self.private_result_queue, base_queue]
        while not self._closed:
            try:
                popped = await self._consumer.brpop(queues, timeout_seconds=1)
            except asyncio.CancelledError:
                break
            except (OSError, ConnectionError, RedisProtocolError, asyncio.TimeoutError):
                await self._consumer.close()
                if not self._closed:
                    await asyncio.sleep(0.2)
                continue

            if popped is None:
                continue
            _qname, raw_val = popped
            try:
                msg = json.loads(raw_val)
            except ValueError:
                continue
            if not isinstance(msg, dict):
                continue

            rid = str(msg.get("id") or "")
            fut = self._inflight.get(rid)
            if fut is not None:
                if not fut.done():
                    fut.set_result(msg)
                continue

            owner_tag = _extract_worker_tag(rid)
            if owner_tag and owner_tag != self._session_tag:
                try:
                    await self._producer.rpush_with_ttl(
                        f"{base_queue}:{owner_tag}",
                        raw_val,
                        _WORKER_RELAY_TTL_SECONDS,
                    )
                except (OSError, ConnectionError, RedisProtocolError):
                    pass

    async def process_request(
        self,
        data: InferenceAPIData,
        stage_id: int,
        scheduled_time: float,
        lora_adapter: Optional[str] = None,
    ) -> None:
        await self._ensure_consumer_started()

        effective_model_name = lora_adapter if lora_adapter else self.client.model_name
        payload = await data.to_request_body(
            effective_model_name=effective_model_name,
            max_tokens=self.client.max_completion_tokens,
            ignore_eos=self.client.ignore_eos,
            streaming=False,
        )
        if self.client.api_config.clean_tool_schemas:
            clean_request_tool_schemas(payload, self.client.api_config.type)
        if self.client.api_config.type != APIType.AnthropicMessages and self.client.api_config.response_format:
            payload["response_format"] = self.client.api_config.response_format.to_api_format()

        combined_headers: Dict[str, str] = {}
        if self.client.api_config.headers:
            _update_headers_case_insensitive(combined_headers, self.client.api_config.headers)
        if data.headers:
            _update_headers_case_insensitive(combined_headers, data.headers)

        deadline_offset_sec = _resolve_deadline_offset_seconds(
            self.client.api_config,
            combined_headers,
            self.client.redis_config.default_deadline_seconds,
        )

        tenant = self.client.redis_config.tenant
        forwarded_headers: Dict[str, str] = {}
        for k, v in combined_headers.items():
            kl = k.lower()
            if kl == "x-llm-d-tenant":
                tenant = v
            elif kl not in ("x-llm-d-async-mode", "x-llm-d-request-timeout-seconds"):
                forwarded_headers[k] = v

        self._seq += 1
        req_id = f"{_REQ_ID_PREFIX}{self._session_tag}-{self._seq:06d}-{secrets.token_hex(4)}"
        request_data = json.dumps(payload)

        now_epoch = time.time()
        created_unix = int(now_epoch)
        deadline_unix = int(math.ceil(now_epoch + deadline_offset_sec))

        keys, argv, _token = build_sortedset_enqueue_command(
            req_id=req_id,
            created_unix=created_unix,
            deadline_unix=deadline_unix,
            payload=payload,
            request_queue_name=self.client.redis_config.request_queue_name,
            result_queue_name=self.client.redis_config.result_queue_name,
            endpoint=data.get_route(),
            model_name=effective_model_name,
            metadata={"userid": tenant, "tenant": tenant},
            headers=forwarded_headers,
            now_epoch=now_epoch,
        )

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Dict[str, Any]] = loop.create_future()
        self._inflight[req_id] = fut

        start = time.perf_counter()
        info: Optional[InferenceInfo] = None
        error: Optional[ErrorResponseInfo] = None
        response_content = ""
        cancelled = False
        wait_budget = self.client.timeout if self.client.timeout is not None else float(deadline_offset_sec + 30.0)

        try:
            await self._producer.eval_script(SUBMIT_REQUEST_LUA, keys, argv)
            result_msg = await asyncio.wait_for(fut, timeout=wait_budget)

            status_code = int(result_msg.get("status_code") or 0)
            error_code = str(result_msg.get("error_code") or "")
            error_message = str(result_msg.get("error_message") or "")
            raw_payload = result_msg.get("payload")
            if isinstance(raw_payload, str):
                response_content = raw_payload
            elif raw_payload is not None:
                response_content = json.dumps(raw_payload)

            buffered_resp = cast(ClientResponse, _BufferedJsonResponse(status_code, response_content))
            if 200 <= status_code < 300 and not error_code:
                try:
                    info = await data.process_response(
                        response=buffered_resp,
                        config=self.client.api_config,
                        tokenizer=self.client.tokenizer,
                        lora_adapter=lora_adapter,
                    )
                except Exception as read_err:
                    orig: Exception = read_err
                    if isinstance(read_err, InvalidResponseError) and read_err.raw_content:
                        response_content = read_err.raw_content
                    error = ErrorResponseInfo(error_msg=str(orig), error_type=type(orig).__name__)
                    info = await data.process_failure(
                        response=None,
                        config=self.client.api_config,
                        tokenizer=self.client.tokenizer,
                        exception=orig,
                        lora_adapter=lora_adapter,
                    )
            else:
                err_type = error_code if error_code else f"HTTP Error {status_code}"
                err_msg = error_message or response_content or err_type
                error = ErrorResponseInfo(error_msg=err_msg, error_type=err_type)
                exc = Exception(f"{err_type}: {err_msg}")
                info = await data.process_failure(
                    response=buffered_resp,
                    config=self.client.api_config,
                    tokenizer=self.client.tokenizer,
                    exception=exc,
                    lora_adapter=lora_adapter,
                )
        except asyncio.CancelledError as exc:
            cancelled = True
            error = stage_teardown_cancelled_error()
            info = await data.process_failure(
                response=None,
                config=self.client.api_config,
                tokenizer=self.client.tokenizer,
                exception=exc,
                lora_adapter=lora_adapter,
            )
        except asyncio.TimeoutError as exc:
            error = ErrorResponseInfo(error_msg="Request timed out", error_type="TimeoutError")
            info = await data.process_failure(
                response=None,
                config=self.client.api_config,
                tokenizer=self.client.tokenizer,
                exception=exc,
                lora_adapter=lora_adapter,
            )
        except Exception as exc:
            error = ErrorResponseInfo(error_msg=str(exc), error_type=type(exc).__name__)
            info = await data.process_failure(
                response=None,
                config=self.client.api_config,
                tokenizer=self.client.tokenizer,
                exception=exc,
                lora_adapter=lora_adapter,
            )
        finally:
            self._inflight.pop(req_id, None)

        end_time = time.perf_counter()

        if not info:
            info = InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=0)))
        if data.labels:
            info.labels = data.labels
        if data.graph_event_id:
            info.graph_event_id = data.graph_event_id

        if self.client.api_config.type == APIType.Embeddings and error is None and not is_stage_teardown_cancelled(error):
            response_content = _strip_embeddings_vectors(response_content)

        metric = RequestLifecycleMetric(
            stage_id=stage_id,
            session_id=data.session_id if isinstance(data.session_id, str) else None,
            request_session_id=data.request_session_id if isinstance(data.request_session_id, str) else None,
            request_data=request_data,
            response_data=response_content,
            info=info,
            error=error,
            start_time=start,
            end_time=end_time,
            scheduled_time=scheduled_time,
        )
        if "request_latency_slo_sec" in RequestLifecycleMetric.model_fields:
            setattr(metric, "request_latency_slo_sec", float(deadline_offset_sec))
        self.client.metrics_collector.record_metric(metric)

        if cancelled:
            raise asyncio.CancelledError()

    async def close(self) -> None:
        self._closed = True
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except (asyncio.CancelledError, Exception):
                pass
            self._consumer_task = None
        await self._producer.close()
        await self._consumer.close()
