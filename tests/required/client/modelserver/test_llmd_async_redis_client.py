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
"""Unit and local TCP RESP2 round-trip tests for LLMDAsyncRedisModelServerClient."""

from __future__ import annotations

import asyncio
import json
import pickle
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from backend_client_suite import StubTokenizer
from inference_perf.apis import ChatCompletionAPIData, ChatMessage, RequestLifecycleMetric
from inference_perf.client.modelserver.llmd_async_redis_client import (
    LLMDAsyncRedisClientSession,
    LLMDAsyncRedisModelServerClient,
    build_sortedset_enqueue_command,
    compute_queue_score,
)
from inference_perf.config import (
    APIConfig,
    APIType,
    CustomTokenizerConfig,
    LLMDAsyncRedisConfig,
    ModelServerClientConfig,
    ModelServerType,
)
from inference_perf.config.redaction import REDACTED, redact


class _InMemoryResp2RedisServer:
    """Lightweight local TCP RESP2 server supporting PING, AUTH, SELECT, DEL, RPUSH, EXPIRE, EVAL, and BRPOP."""

    def __init__(
        self,
        on_enqueue: Optional[Callable[["_InMemoryResp2RedisServer", float, Dict[str, Any]], None]] = None,
    ) -> None:
        self.lists: Dict[str, List[str]] = {}
        self.zsets: Dict[str, List[tuple[float, str]]] = {}
        self.kv: Dict[str, str] = {}
        self.counters: Dict[str, int] = {}
        self.auth_tokens: List[str] = []
        self._notify = asyncio.Condition()
        self._server: Optional[asyncio.AbstractServer] = None
        self.port = 0
        self.on_enqueue = on_enqueue

    async def start(self) -> str:
        self._server = await asyncio.start_server(self._handle_client, "127.0.0.1", 0)
        sockets = self._server.sockets
        assert sockets
        self.port = int(sockets[0].getsockname()[1])
        return f"redis://127.0.0.1:{self.port}/0"

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def push_result(self, queue_name: str, result_obj: Dict[str, Any]) -> None:
        async with self._notify:
            self.lists.setdefault(queue_name, []).insert(0, json.dumps(result_obj))
            self._notify.notify_all()

    @staticmethod
    async def _read_command(reader: asyncio.StreamReader) -> Optional[List[str]]:
        header = await reader.readline()
        if not header:
            return None
        assert header.startswith(b"*")
        argc = int(header[1:-2])
        args: List[str] = []
        for _ in range(argc):
            blen_line = await reader.readline()
            assert blen_line.startswith(b"$")
            blen = int(blen_line[1:-2])
            data = await reader.readexactly(blen + 2)
            args.append(data[:-2].decode("utf-8"))
        return args

    @staticmethod
    def _encode_bulk(val: Optional[str]) -> bytes:
        if val is None:
            return b"$-1\r\n"
        raw = val.encode("utf-8")
        return f"${len(raw)}\r\n".encode("ascii") + raw + b"\r\n"

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                cmd = await self._read_command(reader)
                if cmd is None:
                    break
                op = cmd[0].upper()
                if op == "PING":
                    writer.write(b"+PONG\r\n")
                elif op == "AUTH":
                    self.auth_tokens.append(":".join(cmd[1:]))
                    writer.write(b"+OK\r\n")
                elif op == "SELECT":
                    writer.write(b"+OK\r\n")
                elif op == "DEL":
                    removed = 0
                    for k in cmd[1:]:
                        if k in self.lists:
                            del self.lists[k]
                            removed += 1
                        if k in self.zsets:
                            del self.zsets[k]
                            removed += 1
                        if k in self.kv:
                            del self.kv[k]
                            removed += 1
                    writer.write(f":{removed}\r\n".encode("ascii"))
                elif op == "EXPIRE":
                    writer.write(b":1\r\n")
                elif op == "RPUSH":
                    key, val = cmd[1], cmd[2]
                    async with self._notify:
                        self.lists.setdefault(key, []).insert(0, val)
                        self._notify.notify_all()
                    writer.write(f":{len(self.lists[key])}\r\n".encode("ascii"))
                elif op == "EVAL":
                    numkeys = int(cmd[2])
                    keys = cmd[3 : 3 + numkeys]
                    argv = cmd[3 + numkeys :]
                    seq_key, cancel_key, active_key, zset_key = keys[0], keys[1], keys[2], keys[3]
                    seq = self.counters.get(seq_key, 0) + 1
                    self.counters[seq_key] = seq
                    self.kv.pop(cancel_key, None)
                    self.kv[active_key] = argv[0]
                    score = compute_queue_score(int(argv[5]), seq)
                    member = f"{argv[3]}{seq}{argv[4]}"
                    self.zsets.setdefault(zset_key, []).append((score, member))
                    if self.on_enqueue is not None:
                        self.on_enqueue(self, score, json.loads(member))
                    writer.write(f":{seq}\r\n".encode("ascii"))
                elif op == "BRPOP":
                    watch_keys = cmd[1:-1]
                    timeout_s = float(cmd[-1])
                    popped: Optional[tuple[str, str]] = None
                    async with self._notify:
                        deadline = asyncio.get_running_loop().time() + timeout_s
                        while True:
                            for wk in watch_keys:
                                bucket = self.lists.get(wk)
                                if bucket:
                                    popped = (wk, bucket.pop())
                                    break
                            if popped is not None:
                                break
                            rem = deadline - asyncio.get_running_loop().time()
                            if rem <= 0:
                                break
                            try:
                                await asyncio.wait_for(self._notify.wait(), timeout=rem)
                            except asyncio.TimeoutError:
                                break
                    if popped is None:
                        writer.write(b"*-1\r\n")
                    else:
                        qname, item = popped
                        writer.write(b"*2\r\n" + self._encode_bulk(qname) + self._encode_bulk(item))
                else:
                    writer.write(f"-ERR unsupported command {op}\r\n".encode("ascii"))
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except OSError:
                pass


def test_modelserver_config_validation_and_secret_redaction() -> None:
    with pytest.raises(ValueError, match="model_name is required when type is 'llmd_async_redis'"):
        ModelServerClientConfig(type=ModelServerType.LLMD_ASYNC_REDIS, base_url="redis://localhost:6379/0")

    with pytest.raises(ValueError, match="llmd_async_redis options are only valid when type is 'llmd_async_redis'"):
        ModelServerClientConfig(
            type=ModelServerType.VLLM,
            base_url="http://localhost:8000",
            model_name="Qwen/Qwen3-8B",
            llmd_async_redis=LLMDAsyncRedisConfig(),
        )

    raw = {
        "type": "llmd_async_redis",
        "base_url": "redis://localhost:6379/0",
        "model_name": "Qwen/Qwen3-8B",
        "llmd_async_redis": {
            "request_queue_name": "batch-q",
            "password": "super-secret-redis-pass",
        },
    }
    masked = redact(raw, ModelServerClientConfig)
    assert masked["llmd_async_redis"]["password"] == REDACTED
    assert masked["llmd_async_redis"]["request_queue_name"] == "batch-q"


def test_build_sortedset_enqueue_command_matches_llmd_async_wire_contract() -> None:
    keys, argv, token = build_sortedset_enqueue_command(
        req_id="req-001",
        created_unix=1700000000,
        deadline_unix=1700000300,
        payload={"model": "Qwen/Qwen3-8B", "prompt": "hi", "max_tokens": 16},
        request_queue_name="request-sortedset",
        result_queue_name="result-list",
        endpoint="/v1/completions",
        model_name="Qwen/Qwen3-8B",
        metadata={"userid": "tenant-a", "tenant": "tenant-a"},
        now_epoch=1700000000.0,
    )
    assert keys == [
        "request-seq:request-sortedset:1700000300",
        "request-cancel:req-001",
        "request-active:req-001",
        "request-sortedset",
    ]
    assert argv[0] == token
    assert argv[1] == "300000"
    assert argv[5] == "1700000300"

    reconstructed = json.loads(f"{argv[3]}7{argv[4]}")
    assert reconstructed["internal"]["enqueue_seq"] == 7
    assert reconstructed["internal"]["request_queue_name"] == "request-sortedset"
    assert reconstructed["internal"]["result_queue_name"] == "result-list"
    assert reconstructed["data"]["id"] == "req-001"
    assert reconstructed["data"]["deadline"] == 1700000300
    assert compute_queue_score(1700000300, 1) > 1700000300.0
    assert compute_queue_score(1700000300, 2) > compute_queue_score(1700000300, 1)


def test_llmd_async_redis_client_is_picklable_for_forkserver_workers() -> None:
    with patch("inference_perf.client.modelserver.llmd_async_redis_client.CustomTokenizer", StubTokenizer):
        client = LLMDAsyncRedisModelServerClient(
            metrics_collector=None,
            api_config=APIConfig(type=APIType.Completion, streaming=False),
            uri="redis://127.0.0.1:6379/0",
            model_name="Qwen/Qwen3-8B",
            tokenizer_config=CustomTokenizerConfig(pretrained_model_name_or_path="Qwen/Qwen3-8B"),
            additional_filters=[],
            redis_config=LLMDAsyncRedisConfig(default_deadline_seconds=120),
        )
        restored: LLMDAsyncRedisModelServerClient = pickle.loads(pickle.dumps(client))
        assert restored.model_name == "Qwen/Qwen3-8B"
        assert restored.redis_config.default_deadline_seconds == 120


@pytest.mark.asyncio
async def test_llmd_async_redis_e2e_roundtrip_success_failure_and_multiworker_relay() -> None:
    loop = asyncio.get_running_loop()

    def _auto_respond(srv: _InMemoryResp2RedisServer, _score: float, envelope: Dict[str, Any]) -> None:
        data = envelope["data"]
        rid = data["id"]
        res_q = envelope["internal"]["result_queue_name"]
        payload = data["payload"]
        prompt_str = json.dumps(payload)
        if "EXPIRE_ME" in prompt_str:
            reply = {
                "id": rid,
                "status_code": 504,
                "error_code": "DEADLINE_EXCEEDED",
                "error_message": "Request exceeded deadline in queue",
            }
        else:
            reply = {
                "id": rid,
                "status_code": 200,
                "payload": json.dumps(
                    {
                        "id": f"cmpl-{rid}",
                        "choices": [{"message": {"role": "assistant", "content": "ok response"}}],
                        "usage": {"prompt_tokens": 14, "completion_tokens": 9},
                    }
                ),
            }
        loop.create_task(srv.push_result(res_q, reply))

    redis_srv = _InMemoryResp2RedisServer(on_enqueue=_auto_respond)
    redis_url = await redis_srv.start()
    try:
        collector = MagicMock()
        api_cfg = APIConfig(
            type=APIType.Chat,
            streaming=False,
            headers={"x-llm-d-tenant": "team-batch", "x-llm-d-request-timeout-seconds": "90"},
        )
        with patch("inference_perf.client.modelserver.llmd_async_redis_client.CustomTokenizer", StubTokenizer):
            client = LLMDAsyncRedisModelServerClient(
                metrics_collector=collector,
                api_config=api_cfg,
                uri=redis_url,
                model_name="Qwen/Qwen3-8B",
                tokenizer_config=CustomTokenizerConfig(pretrained_model_name_or_path="Qwen/Qwen3-8B"),
                additional_filters=[],
                redis_config=LLMDAsyncRedisConfig(
                    request_queue_name="request-sortedset",
                    result_queue_name="result-list",
                    default_deadline_seconds=300,
                ),
                redis_password="test-redis-password",
            )

        # Two independent sessions simulate two forkserver worker processes sharing 'result-list'
        session_a = LLMDAsyncRedisClientSession(client)
        session_b = LLMDAsyncRedisClientSession(client)
        try:
            req_ok_a = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="Hello from worker A")])
            req_ok_b = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="Hello from worker B")])
            req_fail = ChatCompletionAPIData(
                messages=[ChatMessage(role="user", content="EXPIRE_ME")],
                headers={"x-llm-d-request-timeout-seconds": "15"},
            )

            await asyncio.gather(
                session_a.process_request(req_ok_a, stage_id=0, scheduled_time=0.0),
                session_b.process_request(req_ok_b, stage_id=0, scheduled_time=0.0),
                session_a.process_request(req_fail, stage_id=0, scheduled_time=0.0),
            )
        finally:
            await session_a.close()
            await session_b.close()

        assert "test-redis-password" in redis_srv.auth_tokens
        assert collector.record_metric.call_count == 3

        # Verify deadline offsets stamped into SortedSet envelopes (90s default header vs 15s per-request override)
        enqueued_envelopes = [json.loads(member) for _score, member in redis_srv.zsets["request-sortedset"]]
        deadline_deltas = sorted(env["data"]["deadline"] - env["data"]["created"] for env in enqueued_envelopes)
        assert deadline_deltas[0] in (15, 16)
        assert all(d in (90, 91) for d in deadline_deltas[1:])

        metrics: List[RequestLifecycleMetric] = [c.args[0] for c in collector.record_metric.call_args_list]
        successes = [m for m in metrics if m.error is None]
        failures = [m for m in metrics if m.error is not None]
        assert len(successes) == 2
        assert len(failures) == 1

        for s in successes:
            assert s.info.request_metrics.text.input_tokens == 14
            assert s.info.response_metrics is not None

        assert failures[0].error is not None
        assert failures[0].error.error_type == "DEADLINE_EXCEEDED"
    finally:
        await redis_srv.close()
