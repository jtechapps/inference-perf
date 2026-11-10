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
"""Tests for asynchronous HTTP 202 Accepted enqueue + GET status polling."""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_perf.apis import CompletionAPIData, RequestLifecycleMetric
from inference_perf.client.modelserver.openai_client import openAIModelServerClientSession
from inference_perf.config import APIConfig, APIType, AsyncPollingConfig
from inference_perf.config.redaction import REDACTED, redact


class _MockHttpContext:
    def __init__(self, status: int, body: str, headers: Dict[str, str] | None = None) -> None:
        self.resp = MagicMock()
        self.resp.status = status
        self.resp.headers = headers or {}
        self.resp.text = AsyncMock(return_value=body)
        self.resp.read = AsyncMock(return_value=body.encode("utf-8"))
        self.resp.json = AsyncMock(side_effect=lambda: json.loads(body))

    async def __aenter__(self) -> MagicMock:
        return self.resp

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


def _make_mock_client(api_config: APIConfig, timeout: float | None = 10.0) -> MagicMock:
    client = MagicMock()
    client.uri = "http://gateway:8081"
    client.model_name = "Qwen/Qwen3-0.6B"
    client.max_completion_tokens = 32
    client.ignore_eos = True
    client.api_key = "secret-key"
    client.api_config = api_config
    client.timeout = timeout
    client.cert_path = None
    client.key_path = None
    client.request_retries = 2
    client.request_retry_backoff_sec = 0.01
    client.otel = MagicMock()
    client.otel.enabled = False
    client.otel.trace_llm_request.return_value.__enter__.return_value = None
    client.otel.trace_llm_request.return_value.__exit__.return_value = None
    client.tokenizer = MagicMock()
    client.tokenizer.count_tokens.return_value = 8
    client.metrics_collector = MagicMock()
    return client


def test_async_polling_config_validation_and_redaction() -> None:
    with pytest.raises(ValueError, match="must contain '\\{id\\}'"):
        AsyncPollingConfig(poll_url_template="/v1/requests/static")

    with pytest.raises(ValueError, match="poll_url_template must start with"):
        AsyncPollingConfig(poll_url_template="v1/requests/{id}")

    with pytest.raises(ValueError, match="streaming is not supported when async_polling is configured"):
        APIConfig(
            type=APIType.Completion,
            streaming=True,
            async_polling=AsyncPollingConfig(),
        )

    raw = {
        "type": "completion",
        "headers": {"x-llm-d-tenant": "batch", "Authorization": "Bearer top-secret"},
        "async_polling": {
            "poll_url_template": "/v1/requests/{id}",
            "headers": {"X-Api-Key": "poll-secret", "x-llm-d-tenant": "batch"},
        },
    }
    masked = redact(raw, APIConfig)
    assert masked["headers"]["Authorization"] == REDACTED
    assert masked["headers"]["x-llm-d-tenant"] == "batch"
    assert masked["async_polling"]["headers"]["X-Api-Key"] == REDACTED
    assert masked["async_polling"]["headers"]["x-llm-d-tenant"] == "batch"


@pytest.mark.asyncio
async def test_enqueue_202_polls_until_200_and_preserves_tenant_header() -> None:
    api_cfg = APIConfig(
        type=APIType.Completion,
        streaming=False,
        headers={
            "x-llm-d-async-mode": "enqueue",
            "x-llm-d-tenant": "premium",
            "x-llm-d-request-timeout-seconds": "45",
        },
        async_polling=AsyncPollingConfig(
            poll_url_template="/v1/requests/{id}",
            poll_interval_sec=0.005,
        ),
    )
    client = _make_mock_client(api_cfg)
    session = openAIModelServerClientSession(client)
    session.session = MagicMock()

    session.session.post.return_value = _MockHttpContext(
        202,
        json.dumps({"id": "job-4217", "status": "pending"}),
    )

    completion_payload = json.dumps(
        {
            "id": "cmpl-4217",
            "choices": [{"text": "hello async world"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 6},
        }
    )
    get_calls: List[tuple[str, Dict[str, str]]] = []

    def _fake_get(url: str, headers: Dict[str, str], **kwargs: Any) -> _MockHttpContext:
        get_calls.append((url, dict(headers)))
        if len(get_calls) == 1:
            return _MockHttpContext(202, json.dumps({"id": "job-4217", "status": "pending"}))
        return _MockHttpContext(200, completion_payload)

    session.session.get.side_effect = _fake_get

    req = CompletionAPIData(prompt="Summarize async batching.", max_tokens=16)
    await session.process_request(req, stage_id=0, scheduled_time=0.0)

    assert len(get_calls) == 2
    poll_url, poll_headers = get_calls[0]
    assert poll_url == "http://gateway:8081/v1/requests/job-4217"
    assert poll_headers.get("x-llm-d-tenant") == "premium"
    assert poll_headers.get("Authorization") == "Bearer secret-key"
    assert "x-llm-d-async-mode" not in {k.lower() for k in poll_headers}
    assert "content-type" not in {k.lower() for k in poll_headers}

    client.metrics_collector.record_metric.assert_called_once()
    recorded: RequestLifecycleMetric = client.metrics_collector.record_metric.call_args.args[0]
    assert recorded.error is None
    assert recorded.info.request_metrics.text.input_tokens == 12
    assert recorded.info.response_metrics is not None


@pytest.mark.asyncio
async def test_enqueue_202_terminal_504_deadline_exceeded_is_not_retried() -> None:
    api_cfg = APIConfig(
        type=APIConfig.model_fields["type"].default,
        streaming=False,
        headers={
            "x-llm-d-async-mode": "enqueue",
            "x-llm-d-tenant": "batch",
            "x-llm-d-request-timeout-seconds": "10",
        },
        async_polling=AsyncPollingConfig(poll_interval_sec=0.005),
    )
    client = _make_mock_client(api_cfg)
    session = openAIModelServerClientSession(client)
    session.session = MagicMock()

    # Body has no id, so response_id_header ('x-request-id') fallback is exercised too.
    session.session.post.return_value = _MockHttpContext(
        202,
        "{}",
        headers={"X-Request-Id": "job-expired-99"},
    )
    session.session.get.return_value = _MockHttpContext(
        504,
        json.dumps({"error": {"code": "DEADLINE_EXCEEDED", "message": "deadline exceeded"}}),
    )

    req = CompletionAPIData(prompt="Late job", max_tokens=16)
    await session.process_request(req, stage_id=0, scheduled_time=0.0)

    # POST was accepted (202), so transport retry loop must not re-POST a duplicate job.
    assert session.session.post.call_count == 1
    recorded: RequestLifecycleMetric = client.metrics_collector.record_metric.call_args.args[0]
    assert recorded.error is not None
    assert recorded.error.error_type == "HTTP Error 504"


@pytest.mark.asyncio
async def test_enqueue_202_poll_timeout_records_timeout_error() -> None:
    api_cfg = APIConfig(
        type=APIType.Completion,
        streaming=False,
        async_polling=AsyncPollingConfig(
            poll_interval_sec=0.01,
            max_poll_duration_sec=0.025,
        ),
    )
    client = _make_mock_client(api_cfg, timeout=5.0)
    session = openAIModelServerClientSession(client)
    session.session = MagicMock()

    session.session.post.return_value = _MockHttpContext(202, json.dumps({"id": "job-stuck"}))
    session.session.get.return_value = _MockHttpContext(202, json.dumps({"id": "job-stuck", "status": "pending"}))

    req = CompletionAPIData(prompt="Stuck job", max_tokens=16)
    await session.process_request(req, stage_id=0, scheduled_time=0.0)

    assert session.session.post.call_count == 1
    recorded: RequestLifecycleMetric = client.metrics_collector.record_metric.call_args.args[0]
    assert recorded.error is not None
    assert recorded.error.error_type == "TimeoutError"
