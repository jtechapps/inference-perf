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
from enum import Enum
from typing import Optional

from inference_perf.config.common import StrictBaseModel
from pydantic import Field, SecretStr, model_validator


class ModelServerType(Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
    TGI = "tgi"
    MOCK = "mock"
    LLMD_ASYNC_REDIS = "llmd_async_redis"


class LLMDAsyncRedisConfig(StrictBaseModel):
    """Queue and deadline options for the llm-d-async Redis SortedSet client (type 'llmd_async_redis')."""

    request_queue_name: str = Field(
        default="request-sortedset",
        min_length=1,
        description="Redis SortedSet key where llm-d-async request envelopes are enqueued.",
    )
    result_queue_name: str = Field(
        default="result-list",
        min_length=1,
        description="Redis list key from which llm-d-async ResultMessage envelopes are consumed via BRPOP.",
    )
    default_deadline_seconds: int = Field(
        default=300,
        gt=0,
        description="Default per-request deadline offset in seconds when no per-request deadline header is present.",
    )
    tenant: str = Field(
        default="default",
        min_length=1,
        description="Default tenant identifier stamped into envelope metadata when x-llm-d-tenant is not set.",
    )
    password: Optional[SecretStr] = Field(
        default=None,
        description="Optional Redis AUTH password overriding any password in base_url. Redacted in the logged and saved config.",
    )
    clear_queues_before_run: bool = Field(
        default=False,
        description="Delete request_queue_name and result_queue_name once before starting load.",
    )


class ModelServerClientConfig(StrictBaseModel):
    type: ModelServerType = Field(default=ModelServerType.VLLM, description="Type of model server being benchmarked.")
    model_name: Optional[str] = Field(
        default=None, description="Model name sent in each request. Auto-detected from the server if unset."
    )
    base_url: str = Field(description="Base URL of the model server, e.g. 'http://localhost:8000'.")
    ignore_eos: bool = Field(
        default=True,
        description="Ask the server to keep generating past the end-of-sequence token so outputs hit the requested length.",
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key sent as a bearer token with each request. Redacted in the logged and saved config.",
    )
    cert_path: Optional[str] = Field(default=None, description="Path to a client TLS certificate file.")
    key_path: Optional[str] = Field(default=None, description="Path to the private key for the client TLS certificate.")
    llmd_async_redis: Optional[LLMDAsyncRedisConfig] = Field(
        default=None,
        description="Queue and deadline options for the llm-d-async Redis SortedSet client. Only valid when type is 'llmd_async_redis'.",
    )

    @model_validator(mode="after")
    def validate_server_options(self) -> "ModelServerClientConfig":
        if self.type != ModelServerType.LLMD_ASYNC_REDIS and self.llmd_async_redis is not None:
            raise ValueError("llmd_async_redis options are only valid when type is 'llmd_async_redis'")
        if self.type == ModelServerType.LLMD_ASYNC_REDIS and not self.model_name:
            raise ValueError("model_name is required when type is 'llmd_async_redis'")
        return self
