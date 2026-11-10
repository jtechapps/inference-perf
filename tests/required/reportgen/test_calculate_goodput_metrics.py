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
from unittest.mock import Mock
from inference_perf.apis import RequestLifecycleMetric
from inference_perf.config import GoodputConfig
from inference_perf.reportgen.base import calculate_goodput_metrics


def test_calculate_goodput_metrics_empty_metrics() -> None:
    """Test with empty metrics list returns None."""
    config = GoodputConfig(constraints={"ttft": 0.5})
    result = calculate_goodput_metrics([], config, [], [], [], [], [])
    assert result is None


def test_calculate_goodput_metrics_no_config() -> None:
    """Test with no goodput config returns None."""
    metric = Mock(spec=RequestLifecycleMetric)
    metric.ttft_slo_sec = None
    metric.tpot_slo_sec = None
    result = calculate_goodput_metrics([metric], None, [0.3], [None], [0.1], [1.0], [None])
    assert result is None


def test_calculate_goodput_metrics_no_constraints() -> None:
    """Test with empty constraints returns None."""
    config = GoodputConfig(constraints={})
    metric = Mock(spec=RequestLifecycleMetric)
    metric.ttft_slo_sec = None
    metric.tpot_slo_sec = None
    result = calculate_goodput_metrics([metric], config, [0.3], [None], [0.1], [1.0], [None])
    assert result is None


def test_calculate_goodput_metrics_ttft_constraint() -> None:
    """Test goodput calculation with TTFT constraint."""
    config = GoodputConfig(constraints={"ttft": 0.5})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.request_metrics.text.input_tokens = 0
    metric1.info.response_metrics.output_tokens = 10
    metric1.ttft_slo_sec = None
    metric1.tpot_slo_sec = None

    metric2 = Mock(spec=RequestLifecycleMetric)
    metric2.start_time = 2.0
    metric2.end_time = 4.0
    metric2.info = Mock()
    metric2.info.request_metrics.text.input_tokens = 0
    metric2.info.response_metrics.output_tokens = 20
    metric2.ttft_slo_sec = None
    metric2.tpot_slo_sec = None

    metrics = [metric1, metric2]
    ttft_values: list[float | None] = [0.3, 0.6]  # metric1 meets, metric2 fails

    result = calculate_goodput_metrics(metrics, config, ttft_values, [None, None], [0.1, 0.1], [2.0, 2.0], [None, None])  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_percentage"] == 50.0
    assert result["good_requests"] == 1
    assert result["total_requests"] == 2
    # Total benchmark time = 4.0 - 0.0 = 4.0
    assert result["request_goodput"] == 1 / 4.0
    # Good tokens = 10 (from metric1)
    assert result["token_goodput"] == 10 / 4.0


def test_calculate_goodput_metrics_total_tokens() -> None:
    """Test that token goodput uses total tokens (input + output)."""
    config = GoodputConfig(constraints={"ttft": 0.5})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.request_metrics.text.input_tokens = 5
    metric1.info.response_metrics.output_tokens = 10
    metric1.ttft_slo_sec = None
    metric1.tpot_slo_sec = None

    metrics = [metric1]
    ttft_values: list[float | None] = [0.3]  # Meets constraint

    result = calculate_goodput_metrics(metrics, config, ttft_values, [None], [0.1], [2.0], [None])  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_percentage"] == 100.0
    assert result["good_requests"] == 1
    # Total tokens = 5 + 10 = 15
    # Total benchmark time = 2.0 - 0.0 = 2.0
    assert result["token_goodput"] == 15 / 2.0


def test_calculate_goodput_metrics_multiple_constraints() -> None:
    """Test goodput calculation with multiple constraints."""
    config = GoodputConfig(constraints={"ttft": 0.5, "tpot": 0.1})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.request_metrics.text.input_tokens = 0
    metric1.info.response_metrics.output_tokens = 10
    metric1.ttft_slo_sec = None
    metric1.tpot_slo_sec = None

    metric2 = Mock(spec=RequestLifecycleMetric)
    metric2.start_time = 2.0
    metric2.end_time = 4.0
    metric2.info = Mock()
    metric2.info.request_metrics.text.input_tokens = 0
    metric2.info.response_metrics.output_tokens = 20
    metric2.ttft_slo_sec = None
    metric2.tpot_slo_sec = None

    metrics = [metric1, metric2]
    ttft_values: list[float | None] = [0.3, 0.3]  # Both meet TTFT
    tpot_values: list[float | None] = [0.08, 0.12]  # metric1 meets, metric2 fails TPOT

    result = calculate_goodput_metrics(metrics, config, ttft_values, tpot_values, [0.1, 0.1], [2.0, 2.0], [None, None])  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_percentage"] == 50.0
    assert result["good_requests"] == 1


def test_calculate_goodput_metrics_itl_constraint() -> None:
    """Test goodput calculation with ITL constraint."""
    config = GoodputConfig(constraints={"itl": 0.05})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.request_metrics.text.input_tokens = 0
    metric1.info.response_metrics.output_tokens = 10
    metric1.ttft_slo_sec = None
    metric1.tpot_slo_sec = None

    metrics = [metric1]
    itl_values: list[float | None] = [0.04]  # Meets ITL

    result = calculate_goodput_metrics(metrics, config, [None], [None], [0.1], [2.0], itl_values)  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_percentage"] == 100.0


def test_calculate_goodput_metrics_ntpot_constraint() -> None:
    """Test goodput calculation with NTPOT constraint."""
    config = GoodputConfig(constraints={"ntpot": 0.1})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.request_metrics.text.input_tokens = 0
    metric1.info.response_metrics.output_tokens = 10
    metric1.ttft_slo_sec = None
    metric1.tpot_slo_sec = None

    metrics = [metric1]
    ntpot_values: list[float | None] = [0.05]  # Meets NTPOT

    result = calculate_goodput_metrics(metrics, config, [None], [None], ntpot_values, [2.0], [None])  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_percentage"] == 100.0


def test_calculate_goodput_metrics_request_latency_constraint() -> None:
    """Test goodput calculation with request_latency constraint."""
    config = GoodputConfig(constraints={"request_latency": 1.0})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 0.5
    metric1.info = Mock()
    metric1.info.request_metrics.text.input_tokens = 0
    metric1.info.response_metrics.output_tokens = 10
    metric1.ttft_slo_sec = None
    metric1.tpot_slo_sec = None

    metrics = [metric1]
    request_latency_values: list[float | None] = [0.5]  # Meets latency

    result = calculate_goodput_metrics(metrics, config, [None], [None], [0.1], request_latency_values, [None])  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_percentage"] == 100.0


def test_calculate_goodput_metrics_per_request_override() -> None:
    """Test that per-request SLOs override global constraints."""
    config = GoodputConfig(constraints={"ttft": 0.5})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.request_metrics.text.input_tokens = 0
    metric1.info.response_metrics.output_tokens = 10
    metric1.ttft_slo_sec = 0.2  # Stricter than global 0.5
    metric1.tpot_slo_sec = None

    metrics = [metric1]
    ttft_values: list[float | None] = [0.3]  # Fails stricter per-request SLO, would meet global

    result = calculate_goodput_metrics(metrics, config, ttft_values, [None], [0.1], [2.0], [None])  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_percentage"] == 0.0
    assert result["good_requests"] == 0

    # Now test that it meets if it is within per-request SLO
    ttft_values = [0.1]
    result = calculate_goodput_metrics(metrics, config, ttft_values, [None], [0.1], [2.0], [None])  # type: ignore[arg-type]
    assert result is not None
    assert result["goodput_percentage"] == 100.0
    assert result["good_requests"] == 1


def test_calculate_goodput_metrics_individual_attainment() -> None:
    """Test that individual attainment percentages are calculated."""
    config = GoodputConfig(constraints={"ttft": 0.5, "tpot": 0.1})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.request_metrics.text.input_tokens = 0
    metric1.info.response_metrics.output_tokens = 10
    metric1.ttft_slo_sec = None
    metric1.tpot_slo_sec = None

    metric2 = Mock(spec=RequestLifecycleMetric)
    metric2.start_time = 2.0
    metric2.end_time = 4.0
    metric2.info = Mock()
    metric2.info.request_metrics.text.input_tokens = 0
    metric2.info.response_metrics.output_tokens = 20
    metric2.ttft_slo_sec = None
    metric2.tpot_slo_sec = None

    metrics = [metric1, metric2]
    ttft_values: list[float | None] = [0.3, 0.6]  # metric1 meets, metric2 fails TTFT
    tpot_values: list[float | None] = [0.08, 0.05]  # Both meet TPOT

    result = calculate_goodput_metrics(metrics, config, ttft_values, tpot_values, [0.1, 0.1], [2.0, 2.0], [None, None])  # type: ignore[arg-type]

    assert result is not None
    assert result["ttft_attainment_percentage"] == 50.0
    assert result["tpot_attainment_percentage"] == 100.0


def test_calculate_goodput_metrics_ttft_none_fails() -> None:
    """Test that if TTFT is None, it fails the constraint."""
    config = GoodputConfig(constraints={"ttft": 0.5})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.request_metrics.text.input_tokens = 0
    metric1.info.response_metrics.output_tokens = 10
    metric1.ttft_slo_sec = None
    metric1.tpot_slo_sec = None

    metrics = [metric1]
    ttft_values: list[float | None] = [None]  # Value is None

    result = calculate_goodput_metrics(metrics, config, ttft_values, [None], [0.1], [2.0], [None])  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_percentage"] == 0.0
    assert result["good_requests"] == 0


def test_calculate_goodput_metrics_per_request_deadline_and_output_tokens() -> None:
    """Per-request request_latency_slo_sec activates goodput and reports output_token_goodput."""
    m_on_time = Mock(spec=RequestLifecycleMetric)
    m_on_time.start_time = 0.0
    m_on_time.end_time = 4.0
    m_on_time.info = Mock()
    m_on_time.info.request_metrics.text.input_tokens = 20
    m_on_time.info.response_metrics.output_tokens = 40
    m_on_time.ttft_slo_sec = None
    m_on_time.tpot_slo_sec = None
    m_on_time.request_latency_slo_sec = 5.0

    m_late = Mock(spec=RequestLifecycleMetric)
    m_late.start_time = 1.0
    m_late.end_time = 8.0
    m_late.info = Mock()
    m_late.info.request_metrics.text.input_tokens = 20
    m_late.info.response_metrics.output_tokens = 50
    m_late.ttft_slo_sec = None
    m_late.tpot_slo_sec = None
    m_late.request_latency_slo_sec = 5.0

    m_failed_504 = Mock(spec=RequestLifecycleMetric)
    m_failed_504.start_time = 2.0
    m_failed_504.end_time = 10.0
    m_failed_504.ttft_slo_sec = None
    m_failed_504.tpot_slo_sec = None
    m_failed_504.request_latency_slo_sec = 5.0

    result = calculate_goodput_metrics(
        [m_on_time, m_late],
        None,
        [None, None],
        [None, None],
        [0.1, 0.1],
        [4.0, 7.0],
        [None, None],
        failed_metrics=[m_failed_504],
    )

    assert result is not None
    assert result["good_requests"] == 1
    assert result["total_requests"] == 2
    assert result["submitted_requests"] == 3
    assert result["goodput_percentage"] == 50.0
    assert abs(result["submitted_goodput_percentage"] - (100.0 / 3.0)) < 1e-6
    assert result["good_output_tokens"] == 40
    # Window spans [0.0, 10.0] = 10.0s across successes + failures
    assert result["request_goodput"] == 1 / 10.0
    assert result["output_token_goodput"] == 40 / 10.0
    assert result["token_goodput"] == 60 / 10.0
    assert result["request_latency_attainment_percentage"] == 50.0


def test_calculate_goodput_metrics_all_failed_requests() -> None:
    """When every submitted request fails (e.g. 504 DEADLINE_EXCEEDED), attainment is 0%."""
    config = GoodputConfig(constraints={"request_latency": 5.0})
    m_fail = Mock(spec=RequestLifecycleMetric)
    m_fail.start_time = 0.0
    m_fail.end_time = 5.0
    m_fail.ttft_slo_sec = None
    m_fail.tpot_slo_sec = None
    m_fail.request_latency_slo_sec = None

    result = calculate_goodput_metrics(
        [],
        config,
        [],
        [],
        [],
        [],
        [],
        failed_metrics=[m_fail],
    )
    assert result is not None
    assert result["good_requests"] == 0
    assert result["total_requests"] == 0
    assert result["submitted_requests"] == 1
    assert result["goodput_percentage"] == 0.0
    assert result["submitted_goodput_percentage"] == 0.0
    assert result["request_goodput"] == 0.0
    assert result["output_token_goodput"] == 0.0


def test_extract_request_slo_thresholds_seconds_and_ms() -> None:
    """Verify per-request deadline SLO header parsing in seconds and milliseconds."""
    from inference_perf.client.modelserver.openai_client import extract_request_slo_thresholds
    from inference_perf.config import APIConfig, APIType

    cfg_sec = APIConfig(
        type=APIType.Completion,
        slo_request_latency_header="x-llm-d-request-timeout-seconds",
        headers={"x-llm-d-request-timeout-seconds": "180"},
    )
    ttft, tpot, req_lat = extract_request_slo_thresholds(cfg_sec)
    assert ttft is None
    assert tpot is None
    assert req_lat == 180.0

    cfg_ms = APIConfig(
        type=APIType.Completion,
        slo_unit="ms",
        headers={"x-slo-request-latency-ms": "2500"},
    )
    _, _, req_lat_override = extract_request_slo_thresholds(cfg_ms, {"x-slo-request-latency-ms": "1200"})
    assert req_lat_override == 1.2
