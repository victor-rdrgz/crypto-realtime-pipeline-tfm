import io
import json
import logging

import pytest

from crypto_realtime_pipeline.common.cli import landing_parser, run_context
from crypto_realtime_pipeline.common.log import (
    RunContext,
    configure_logging,
    get_logger,
    log_metrics,
)

CONTEXT = RunContext(bundle="crypto", target="dev", workload="land", run_id="42")


@pytest.fixture
def stream():
    buffer = io.StringIO()
    configure_logging(stream=buffer)
    yield buffer
    logging.getLogger().handlers.clear()


def test_logs_json_with_run_context_and_fields(stream):
    logger = get_logger("test", CONTEXT, dataset="binance_spot")

    log_metrics(logger, "landing run completed", {"records": 10})

    line = json.loads(stream.getvalue())
    assert line["message"] == "landing run completed"
    assert line["level"] == "INFO"
    assert line["bundle"] == "crypto"
    assert line["target"] == "dev"
    assert line["workload"] == "land"
    assert line["run_id"] == "42"
    assert line["dataset"] == "binance_spot"
    assert line["records"] == 10


def test_logs_exceptions(stream):
    logger = get_logger("test", CONTEXT)

    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("failed")

    line = json.loads(stream.getvalue())
    assert line["level"] == "ERROR"
    assert "ValueError: boom" in line["exception"]


def test_landing_parser_builds_run_context(tmp_path):
    args = landing_parser("test").parse_args(
        ["--bundle", "b", "--target", "t", "--run-id", "7", "--landing-root", str(tmp_path)]
    )

    assert run_context(args, "w") == RunContext(bundle="b", target="t", workload="w", run_id="7")
    assert args.landing_root == tmp_path
