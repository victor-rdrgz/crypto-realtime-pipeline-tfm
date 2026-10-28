import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from crypto_realtime_pipeline.common.landing import LandingCheckError, LandingRun, run_folder

RUN_TS = datetime(2026, 10, 27, 21, 5, 9, tzinfo=UTC)


def test_run_folder_uses_utc_partitions(tmp_path):
    madrid = RUN_TS.astimezone(timezone(timedelta(hours=1)))

    assert run_folder(tmp_path, "ds", madrid) == (
        tmp_path / "ds" / "extraction_date=2026-10-27" / "run_ts=20261027T210509Z"
    )


def test_writes_ndjson_pages_with_metadata(tmp_path):
    run = LandingRun(root=tmp_path, dataset="ds", source="api", run_ts=RUN_TS)

    run.write_page("https://api/1", [{"a": 1}, {"a": 2}])
    run.write_page("https://api/2", [{"a": 3}])

    first = (run.path / "part-00001.ndjson").read_text(encoding="utf-8").splitlines()
    second = (run.path / "part-00002.ndjson").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in first + second]
    assert [r["a"] for r in records] == [1, 2, 3]
    assert {r["_dataset_name"] for r in records} == {"ds"}
    assert [r["_page"] for r in records] == [1, 1, 2]
    assert records[2]["_source_url"] == "https://api/2"
    assert all(datetime.fromisoformat(r["_extracted_at"]).tzinfo for r in records)
    assert run.metrics() == {
        "dataset": "ds",
        "source": "api",
        "target_path": str(run.path),
        "pages": 2,
        "records": 3,
        "bytes": sum(f.stat().st_size for f in run.path.iterdir()),
    }


def test_never_reuses_an_existing_run_folder(tmp_path):
    LandingRun(root=tmp_path, dataset="ds", source="api", run_ts=RUN_TS)

    with pytest.raises(LandingCheckError, match="already exists"):
        LandingRun(root=tmp_path, dataset="ds", source="api", run_ts=RUN_TS)


def test_rejects_records_that_are_not_objects(tmp_path):
    run = LandingRun(root=tmp_path, dataset="ds", source="api", run_ts=RUN_TS)

    with pytest.raises(LandingCheckError, match="not a JSON object"):
        run.write_page("https://api", [[1, 2]])


def test_rejects_records_with_metadata_columns(tmp_path):
    run = LandingRun(root=tmp_path, dataset="ds", source="api", run_ts=RUN_TS)

    with pytest.raises(LandingCheckError, match="_page"):
        run.write_page("https://api", [{"_page": 9}])


def test_check_fails_on_empty_run_unless_allowed(tmp_path):
    run = LandingRun(root=tmp_path, dataset="ds", source="api", run_ts=RUN_TS)
    run.write_page("https://api", [])

    with pytest.raises(LandingCheckError, match="no records"):
        run.check()
    run.check(allow_empty=True)
    assert (run.path / "part-00001.ndjson").read_bytes() == b""


def test_check_compares_with_declared_total(tmp_path):
    run = LandingRun(root=tmp_path, dataset="ds", source="api", run_ts=RUN_TS)
    run.write_page("https://api", [{"a": 1}])

    run.check(expected_records=1)
    with pytest.raises(LandingCheckError, match="wrote 1 records, source declared 2"):
        run.check(expected_records=2)
