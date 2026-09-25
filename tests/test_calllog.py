from datetime import UTC, datetime

from kodds.calllog import CROCKFORD, CallLog, ulid


def test_ulid_shape_and_time_order():
    a, b = ulid(1_000), ulid(2_000)
    assert len(a) == 26 and set(a) <= set(CROCKFORD)
    assert a < b
    assert ulid(1_000) != ulid(1_000)


def test_log_file_is_per_utc_month(tmp_path):
    log = CallLog(tmp_path)
    when = datetime(2026, 9, 30, 23, 59, tzinfo=UTC)
    assert log.path(when) == tmp_path / "2026-09.jsonl"


def test_append_is_one_line_per_record(tmp_path):
    log = CallLog(tmp_path / "calls")
    assert log.append({"a": 1}) and log.append({"a": "é"})
    assert log.path().read_text(encoding="utf-8") == '{"a":1}\n{"a":"é"}\n'


def test_unserialisable_record_is_a_warning_not_an_exception(tmp_path, capsys):
    log = CallLog(tmp_path)
    assert log.append({"a": object()}) is False
    assert "call log write" in capsys.readouterr().err


def test_warning_rearms_after_a_success(tmp_path, capsys):
    log = CallLog(tmp_path)
    log.append({"a": object()})
    log.append({"a": object()})
    log.append({"a": 1})
    log.append({"a": object()})
    assert capsys.readouterr().err.count("call log write") == 2
