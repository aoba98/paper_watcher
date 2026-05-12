import json
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from paper_watcher import (
    StateManager,
    ArxivFetcher,
    LLMAnalyzer,
    Mailer,
    Paper,
    PRIORITY_ORDER,
    MIN_RELEVANCE_SCORE,
)

# ── StateManager ──────────────────────────────────────────────────────────────

def test_state_manager_load_empty_when_file_missing(tmp_path):
    sm = StateManager(str(tmp_path / "sent_ids.json"))
    sm.load()
    assert sm.sent_ids == set()


def test_state_manager_load_reads_existing_ids(tmp_path):
    f = tmp_path / "sent_ids.json"
    f.write_text(json.dumps(["2405.001", "2405.002"]))
    sm = StateManager(str(f)).load()
    assert sm.sent_ids == {"2405.001", "2405.002"}


def test_state_manager_save_writes_ids(tmp_path):
    f = tmp_path / "sent_ids.json"
    sm = StateManager(str(f)).load()
    sm.mark_sent("2405.001")
    sm.save()
    assert "2405.001" in json.loads(f.read_text())


def test_state_manager_is_new_true_for_unknown(tmp_path):
    sm = StateManager(str(tmp_path / "sent_ids.json")).load()
    assert sm.is_new("2405.999") is True


def test_state_manager_is_new_false_after_mark_sent(tmp_path):
    sm = StateManager(str(tmp_path / "sent_ids.json")).load()
    sm.mark_sent("2405.001")
    assert sm.is_new("2405.001") is False
