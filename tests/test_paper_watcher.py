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


# ── ArxivFetcher ──────────────────────────────────────────────────────────────

SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2405.00001v1</id>
    <title>Test Paper Title</title>
    <summary>This is the abstract text.</summary>
    <author><name>Alice</name></author>
    <author><name>Bob</name></author>
    <published>2026-05-11T00:00:00Z</published>
  </entry>
</feed>"""

SAMPLE_ATOM_OLD = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>Old Paper</title>
    <summary>Old abstract.</summary>
    <author><name>Carol</name></author>
    <published>2023-01-01T00:00:00Z</published>
  </entry>
</feed>"""


def test_arxiv_fetcher_parse_returns_paper(tmp_path):
    state = StateManager(str(tmp_path / "sent_ids.json")).load()
    fetcher = ArxivFetcher("cat:cs.CV", max_results=10, date_window_days=365)
    cutoff = datetime.now(timezone.utc) - timedelta(days=365)
    papers = fetcher._parse(SAMPLE_ATOM, state, cutoff)
    assert len(papers) == 1
    assert papers[0].title == "Test Paper Title"
    assert papers[0].arxiv_id == "2405.00001"
    assert papers[0].authors == ["Alice", "Bob"]
    assert papers[0].abstract == "This is the abstract text."


def test_arxiv_fetcher_parse_filters_sent_ids(tmp_path):
    state = StateManager(str(tmp_path / "sent_ids.json")).load()
    state.mark_sent("2405.00001")
    fetcher = ArxivFetcher("cat:cs.CV", max_results=10, date_window_days=365)
    cutoff = datetime.now(timezone.utc) - timedelta(days=365)
    papers = fetcher._parse(SAMPLE_ATOM, state, cutoff)
    assert len(papers) == 0


def test_arxiv_fetcher_parse_filters_old_papers(tmp_path):
    state = StateManager(str(tmp_path / "sent_ids.json")).load()
    fetcher = ArxivFetcher("cat:cs.CV", max_results=10, date_window_days=2)
    cutoff = datetime.now(timezone.utc) - timedelta(days=2)
    papers = fetcher._parse(SAMPLE_ATOM_OLD, state, cutoff)
    assert len(papers) == 0


def test_arxiv_fetcher_fetch_calls_api(tmp_path):
    state = StateManager(str(tmp_path / "sent_ids.json")).load()
    fetcher = ArxivFetcher("cat:cs.CV", max_results=10, date_window_days=365)
    with patch("paper_watcher.requests") as mock_requests:
        mock_requests.get.return_value.text = SAMPLE_ATOM
        mock_requests.get.return_value.raise_for_status = MagicMock()
        papers = fetcher.fetch(state)
    mock_requests.get.assert_called_once()
    _, call_kwargs = mock_requests.get.call_args
    params = call_kwargs["params"]
    assert params["sortBy"] == "submittedDate"
    assert params["sortOrder"] == "descending"
    assert params["max_results"] == 10
    assert len(papers) == 1
    assert papers[0].arxiv_id == "2405.00001"
