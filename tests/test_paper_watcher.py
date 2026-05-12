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


# ── LLMAnalyzer ───────────────────────────────────────────────────────────────

SAMPLE_ANALYSIS = {
    "is_relevant": True,
    "relevance_score": 9,
    "main_direction": "video reward model",
    "sub_directions": ["DPO", "preference data"],
    "task": "score video generation quality",
    "method": "contrastive reward model",
    "innovation": "first video-specific reward model",
    "possible_use_for_me": "use as scoring baseline for aesthetic data",
    "keywords": ["video reward", "RLHF", "DiT"],
    "priority": "high",
    "one_sentence_summary": "提出了视频生成奖励模型。",
}


def _make_analyzer_with_mock_client(mock_client):
    analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
    analyzer.client = mock_client
    analyzer.model = "deepseek-chat"
    analyzer.temperature = 0.2
    return analyzer


def _make_paper(arxiv_id="2405.001"):
    paper = MagicMock(spec=Paper)
    paper.title = "Test Paper"
    paper.abstract = "Test abstract."
    paper.arxiv_id = arxiv_id
    return paper


def test_llm_analyzer_returns_parsed_dict():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        json.dumps(SAMPLE_ANALYSIS)
    )
    analyzer = _make_analyzer_with_mock_client(mock_client)
    result = analyzer.analyze(_make_paper())
    assert result["is_relevant"] is True
    assert result["relevance_score"] == 9
    assert result["priority"] == "high"


def test_llm_analyzer_returns_none_on_api_exception():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("network error")
    analyzer = _make_analyzer_with_mock_client(mock_client)
    result = analyzer.analyze(_make_paper())
    assert result is None


def test_llm_analyzer_returns_none_on_invalid_json():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "not valid json {"
    )
    analyzer = _make_analyzer_with_mock_client(mock_client)
    result = analyzer.analyze(_make_paper())
    assert result is None


def test_llm_analyzer_passes_correct_messages():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        json.dumps(SAMPLE_ANALYSIS)
    )
    analyzer = _make_analyzer_with_mock_client(mock_client)
    paper = _make_paper()
    analyzer.analyze(paper)
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "one_sentence_summary" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert paper.title in messages[1]["content"]
    assert paper.abstract in messages[1]["content"]


def test_llm_analyzer_returns_none_on_invalid_schema():
    mock_client = MagicMock()
    # LLM returns valid JSON but wrong types (score is a string, not int)
    mock_client.chat.completions.create.return_value.choices[0].message.content = json.dumps({
        "is_relevant": True,
        "relevance_score": "9",  # string instead of int
        "priority": "high",
    })
    analyzer = _make_analyzer_with_mock_client(mock_client)
    result = analyzer.analyze(_make_paper())
    assert result is None


# ── Mailer ────────────────────────────────────────────────────────────────────

def _make_mailer():
    return Mailer(
        sender="sender@example.com",
        password="secret",
        recipient="recv@example.com",
        smtp_host="smtp.example.com",
        smtp_port=465,
    )


def _make_paper_analysis_pair(
    title="My Paper",
    authors=None,
    published="2026-05-11",
    link="https://arxiv.org/abs/2405.001",
    priority="high",
    score=9,
    sub_directions=None,
    main_direction="video reward",
    summary="一句话总结",
    possible_use="对我的启发",
):
    paper = MagicMock(spec=Paper)
    paper.title = title
    paper.authors = authors or ["Alice", "Bob", "Carol", "Dave"]
    paper.published = published
    paper.link = link
    analysis = {
        "priority": priority,
        "relevance_score": score,
        "sub_directions": sub_directions or ["DPO"],
        "main_direction": main_direction,
        "one_sentence_summary": summary,
        "possible_use_for_me": possible_use,
    }
    return paper, analysis


def test_mailer_format_body_contains_key_fields():
    mailer = _make_mailer()
    pair = _make_paper_analysis_pair()
    body = mailer._format_body("2026-05-12", [pair])
    assert "My Paper" in body
    assert "HIGH" in body
    assert "9/10" in body
    assert "https://arxiv.org/abs/2405.001" in body
    assert "一句话总结" in body
    assert "对我的启发" in body


def test_mailer_format_body_truncates_long_author_list():
    mailer = _make_mailer()
    paper, analysis = _make_paper_analysis_pair(authors=["A", "B", "C", "D"])
    body = mailer._format_body("2026-05-12", [(paper, analysis)])
    assert "等" in body


def test_mailer_format_body_short_author_list_no_truncation():
    mailer = _make_mailer()
    paper, analysis = _make_paper_analysis_pair(authors=["Alice", "Bob"])
    body = mailer._format_body("2026-05-12", [(paper, analysis)])
    assert "等" not in body


def test_mailer_send_uses_smtp_ssl_for_port_465():
    mailer = _make_mailer()
    with patch("paper_watcher.smtplib.SMTP_SSL") as mock_ssl:
        mock_server = MagicMock()
        mock_ssl.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_ssl.return_value.__exit__ = MagicMock(return_value=False)
        mailer.send([])
    mock_ssl.assert_called_once()
    mock_server.login.assert_called_once_with("sender@example.com", "secret")
    mock_server.sendmail.assert_called_once()


def test_mailer_send_uses_starttls_for_port_587():
    mailer = Mailer("a@b.com", "pass", "c@d.com", "smtp.example.com", 587)
    with patch("paper_watcher.smtplib.SMTP") as mock_smtp:
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
        mailer.send([])
    mock_smtp.assert_called_once()
    mock_server.starttls.assert_called_once()
    mock_server.login.assert_called_once_with("a@b.com", "pass")
    mock_server.sendmail.assert_called_once()


def test_mailer_zero_results_sends_no_match_email():
    import email
    mailer = _make_mailer()
    with patch("paper_watcher.smtplib.SMTP_SSL") as mock_ssl:
        mock_server = MagicMock()
        mock_ssl.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_ssl.return_value.__exit__ = MagicMock(return_value=False)
        mailer.send([])
    sent_args = mock_server.sendmail.call_args
    msg_bytes = sent_args[0][2]
    msg = email.message_from_bytes(msg_bytes)
    subject = email.header.decode_header(msg["Subject"])
    subject_str = "".join(
        part.decode(enc or "utf-8") if isinstance(part, bytes) else part
        for part, enc in subject
    )
    body_str = ""
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            body_str = part.get_payload(decode=True).decode("utf-8")
            break
    assert "今日无匹配论文" in subject_str or "无匹配" in body_str
