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
