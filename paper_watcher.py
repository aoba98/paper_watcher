import json
import logging
import os
import re
import smtplib
import ssl
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

ARXIV_QUERY = "cat:cs.CV OR cat:cs.AI OR cat:cs.LG"
MAX_RESULTS = 100
MIN_RELEVANCE_SCORE = 7  # papers below this score are dropped before emailing
DATE_WINDOW_DAYS = 2
DEEPSEEK_MODEL = "deepseek-chat"
LLM_TEMPERATURE = 0.2
STATE_FILE = str(Path(__file__).parent / "sent_ids.json")
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_NS = "{http://www.w3.org/2005/Atom}"

SYSTEM_PROMPT = """\
You are a research paper relevance analyzer for an AI researcher \
specializing in video generation and multimodal learning.

Analyze the given paper and return a JSON object with these exact fields:
- is_relevant (bool): true if the paper relates to the researcher's interests
- relevance_score (int 1-10): overall relevance score
- main_direction (str): primary research direction
- sub_directions (list[str]): secondary topics covered
- task (str): what task/problem the paper addresses
- method (str): key method or technique proposed
- innovation (str): main contribution vs prior work
- possible_use_for_me (str): specific actionable insight for this researcher
- keywords (list[str]): 3-6 technical keywords
- priority (str): "high" / "medium" / "low"
- one_sentence_summary (str): one-sentence summary in Simplified Chinese

Priority rules:
- high: directly addresses video generation post-training, reward models, \
preference data, or DiT/MMDiT architectures
- medium: adjacent methods (general RLHF, diffusion theory, multimodal)
- low: only loosely related

Always write the one_sentence_summary field in Simplified Chinese."""

USER_TEMPLATE = """\
My research interests:
1. Video generation model post-training
2. SFT / DPO / RLHF / reward model
3. Aesthetic reward / video reward
4. Prompt-caption data construction
5. Preference data / pairwise data
6. Diffusion / DiT / MMDiT
7. Multimodal agent / RAG

Paper to analyze:
Title: {title}
Abstract: {abstract}"""


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: str
    link: str


class StateManager:
    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = Path(state_file)
        self.sent_ids: set[str] = set()

    def load(self) -> "StateManager":
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.sent_ids = set(data)
            except (json.JSONDecodeError, TypeError):
                logging.warning("State file corrupt or unreadable; starting with empty set.")
                self.sent_ids = set()
        else:
            self.sent_ids = set()
        return self

    def save(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(self.sent_ids), indent=2))
        os.replace(tmp, self.state_file)

    def is_new(self, arxiv_id: str) -> bool:
        return arxiv_id not in self.sent_ids

    def mark_sent(self, arxiv_id: str):
        self.sent_ids.add(arxiv_id)


class ArxivFetcher:
    def __init__(self, query: str, max_results: int, date_window_days: int = DATE_WINDOW_DAYS):
        self.query = query
        self.max_results = max_results
        self.date_window_days = date_window_days

    def fetch(self, state: StateManager) -> list:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.date_window_days)
        params = {
            "search_query": self.query,
            "max_results": self.max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        response = requests.get(ARXIV_API_URL, params=params, timeout=30)
        response.raise_for_status()
        return self._parse(response.text, state, cutoff)

    def _parse(self, xml_text: str, state: StateManager, cutoff: datetime) -> list:
        root = ET.fromstring(xml_text)
        papers = []
        for entry in root.findall(f"{ARXIV_NS}entry"):
            try:
                raw_id = entry.find(f"{ARXIV_NS}id").text.strip()
                arxiv_id = re.sub(r"v\d+$", "", raw_id.split("/abs/")[-1])

                published_str = entry.find(f"{ARXIV_NS}published").text.strip()
                published_dt = datetime.fromisoformat(published_str.replace("Z", "+00:00"))

                if published_dt < cutoff:
                    continue
                if not state.is_new(arxiv_id):
                    continue

                title = entry.find(f"{ARXIV_NS}title").text.strip().replace("\n", " ")
                abstract = entry.find(f"{ARXIV_NS}summary").text.strip().replace("\n", " ")
                authors = [
                    a.find(f"{ARXIV_NS}name").text.strip()
                    for a in entry.findall(f"{ARXIV_NS}author")
                ]
                papers.append(Paper(
                    arxiv_id=arxiv_id,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    published=published_str[:10],
                    link=raw_id,
                ))
            except (AttributeError, ValueError) as e:
                logging.warning("Skipping malformed arXiv entry: %s", e)
                continue
        return papers


class LLMAnalyzer:
    pass


class Mailer:
    pass


def main():
    pass


if __name__ == "__main__":
    main()
