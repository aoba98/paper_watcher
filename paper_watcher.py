import json
import logging
import os
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
    pass


class ArxivFetcher:
    pass


class LLMAnalyzer:
    pass


class Mailer:
    pass


def main():
    pass


if __name__ == "__main__":
    main()
