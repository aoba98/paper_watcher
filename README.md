# Paper Watcher

Daily arXiv paper digest tailored to your research interests, powered by DeepSeek LLM and sent to your inbox via email.

## How It Works

1. Fetches the latest papers from arXiv (cs.CV, cs.AI, cs.LG by default)
2. Scores each paper for relevance using DeepSeek API
3. Filters papers with `relevance_score >= 7`
4. Sends a ranked plain-text email digest
5. Persists processed paper IDs to avoid duplicates across runs

## Local Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your actual values
```

| Variable | Description | Example |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API key (from platform.deepseek.com) | `sk-...` |
| `EMAIL_SENDER` | Your sending email address | `you@gmail.com` |
| `EMAIL_PASSWORD` | SMTP authorization code (not your login password) | `xxxx xxxx xxxx xxxx` |
| `EMAIL_RECIPIENT` | Where to receive the digest | `you@gmail.com` |
| `SMTP_HOST` | SMTP server | `smtp.gmail.com` |
| `SMTP_PORT` | `465` for SSL, `587` for STARTTLS | `465` |

**Gmail users:** Enable 2FA, then generate an App Password at https://myaccount.google.com/apppasswords and use that as `EMAIL_PASSWORD`.

### 3. Run

```bash
python paper_watcher.py
```

## GitHub Actions Setup

Add the following **Repository Secrets** at `Settings → Secrets and variables → Actions → New repository secret`:

| Secret Name | Value |
|---|---|
| `DEEPSEEK_API_KEY` | Your DeepSeek API key |
| `EMAIL_SENDER` | Sender email |
| `EMAIL_PASSWORD` | SMTP authorization code |
| `EMAIL_RECIPIENT` | Recipient email |
| `SMTP_HOST` | e.g. `smtp.gmail.com` |
| `SMTP_PORT` | e.g. `465` |

The workflow runs automatically every day at **09:00 Beijing time** and can also be triggered manually from the **Actions** tab via `workflow_dispatch`.

## Configuration

Key constants at the top of `paper_watcher.py`:

| Constant | Default | Description |
|---|---|---|
| `ARXIV_QUERY` | `cat:cs.CV OR cat:cs.AI OR cat:cs.LG` | arXiv search query |
| `MAX_RESULTS` | `100` | Max papers to fetch per run |
| `MIN_RELEVANCE_SCORE` | `7` | Minimum score to include in email |
| `DATE_WINDOW_DAYS` | `2` | How many days back to look |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | DeepSeek model to use |

## Running Tests

```bash
pytest tests/ -v
```
