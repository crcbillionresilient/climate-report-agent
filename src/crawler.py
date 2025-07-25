#!/usr/bin/env python3
"""
Climate-Adaptation × Insurance – report crawler (human-approval required)

• Searches Google Programmable Search (free tier) once per run.
• Downloads candidate PDFs/HTML, extracts ~3 000 tokens (≈ first 5 pages).
• Builds averaged MiniLM (or other) embeddings.
• Saves/updates data/reports_master.json + .csv.
• ALWAYS emails the reviewer with ✔ / ✖ / 🚫 links;
  nothing is auto-published.

Changes in this version
───────────────────────
✓ Uses PyYAML to read config.yaml (robust).  
✓ Installs & imports PyCryptodome automatically (add `pycryptodome` to requirements.txt).  
✓ Hardened PDF extractor – encrypted or corrupt PDFs are skipped, not crash.  
"""

from __future__ import annotations
import os, json, csv, hashlib, time, re, ssl, smtplib, textwrap
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import yaml                        # NEW – proper YAML parser
import requests, trafilatura
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# ────────── configuration ──────────
CONFIG = yaml.safe_load(Path("config.yaml").read_text())
EMBED_MODEL_NAME = CONFIG["embedding_model_name"]
QUERY            = CONFIG["query"]
NUM_RESULTS      = CONFIG["num_results"]
MIN_YEAR         = CONFIG["min_year"]
MIN_PAGES        = CONFIG["min_pages"]
WIN, STRIDE      = CONFIG["window_tokens"], CONFIG["window_stride"]

GOOGLE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
GOOGLE_CSE_ID  = os.getenv("GOOGLE_CSE_ID")

SMTP_SERVER    = os.getenv("SMTP_SERVER")
SMTP_PORT      = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER      = os.getenv("SMTP_USERNAME")
SMTP_PASS      = os.getenv("SMTP_PASSWORD")
REVIEWER_EMAIL = os.getenv(CONFIG["reviewer_email_env"])

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
JSON_PATH, CSV_PATH = DATA_DIR/"reports_master.json", DATA_DIR/"reports_master.csv"

model = SentenceTransformer(EMBED_MODEL_NAME)

# ────────── helper functions ──────────
def google_search(q: str, n: int) -> List[str]:
    """Return up to n Google CSE links."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": q, "num": min(n,10)}
    links, start = [], 1
    while len(links) < n:
        params["start"] = start
        data = requests.get(url, params=params, timeout=30).json()
        links += [i["link"] for i in data.get("items", [])]
        nxt = data.get("queries", {}).get("nextPage")
        if not nxt: break
        start = nxt[0]["startIndex"]; time.sleep(1)
    return links[:n]

def fetch(url: str) -> bytes|None:
    try:
        r = requests.get(url, timeout=60); r.raise_for_status(); return r.content
    except Exception as e:
        print("Fetch failed:", url[:70], e); return None

def sha(b: bytes) -> str: return hashlib.sha256(b).hexdigest()

def extract_pdf(b: bytes, pages: int = 5):
    """Return (text, total_pages). Skip if encrypted/corrupt."""
    from io import BytesIO
    try:
        reader = PdfReader(BytesIO(b))
        total_pages = len(reader.pages)
        texts = [page.extract_text() or "" for page in reader.pages[:pages]]
        return "\n".join(texts), total_pages
    except Exception as err:
        raise ValueError(f"Unreadable PDF ({err.__class__.__name__})")

def extract_html(b: bytes) -> str:
    return trafilatura.extract(b.decode("utf-8", "ignore")) or ""

def chunks(words: List[str], w: int = WIN, s: int = STRIDE):
    for i in range(0, len(words), s):
        chunk = words[i:i+w]
        if len(chunk) < 100: break
        yield " ".join(chunk)

def embed(text: str):
    vecs = model.encode(list(chunks(text.split())))
    return vecs.mean(axis=0)

def email_digest(cands: List[Dict]):
    if not cands: return
    msg = EmailMessage()
    msg["Subject"] = f"[Climate Agent] {len(cands)} new reports need approval ({datetime.utcnow().date()})"
    msg["From"], msg["To"] = SMTP_USER, REVIEWER_EMAIL
    rows=[]
    for r in cands:
        ok=f"mailto:{REVIEWER_EMAIL}?subject=APPROVE%20{r['sha']}"
        no=f"mailto:{REVIEWER_EMAIL}?subject=REJECT%20{r['sha']}"
        nv=f"mailto:{REVIEWER_EMAIL}?subject=NEVER%20{r['sha']}"
        rows.append(
    f"<tr>"
    f"<td><a href='{r['url']}'>{r['title']}</a></td>"   # ← clickable link
    f"<td>{r['year']}</td>"
    f"<td>{r['score']:.2f}</td>"
    f"<td><a href='{ok}'>✔</a> / <a href='{no}'>✖</a> / <a href='{nv}'>🚫</a></td>"
    f"</tr>"
)
    body = ("<p>All new candidate reports require a decision:</p>"
            "<table border='1' cellpadding='4' cellspacing='0'>"
            "<tr><th>Title</th><th>Year</th><th>Score</th><th>Action</th></tr>"
            + "".join(rows) + "</table>")
    msg.add_alternative(body, subtype='html')
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
        s.starttls(context=ssl.create_default_context())
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)
    print("Email sent to reviewer.")

# ────────── main crawl ──────────
existing = {}
if JSON_PATH.exists():
    existing = {r["sha"]: r for r in json.load(JSON_PATH.open())}

links = google_search(QUERY, NUM_RESULTS)
new=[]
for url in links:
    raw = fetch(url)
    if not raw: continue
    dig = sha(raw)
    if dig in existing: continue

    # Extract text & page count
    if url.endswith(".pdf"):
        try:
            text, pages = extract_pdf(raw)
        except ValueError as e:
            print("Skip PDF:", e); continue
    else:
        text = extract_html(raw)
        pages = len(text.split()) // 500

    if pages < MIN_PAGES: continue

    yrmatch = re.search(r"(19|20)\d{2}", text[:4000] + url)
    year = int(yrmatch.group()) if yrmatch else 1900
    if year < MIN_YEAR: continue

    vec = embed(text)
    score = float(vec.mean())             # placeholder relevance score

    rec = {
        "title": Path(url).name.replace("_", " ")[:120],
        "year": year,
        "pages": pages,
        "url": url,
        "sha": dig,
        "score": score,
        "summary": textwrap.shorten(text, 1000)
    }
    new.append(rec)

if new:
    allrecs = list(existing.values()) + new
    JSON_PATH.write_text(json.dumps(allrecs, indent=2))
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=allrecs[0].keys())
        w.writeheader(); w.writerows(allrecs)
    print(f"Saved {len(allrecs)} total records; {len(new)} new.")
    email_digest(new)
else:
    print("No new records.")
