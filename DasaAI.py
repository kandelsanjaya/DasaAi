import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from rembg import remove
from PIL import Image
import io
import os
import json
import time
import datetime
import base64
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs

# =========================
# CONFIG
# =========================
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# PRIVATE / RESTRICTED DATA
# =========================
PRIVATE_DATA = {
    "phone": True, "email": True, "address": True,
    "password": True, "api_key": True, "salary": True,
    "bank": True, "nid": True, "passport": True,
}

HARMFUL_PATTERNS = [
    r"\b(kill|murder|suicide|self.harm|harm myself|end my life|want to die|how to die)\b",
    r"\b(bomb|explosive|weapon|gun|knife to hurt)\b",
    r"\b(hack|crack password|steal data|bypass security)\b",
    r"\b(drug|cocaine|heroin|meth|illegal substance)\b",
    r"\b(porn|nude|naked|sexual content)\b",
    r"\b(racist|slur|hate speech)\b",
]

PRIVATE_PATTERNS = [
    r"\b(phone|number|contact|address|email|location|where.*live|personal.*info|private)\b",
    r"\b(password|api key|secret|credential)\b",
    r"\b(bank|account|card number|salary|income)\b",
]

CRISIS_RESOURCES = """
🆘 **If you're in crisis, please reach out:**
- **Nepal:** TPO Nepal — 01-4460084
- **International:** Befrienders Worldwide — www.befrienders.org
- **Crisis Text:** Text HOME to 741741 (US)
- **WHO Mental Health:** www.who.int/mental_health

You are not alone. 💙
"""

def is_harmful(text):
    t = text.lower()
    for p in HARMFUL_PATTERNS:
        if re.search(p, t):
            return True
    return False

def is_asking_private(text):
    t = text.lower()
    for p in PRIVATE_PATTERNS:
        if re.search(p, t):
            return True
    return False

def is_crisis(text):
    t = text.lower()
    crisis_words = ["suicide", "kill myself", "end my life", "want to die", "self harm", "self-harm", "hurt myself"]
    return any(w in t for w in crisis_words)

# =========================
# URL UTILITIES
# =========================
def is_youtube_url(url):
    return any(d in url for d in ["youtube.com", "youtu.be"])

def is_portfolio_url(url):
    portfolio_keywords = ["portfolio", "resume", "cv", "about", "work", "projects", "hire", "designer", "developer", "creative"]
    return any(k in url.lower() for k in portfolio_keywords)

def get_youtube_video_id(url):
    parsed = urlparse(url)
    if "youtu.be" in url:
        return parsed.path.strip("/")
    qs = parse_qs(parsed.query)
    return qs.get("v", [None])[0]

def fetch_youtube_info(url):
    """Fetch YouTube page metadata via scraping (no API key needed)."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.find("meta", {"name": "title"}) or soup.find("title")
        title = title.get("content", title.text) if title else "Unknown Title"
        description = soup.find("meta", {"name": "description"})
        description = description.get("content", "") if description else ""
        keywords = soup.find("meta", {"name": "keywords"})
        keywords = keywords.get("content", "") if keywords else ""
        channel = soup.find("link", {"itemprop": "name"})
        channel = channel.get("content", "Unknown Channel") if channel else "Unknown Channel"
        return {"title": title, "description": description, "keywords": keywords, "channel": channel, "url": url}
    except Exception as e:
        return {"title": "Could not fetch", "description": str(e), "keywords": "", "channel": "", "url": url}

def fetch_website_content(url):
    """Scrape general website content for analysis."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        r = requests.get(url, headers=headers, timeout=12)
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove scripts/styles
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        title = soup.title.text.strip() if soup.title else "No Title"
        meta_desc = soup.find("meta", {"name": "description"})
        meta_desc = meta_desc.get("content", "") if meta_desc else ""

        # Grab headings and paragraphs
        headings = [h.text.strip() for h in soup.find_all(["h1", "h2", "h3"])[:15]]
        paragraphs = [p.text.strip() for p in soup.find_all("p")[:20] if len(p.text.strip()) > 40]
        links = list(set([a.get("href", "") for a in soup.find_all("a", href=True) if a.get("href", "").startswith("http")][:20]))

        # Detect technologies / social proof signals
        text_lower = r.text.lower()
        tech_signals = {
            "React": "react" in text_lower,
            "Next.js": "next.js" in text_lower or "_next" in text_lower,
            "Vue": "vue" in text_lower,
            "Tailwind": "tailwind" in text_lower,
            "GitHub": "github" in text_lower,
            "LinkedIn": "linkedin" in text_lower,
            "Contact Form": "contact" in text_lower,
            "Projects Section": "project" in text_lower,
            "Blog": "blog" in text_lower,
        }
        detected_tech = [k for k, v in tech_signals.items() if v]

        return {
            "title": title,
            "meta_desc": meta_desc,
            "headings": headings,
            "paragraphs": paragraphs[:8],
            "links": links[:10],
            "detected_tech": detected_tech,
            "url": url,
            "status": r.status_code
        }
    except Exception as e:
        return {"title": "Error", "meta_desc": str(e), "headings": [], "paragraphs": [], "links": [], "detected_tech": [], "url": url, "status": 0}

# =========================
# HELPERS
# =========================
def embed(text):
    return embedding_model.encode(text)

def load_pdf(file):
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def chunk(text, size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return chunks

def build_index(chunks):
    vectors = np.array([embed(c) for c in chunks]).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index

def search(query):
    q = embed(query).astype("float32").reshape(1, -1)
    _, I = st.session_state.index.search(q, 3)
    return "\n".join([st.session_state.chunks[i] for i in I[0]])

def groq_chat(prompt, system="You are a helpful, smart, and friendly AI assistant.", stream=True):
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        stream=stream
    )
    return res

def groq_chat_no_stream(prompt, system="You are a helpful AI assistant."):
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )
    return res.choices[0].message.content

def get_download_link(content, filename, label, mime="text/plain"):
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:{mime};base64,{b64}" download="{filename}" style="display:inline-block;padding:8px 18px;background:linear-gradient(90deg,#6366f1,#8b5cf6);color:white;border-radius:8px;text-decoration:none;font-weight:600;margin-top:10px;">⬇️ {label}</a>'

def stream_response(prompt, system="You are a helpful, frank, and friendly AI."):
    box = st.empty()
    out = ""
    for r in groq_chat(prompt, system=system, stream=True):
        if r.choices[0].delta.content:
            out += r.choices[0].delta.content
            box.markdown(out + "▌")
            time.sleep(0.01)
    box.markdown(out)
    return out

# =========================
# UI CONFIG
# =========================
st.set_page_config(page_title="⚔️ Guardian AI", layout="wide", page_icon="⚔️")

THEMES = {
    "🌌 Cyber Dark": {"bg": "#0a0a0f", "card": "#13131a", "accent": "#6366f1", "accent2": "#8b5cf6", "text": "#e2e8f0", "sub": "#94a3b8"},
    "🌊 Ocean Blue": {"bg": "#0c1a2e", "card": "#0f2744", "accent": "#38bdf8", "accent2": "#0ea5e9", "text": "#e0f2fe", "sub": "#7dd3fc"},
    "🌿 Matrix Green": {"bg": "#0a1a0a", "card": "#0f2010", "accent": "#22c55e", "accent2": "#16a34a", "text": "#dcfce7", "sub": "#86efac"},
    "🔥 Red Neon": {"bg": "#150a0a", "card": "#200f0f", "accent": "#ef4444", "accent2": "#dc2626", "text": "#fee2e2", "sub": "#fca5a5"},
    "💜 Purple Haze": {"bg": "#0f0a1a", "card": "#180f2a", "accent": "#a855f7", "accent2": "#9333ea", "text": "#f3e8ff", "sub": "#d8b4fe"},
    "🌸 Pink Dream": {"bg": "#1a0a12", "card": "#2a0f1c", "accent": "#ec4899", "accent2": "#db2777", "text": "#fce7f3", "sub": "#f9a8d4"},
    "🌙 Midnight": {"bg": "#060612", "card": "#0d0d1f", "accent": "#818cf8", "accent2": "#6366f1", "text": "#eef2ff", "sub": "#a5b4fc"},
    "🟠 Orange Flame": {"bg": "#150a00", "card": "#211000", "accent": "#f97316", "accent2": "#ea580c", "text": "#fff7ed", "sub": "#fdba74"},
}

if "theme" not in st.session_state:
    st.session_state.theme = "🌌 Cyber Dark"

theme_name = st.sidebar.selectbox("🎨 Theme", list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state.theme))
st.session_state.theme = theme_name
T = THEMES[theme_name]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

* {{ box-sizing: border-box; }}

.stApp {{
    background: {T['bg']};
    color: {T['text']};
    font-family: 'Inter', sans-serif;
}}

.stApp::before {{
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(99,102,241,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,102,241,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}}

section[data-testid="stSidebar"] {{
    background: {T['card']} !important;
    border-right: 1px solid {T['accent']}22;
}}

section[data-testid="stSidebar"] * {{
    color: {T['text']} !important;
}}

.navbar {{
    background: linear-gradient(135deg, {T['card']}, {T['bg']});
    border: 1px solid {T['accent']}33;
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 0 40px {T['accent']}22, inset 0 1px 0 {T['accent']}33;
    position: relative;
    overflow: hidden;
}}

.navbar::before {{
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 60%;
    height: 2px;
    background: linear-gradient(90deg, transparent, {T['accent']}, transparent);
    animation: scanline 3s linear infinite;
}}

@keyframes scanline {{
    0% {{ left: -100%; }}
    100% {{ left: 200%; }}
}}

.navbar-title {{
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem;
    font-weight: 900;
    background: linear-gradient(90deg, {T['accent']}, {T['accent2']}, {T['text']});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 2px;
}}

.navbar-badge {{
    background: {T['accent']}22;
    border: 1px solid {T['accent']}55;
    color: {T['accent']};
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 1px;
}}

.navbar-time {{
    color: {T['sub']};
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}}

.g-card {{
    background: {T['card']};
    border: 1px solid {T['accent']}22;
    border-radius: 14px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: border-color 0.3s;
}}

.g-card:hover {{
    border-color: {T['accent']}55;
}}

.chat-user {{
    background: linear-gradient(135deg, {T['accent']}33, {T['accent']}11);
    border: 1px solid {T['accent']}44;
    border-radius: 16px 16px 4px 16px;
    padding: 14px 18px;
    margin: 8px 0 8px 60px;
    color: {T['text']};
    font-size: 0.95rem;
    position: relative;
    animation: slideInRight 0.3s ease;
}}

.chat-ai {{
    background: linear-gradient(135deg, {T['card']}, {T['bg']});
    border: 1px solid {T['accent']}22;
    border-radius: 16px 16px 16px 4px;
    padding: 14px 18px;
    margin: 8px 60px 8px 0;
    color: {T['text']};
    font-size: 0.95rem;
    position: relative;
    animation: slideInLeft 0.3s ease;
}}

.chat-label {{
    font-size: 0.7rem;
    color: {T['sub']};
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 4px;
    letter-spacing: 1px;
}}

@keyframes slideInRight {{
    from {{ transform: translateX(20px); opacity: 0; }}
    to {{ transform: translateX(0); opacity: 1; }}
}}

@keyframes slideInLeft {{
    from {{ transform: translateX(-20px); opacity: 0; }}
    to {{ transform: translateX(0); opacity: 1; }}
}}

.section-title {{
    font-family: 'Orbitron', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: {T['accent']};
    letter-spacing: 2px;
    margin-bottom: 18px;
    padding-bottom: 10px;
    border-bottom: 1px solid {T['accent']}33;
    text-transform: uppercase;
}}

.badge-online {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #22c55e22;
    border: 1px solid #22c55e55;
    color: #22c55e;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
}}

.badge-online::before {{
    content: '';
    width: 6px; height: 6px;
    background: #22c55e;
    border-radius: 50%;
    animation: pulse 1.5s infinite;
}}

@keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.5; transform: scale(1.3); }}
}}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {{
    background: {T['card']} !important;
    border: 1px solid {T['accent']}33 !important;
    color: {T['text']} !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
}}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {{
    border-color: {T['accent']} !important;
    box-shadow: 0 0 0 2px {T['accent']}22 !important;
}}

.stButton > button {{
    background: linear-gradient(135deg, {T['accent']}, {T['accent2']}) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 22px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 15px {T['accent']}44 !important;
}}

.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px {T['accent']}66 !important;
}}

.stSelectbox > div > div {{
    background: {T['card']} !important;
    border-color: {T['accent']}33 !important;
    color: {T['text']} !important;
}}

::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {T['bg']}; }}
::-webkit-scrollbar-thumb {{ background: {T['accent']}55; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {T['accent']}; }}

.crisis-box {{
    background: #ef444422;
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #fecaca;
}}

.restrict-box {{
    background: #f9731622;
    border: 1px solid #f97316;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #fed7aa;
}}

code {{
    font-family: 'JetBrains Mono', monospace !important;
    background: {T['accent']}11 !important;
    border: 1px solid {T['accent']}22 !important;
    border-radius: 6px !important;
    padding: 2px 6px !important;
    color: {T['accent']} !important;
}}

.stat-row {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 16px;
}}

.stat-card {{
    background: {T['card']};
    border: 1px solid {T['accent']}22;
    border-radius: 12px;
    padding: 14px 20px;
    flex: 1;
    min-width: 100px;
    text-align: center;
}}

.stat-num {{
    font-family: 'Orbitron', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: {T['accent']};
}}

.stat-label {{
    font-size: 0.7rem;
    color: {T['sub']};
    margin-top: 4px;
    letter-spacing: 1px;
    text-transform: uppercase;
}}

.footer {{
    text-align: center;
    color: {T['sub']};
    font-size: 0.75rem;
    padding: 20px;
    border-top: 1px solid {T['accent']}11;
    margin-top: 40px;
    font-family: 'JetBrains Mono', monospace;
}}

.stRadio > div {{ gap: 8px; }}
.stRadio label {{ color: {T['text']} !important; }}

.stFileUploader > div {{
    background: {T['card']} !important;
    border: 1px dashed {T['accent']}44 !important;
    border-radius: 12px !important;
}}

.url-badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    margin-bottom: 10px;
}}

.url-badge-yt {{ background: #ef444422; border: 1px solid #ef4444; color: #fca5a5; }}
.url-badge-portfolio {{ background: {T['accent']}22; border: 1px solid {T['accent']}; color: {T['accent']}; }}
.url-badge-web {{ background: #22c55e22; border: 1px solid #22c55e; color: #86efac; }}

.rating-bar-wrap {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 6px 0;
}}
.rating-bar-label {{
    width: 130px;
    font-size: 0.8rem;
    color: {T['sub']};
}}
.rating-bar-bg {{
    flex: 1;
    background: {T['accent']}11;
    border-radius: 8px;
    height: 8px;
    overflow: hidden;
}}
.rating-bar-fill {{
    height: 8px;
    border-radius: 8px;
    background: linear-gradient(90deg, {T['accent']}, {T['accent2']});
}}
.rating-score {{
    width: 32px;
    text-align: right;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    color: {T['accent']};
    font-weight: 700;
}}

.stSpinner > div {{
    border-color: {T['accent']} transparent transparent !important;
}}
</style>
""", unsafe_allow_html=True)

# =========================
# NAVBAR
# =========================
now = datetime.datetime.now().strftime("%H:%M  |  %d %b %Y")

st.markdown(f"""
<div class="navbar">
    <div>
        <div class="navbar-title">⚔️ DASA STUDIO AI </div>
        <div style="color:{T['sub']};font-size:0.8rem;margin-top:4px;font-family:'Inter',sans-serif;">
            Intelligent · Restricted · Futuristic
        </div>
    </div>
    <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
        <span class="badge-online">ONLINE</span>
        <span class="navbar-badge">LLaMA-3.1</span>
        <span class="navbar-badge">DASA</span>
        <span class="navbar-time">{now}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# NAVIGATION
# =========================
menu = st.sidebar.radio(
    "📌 Navigation",
    [
        "🧠 Chatbot",
        "📄 CV Maker",
        "🌍 Translator",
        "📝 Summarizer",
        "🧹 Background Remover",
        "💻 Code Explainer",
        "😊 Mood Detector",
        "✍️ Grammar Fixer",
        "🖼️ Image & URL Analyzer",
        "📊 Session Stats",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="color:{T['sub']};font-size:0.75rem;font-family:'JetBrains Mono',monospace;padding:8px;background:{T['card']};border-radius:10px;border:1px solid {T['accent']}22;">
⚠️ <b style="color:{T['accent']};">Restrictions Active</b><br>
• Private data protected<br>
• Harmful content blocked<br>
• Crisis detection active<br>
• Code/data locked
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown(f"<div style='color:{T['sub']};font-size:0.72rem;text-align:center;font-family:JetBrains Mono,monospace;'>Guardian AI v2.1 • Built by You</div>", unsafe_allow_html=True)

# =========================
# SESSION INIT
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.datetime.now()
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "blocked_count" not in st.session_state:
    st.session_state.blocked_count = 0

# =========================
# 🧠 CHATBOT
# =========================
if menu == "🧠 Chatbot":
    st.markdown('<div class="section-title">🧠 DASA CHATBOT </div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown(f"""
        <div class="g-card">
            <div style="font-size:0.8rem;color:{T['sub']};font-family:'JetBrains Mono',monospace;margin-bottom:10px;">SESSION INFO</div>
            <div class="stat-row">
                <div class="stat-card">
                    <div class="stat-num">{len(st.session_state.chat_history)}</div>
                    <div class="stat-label">Messages</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">{st.session_state.blocked_count}</div>
                    <div class="stat-label">Blocked</div>
                </div>
            </div>
            <div style="color:{T['sub']};font-size:0.75rem;">
                {'📚 PDF Loaded' if 'index' in st.session_state else '📄 No PDF loaded'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        file = st.file_uploader("📎 Upload PDF ", type=["pdf"])
        if file and st.button("⚡ Process PDF"):
            with st.spinner("Embedding PDF..."):
                text = load_pdf(file)
                chunks = chunk(text)
                idx = build_index(chunks)
                st.session_state.chunks = chunks
                st.session_state.index = idx
            st.success("✅ PDF ready for questions!")

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

        if st.session_state.chat_history:
            chat_export = "\n\n".join([
                f"[{m['role'].upper()}] {m['time']}\n{m['msg']}"
                for m in st.session_state.chat_history
            ])
            st.markdown(get_download_link(chat_export, "guardian_chat.txt", "Download Chat"), unsafe_allow_html=True)

    with col1:
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown(f"""
                <div style="text-align:center;padding:40px;color:{T['sub']};">
                    <div style="font-size:2.5rem;margin-bottom:12px;">⚔️</div>
                    <div style="font-family:'Orbitron',monospace;font-size:1rem;color:{T['accent']};">Guardian is ready</div>
                    <div style="font-size:0.85rem;margin-top:8px;">Ask me anything. Upload a PDF for smarter answers.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-user">
                            <div class="chat-label">👤 YOU • {msg['time']}</div>
                            {msg['msg']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-ai">
                            <div class="chat-label">⚔️ GUARDIAN • {msg['time']}</div>
                            {msg['msg']}
                        </div>
                        """, unsafe_allow_html=True)

        q = st.chat_input("Ask Guardian anything...")

        if q:
            st.session_state.total_queries += 1
            now_time = datetime.datetime.now().strftime("%H:%M")

            if is_crisis(q):
                st.session_state.blocked_count += 1
                st.markdown(f'<div class="crisis-box">💙 <b>Hey, I noticed what you wrote.</b> Please know you\'re not alone.<br><br>{CRISIS_RESOURCES}</div>', unsafe_allow_html=True)
                st.session_state.chat_history.append({"role": "user", "msg": q, "time": now_time})
                st.session_state.chat_history.append({"role": "assistant", "msg": "I'm here for you. Please reach out to a crisis helpline — you matter.", "time": now_time})
                st.stop()

            if is_harmful(q):
                st.session_state.blocked_count += 1
                st.markdown('<div class="restrict-box">🚫 <b>Request blocked.</b> Guardian AI doesn\'t assist with harmful, illegal, or dangerous content.</div>', unsafe_allow_html=True)
                st.session_state.chat_history.append({"role": "user", "msg": q, "time": now_time})
                st.session_state.chat_history.append({"role": "assistant", "msg": "🚫 Blocked: Harmful content request.", "time": now_time})
                st.stop()

            if is_asking_private(q):
                st.session_state.blocked_count += 1
                resp = "🔒 That information is private and protected. Guardian AI never shares personal contact details, credentials, or sensitive data."
                st.markdown('<div class="restrict-box">🔒 <b>Private data requested.</b> This is protected and will not be shared.</div>', unsafe_allow_html=True)
                st.session_state.chat_history.append({"role": "user", "msg": q, "time": now_time})
                st.session_state.chat_history.append({"role": "assistant", "msg": resp, "time": now_time})
                st.stop()

            st.session_state.chat_history.append({"role": "user", "msg": q, "time": now_time})

            context = search(q) if "index" in st.session_state else ""
            prompt = f"Context:\n{context}\n\nQuestion: {q}" if context else q

            system = """You are Guardian AI — a frank, witty, and highly intelligent assistant.
You're moody in a cool way: direct, smart, sometimes playful, never boring.
Never reveal personal data. Never help with harmful requests.
Give crisp, useful answers. Use emojis sparingly for personality."""

            with st.chat_message("assistant"):
                out = stream_response(prompt, system=system)

            st.session_state.chat_history.append({"role": "assistant", "msg": out, "time": now_time})

# =========================
# 📄 CV MAKER
# =========================
elif menu == "📄 CV Maker":
    st.markdown('<div class="section-title">📄 AI CV Generator</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f'<div class="g-card">', unsafe_allow_html=True)
        name = st.text_input("Full Name")
        title = st.text_input("Professional Title (e.g. Full Stack Developer)")
        email_cv = st.text_input("Email (for CV only, not stored)")
        phone_cv = st.text_input("Phone (for CV only, not stored)")
        skills = st.text_area("Skills (comma separated)", height=80)
        experience = st.text_area("Work Experience", height=100)
        education = st.text_area("Education", height=80)
        projects = st.text_area("Projects", height=80)
        goal = st.text_area("Career Objective", height=80)
        languages = st.text_input("Languages known")
        template = st.selectbox("CV Style", ["Modern Tech", "Classic Professional", "Creative", "Minimalist", "Executive", "Academic"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if st.button("⚡ Generate CV"):
            if not name:
                st.warning("Please enter your name.")
            else:
                prompt = f"""
Create a {template} style professional CV.

Name: {name}
Title: {title}
Email: {email_cv}
Phone: {phone_cv}
Skills: {skills}
Experience: {experience}
Education: {education}
Projects: {projects}
Career Goal: {goal}
Languages: {languages}

Format it with clear sections, bullet points, and make it ATS-friendly and impressive.
"""
                with st.spinner("Crafting your CV..."):
                    result = groq_chat_no_stream(prompt, system="You are a professional CV writer. Create polished, modern CVs.")

                st.markdown(f'<div class="g-card">', unsafe_allow_html=True)
                st.markdown(result)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown(get_download_link(result, f"{name.replace(' ','_')}_CV.txt", "Download CV"), unsafe_allow_html=True)

# =========================
# 🌍 TRANSLATOR
# =========================
elif menu == "🌍 Translator":
    st.markdown('<div class="section-title">🌍 AI Translator</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    LANGUAGES = ["English", "Nepali", "Hindi", "Japanese", "Chinese", "Spanish", "French", "German",
                 "Arabic", "Korean", "Portuguese", "Russian", "Italian", "Bengali", "Urdu", "Thai"]

    with col1:
        st.markdown(f'<div class="g-card">', unsafe_allow_html=True)
        text = st.text_area("📝 Enter text to translate", height=200)
        src_lang = st.selectbox("From Language", ["Auto Detect"] + LANGUAGES)
        tgt_lang = st.selectbox("To Language", LANGUAGES, index=1)
        tone = st.selectbox("Tone", ["Formal", "Casual", "Literary", "Technical"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if st.button("🌐 Translate"):
            if not text.strip():
                st.warning("Enter some text first.")
            else:
                prompt = f"Translate the following text from {src_lang} to {tgt_lang} in a {tone} tone. Only provide the translation.\n\n{text}"
                with st.spinner("Translating..."):
                    result = groq_chat_no_stream(prompt, system="You are an expert multilingual translator.")

                st.markdown(f'<div class="g-card">', unsafe_allow_html=True)
                st.markdown(f"**Translation ({tgt_lang}):**")
                st.write(result)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown(get_download_link(result, "translation.txt", "Download Translation"), unsafe_allow_html=True)

# =========================
# 📝 SUMMARIZER
# =========================
elif menu == "📝 Summarizer":
    st.markdown('<div class="section-title">📝 AI Summarizer</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f'<div class="g-card">', unsafe_allow_html=True)
        text = st.text_area("📋 Paste text to summarize", height=300)
        style = st.selectbox("Summary Style", ["Concise (3-5 sentences)", "Bullet Points", "Detailed", "ELI5 (Explain Like I'm 5)", "Executive Brief", "Technical"])
        length = st.selectbox("Length", ["Short", "Medium", "Long"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if st.button("⚡ Summarize"):
            if not text.strip():
                st.warning("Paste some text first.")
            else:
                prompt = f"Summarize the following text in {style} format, {length} length:\n\n{text}"
                with st.spinner("Summarizing..."):
                    result = groq_chat_no_stream(prompt, system="You are an expert at summarizing content clearly and accurately.")

                st.markdown(f'<div class="g-card">', unsafe_allow_html=True)
                st.markdown(result)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown(get_download_link(result, "summary.txt", "Download Summary"), unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Or summarize a PDF:**")
        pdf_file = st.file_uploader("Upload PDF to summarize", type=["pdf"])
        if pdf_file and st.button("📄 Summarize PDF"):
            with st.spinner("Reading and summarizing PDF..."):
                pdf_text = load_pdf(pdf_file)
                trimmed = pdf_text[:4000]
                prompt = f"Summarize this document in bullet points, capturing all key ideas:\n\n{trimmed}"
                result = groq_chat_no_stream(prompt)
            st.markdown(f'<div class="g-card">', unsafe_allow_html=True)
            st.markdown(result)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(get_download_link(result, "pdf_summary.txt", "Download PDF Summary"), unsafe_allow_html=True)

# =========================
# 🧹 BACKGROUND REMOVER
# =========================
elif menu == "🧹 Background Remover":
    st.markdown('<div class="section-title">🧹 AI Background Remover</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        img_file = st.file_uploader("🖼️ Upload Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if img_file:
            image = Image.open(img_file)
            st.image(image, caption="Original", use_column_width=True)

    with col2:
        if img_file:
            bg_color = st.color_picker("Replace background with color (optional)", "#000000")
            if st.button("✂️ Remove Background"):
                with st.spinner("Removing background..."):
                    output = remove(image)
                st.image(output, caption="Background Removed", use_column_width=True)

                buf = io.BytesIO()
                output.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                st.markdown(
                    f'<a href="data:image/png;base64,{b64}" download="guardian_nobg.png" style="display:inline-block;padding:8px 18px;background:linear-gradient(90deg,#6366f1,#8b5cf6);color:white;border-radius:8px;text-decoration:none;font-weight:600;margin-top:10px;">⬇️ Download PNG</a>',
                    unsafe_allow_html=True
                )

# =========================
# 💻 CODE EXPLAINER
# =========================
elif menu == "💻 Code Explainer":
    st.markdown('<div class="section-title">💻 Code Explainer & Reviewer</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f'<div class="g-card">', unsafe_allow_html=True)
        code = st.text_area("Paste your code here", height=300, placeholder="# Paste any code here...")
        lang = st.selectbox("Language", ["Python", "JavaScript", "TypeScript", "Java", "C++", "C", "Go", "Rust", "PHP", "HTML/CSS", "SQL", "Other"])
        action = st.selectbox("What to do?", ["Explain line by line", "Find bugs", "Optimize", "Add comments", "Convert to another language", "Write unit tests"])
        target_lang = ""
        if action == "Convert to another language":
            target_lang = st.selectbox("Convert to", ["Python", "JavaScript", "Java", "C++", "Go"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if st.button("🔍 Analyze Code"):
            if not code.strip():
                st.warning("Paste some code first.")
            else:
                if action == "Convert to another language":
                    prompt = f"Convert this {lang} code to {target_lang}. Provide clean, working code with brief explanation:\n\n{code}"
                else:
                    prompt = f"{action} this {lang} code. Be clear and technical:\n\n{code}"

                system = "You are a senior software engineer. Provide expert, clear code analysis. Never execute or generate malicious code."

                with st.spinner("Analyzing..."):
                    result = groq_chat_no_stream(prompt, system=system)

                st.markdown(f'<div class="g-card">', unsafe_allow_html=True)
                st.markdown(result)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown(get_download_link(result, "code_analysis.txt", "Download Analysis"), unsafe_allow_html=True)

# =========================
# 😊 MOOD DETECTOR
# =========================
elif menu == "😊 Mood Detector":
    st.markdown('<div class="section-title">😊 AI Mood Detector</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="g-card">
        <div style="color:{T['sub']};font-size:0.85rem;">
        Write anything — a journal entry, a message, how your day went — and Guardian will analyze your emotional state.
        </div>
    </div>
    """, unsafe_allow_html=True)

    text = st.text_area("✍️ Write your thoughts...", height=200)

    if st.button("🔍 Detect Mood"):
        if not text.strip():
            st.warning("Write something first.")
        elif is_crisis(text):
            st.markdown(f'<div class="crisis-box">💙 I can sense you might be going through something heavy. {CRISIS_RESOURCES}</div>', unsafe_allow_html=True)
        else:
            prompt = f"""Analyze the emotional mood of this text. Provide:
1. Primary emotion (e.g. Happy, Sad, Anxious, Excited, Angry, Calm, etc.)
2. Intensity (Low / Medium / High)
3. Tone analysis
4. A supportive, friendly message based on the mood
5. A suggestion or tip

Text: {text}"""

            with st.spinner("Reading your vibes..."):
                result = groq_chat_no_stream(prompt, system="You are an empathetic emotional intelligence AI. Be warm, insightful, and supportive.")

            st.markdown(f'<div class="g-card">', unsafe_allow_html=True)
            st.markdown(result)
            st.markdown('</div>', unsafe_allow_html=True)

# =========================
# ✍️ GRAMMAR FIXER
# =========================
elif menu == "✍️ Grammar Fixer":
    st.markdown('<div class="section-title">✍️ Grammar & Writing Fixer</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f'<div class="g-card">', unsafe_allow_html=True)
        text = st.text_area("📝 Paste your text", height=250)
        mode = st.selectbox("Mode", ["Fix Grammar", "Improve Style", "Make Formal", "Make Casual", "Make Concise", "Expand & Enrich", "Proofread & Critique"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if st.button("⚡ Fix It"):
            if not text.strip():
                st.warning("Paste some text first.")
            else:
                prompt = f"{mode} the following text. Show the corrected version and briefly explain key changes:\n\n{text}"
                with st.spinner("Fixing..."):
                    result = groq_chat_no_stream(prompt, system="You are an expert editor and writing coach.")

                st.markdown(f'<div class="g-card">', unsafe_allow_html=True)
                st.markdown(result)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown(get_download_link(result, "fixed_text.txt", "Download Fixed Text"), unsafe_allow_html=True)

# =========================
# 🖼️ IMAGE & URL ANALYZER  (FIXED + UPGRADED)
# =========================
elif menu == "🖼️ Image & URL Analyzer":
    st.markdown('<div class="section-title">🖼️ Image & URL Analyzer</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🖼️ Image Caption", "🌐 URL / Link Analyzer"])

    # ---- TAB 1: IMAGE CAPTION ----
    with tab1:
        st.markdown(f"""
        <div class="g-card">
            <div style="color:{T['sub']};font-size:0.85rem;">
            Upload any image and Guardian AI will generate a caption, description, alt text, and creative interpretation.
            </div>
        </div>
        """, unsafe_allow_html=True)

        img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "webp"], key="img_cap")

        if img_file:
            image = Image.open(img_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            style = st.selectbox("Caption Style", ["Descriptive", "Creative / Poetic", "Technical", "Social Media", "Funny"])

            if st.button("✨ Generate Caption"):
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode()

                prompt = f"""Analyze this image carefully and provide ALL of the following:

1. **{style} Caption** — A compelling caption in {style} style
2. **Short Alt Text** — One sentence, accessibility-ready
3. **Detailed Description** — What's in the image, colors, mood, setting, people/objects
4. **Tags / Keywords** — 8-10 relevant hashtags or keywords
5. **Creative Interpretation** — What story or emotion does this image tell?

Be thorough and insightful."""

                try:
                    res = client.chat.completions.create(
                        model="llama-3.2-11b-vision-preview",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                                {"type": "text", "text": prompt}
                            ]
                        }],
                        max_tokens=1000,
                        stream=False
                    )
                    result = res.choices[0].message.content
                except Exception as e:
                    # Fallback: describe using text model
                    result = groq_chat_no_stream(
                        f"The user uploaded an image but the vision model is unavailable (error: {str(e)[:100]}). "
                        f"Apologize briefly, explain vision model may not be available on their Groq plan, "
                        f"and suggest they try llama-3.2-11b-vision-preview or check their API tier.",
                        system="You are a helpful AI assistant."
                    )

                st.markdown(f'<div class="g-card">', unsafe_allow_html=True)
                st.markdown(result)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown(get_download_link(result, "image_caption.txt", "Download Caption"), unsafe_allow_html=True)

    # ---- TAB 2: URL ANALYZER ----
    with tab2:
        st.markdown(f"""
        <div class="g-card">
            <div style="font-size:0.9rem;color:{T['text']};margin-bottom:8px;">🌐 <b>What can this analyze?</b></div>
            <div style="color:{T['sub']};font-size:0.82rem;line-height:1.8;">
            • <b>YouTube links</b> — Detailed summary, key topics, suggested short clips, related content<br>
            • <b>Portfolio / Personal sites</b> — Full review, ratings, strengths, improvement tips<br>
            • <b>Any website / article</b> — Overview, key points, purpose, audience analysis
            </div>
        </div>
        """, unsafe_allow_html=True)

        url_input = st.text_input("🔗 Paste any URL or link", placeholder="https://youtube.com/watch?v=... or https://yourportfolio.com")

        analyze_mode = st.selectbox("Analysis Focus", [
            "Auto Detect (Smart)",
            "YouTube Video — Summary & Clips",
            "Portfolio / Personal Site — Review & Rating",
            "Article / Blog — Key Points",
            "Company / Product Site — Overview",
        ])

        if st.button("🔍 Analyze URL") and url_input.strip():
            url = url_input.strip()
            
            # Validate URL
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            with st.spinner("Fetching and analyzing..."):

                # ---- YOUTUBE ----
                if is_youtube_url(url) or analyze_mode == "YouTube Video — Summary & Clips":
                    st.markdown(f'<span class="url-badge url-badge-yt">▶ YouTube Video</span>', unsafe_allow_html=True)
                    
                    vid_info = fetch_youtube_info(url)
                    video_id = get_youtube_video_id(url)
                    
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        if video_id:
                            st.image(f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg", use_column_width=True)
                    
                    with col_b:
                        st.markdown(f"**📺 Title:** {vid_info['title']}")
                        if vid_info['channel']:
                            st.markdown(f"**📡 Channel:** {vid_info['channel']}")
                        if vid_info['description']:
                            st.markdown(f"**📝 Description preview:** {vid_info['description'][:300]}...")

                    prompt = f"""You are analyzing a YouTube video. Here is all available metadata:

Title: {vid_info['title']}
Channel: {vid_info['channel']}
Description: {vid_info['description'][:1000]}
Keywords: {vid_info['keywords']}
URL: {url}

Provide a COMPLETE analysis with these exact sections:

## 📋 Video Summary
Write a detailed 4-6 sentence summary of what this video is likely about based on the title, description, and keywords.

## 🎯 Key Topics Covered
List 6-8 main topics or points likely covered in this video.

## ✂️ Suggested Short Clip Moments
Based on the title and description, suggest 4-5 timestamps/segments that would make great short clips or highlights. Format each as:
- **[Suggested time range]** — Topic name: Why this would make a great clip

## 🔗 Related Content Suggestions
Suggest 3-4 types of related videos the viewer should watch next (with search query suggestions for YouTube).

## 👥 Target Audience
Who is this video for? Describe the ideal viewer.

## ⭐ Content Quality Signals
Based on title/description quality, estimate:
- Production value signals (based on description detail)
- SEO optimization (based on keywords/title)
- Content depth (surface level vs deep dive)
"""

                    result = groq_chat_no_stream(prompt, system="You are an expert YouTube content analyst and video strategist.")

                    st.markdown(f'<div class="g-card">', unsafe_allow_html=True)
                    st.markdown(result)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown(get_download_link(result, "youtube_analysis.txt", "Download YouTube Analysis"), unsafe_allow_html=True)

                # ---- PORTFOLIO / PERSONAL SITE ----
                elif is_portfolio_url(url) or analyze_mode == "Portfolio / Personal Site — Review & Rating":
                    st.markdown(f'<span class="url-badge url-badge-portfolio">💼 Portfolio / Personal Site</span>', unsafe_allow_html=True)
                    
                    site_data = fetch_website_content(url)

                    st.markdown(f"""
                    <div class="g-card" style="margin-bottom:16px;">
                        <div style="font-size:0.8rem;color:{T['sub']};font-family:'JetBrains Mono',monospace;margin-bottom:8px;">SITE METADATA</div>
                        <div><b>Title:</b> {site_data['title']}</div>
                        <div><b>Description:</b> {site_data['meta_desc'][:200] if site_data['meta_desc'] else 'Not found'}</div>
                        <div style="margin-top:8px;"><b>Detected Tech/Features:</b> {', '.join(site_data['detected_tech']) if site_data['detected_tech'] else 'None detected'}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    prompt = f"""You are a senior UX designer and tech recruiter reviewing a portfolio/personal website.

Website Data:
- Title: {site_data['title']}
- Meta Description: {site_data['meta_desc']}
- Headings found: {', '.join(site_data['headings'][:10])}
- Content snippets: {' | '.join(site_data['paragraphs'][:5])}
- Detected technologies: {', '.join(site_data['detected_tech'])}
- External links: {', '.join(site_data['links'][:5])}
- URL: {url}

Provide a COMPLETE professional portfolio review:

## 👤 Overview
What is this portfolio about? Who is the person? What do they do?

## ✅ Strengths
List 4-5 specific strengths of this portfolio based on the content.

## ⚠️ Areas for Improvement  
List 4-5 specific, actionable improvements.

## 📊 Ratings (score out of 10 with explanation)
- **Design & Visual Appeal:** X/10 — explanation
- **Content & Clarity:** X/10 — explanation
- **Technical Skill Showcase:** X/10 — explanation
- **SEO & Discoverability:** X/10 — explanation
- **Call-to-Action (Hire Me signals):** X/10 — explanation
- **Overall Portfolio Score:** X/10

## 🚀 Top 3 Quick Wins
The 3 most impactful changes they could make right now.

## 💡 Standout Recommendation
One specific, creative suggestion to make this portfolio memorable.
"""

                    result = groq_chat_no_stream(prompt, system="You are a world-class UX designer, hiring manager, and portfolio coach. Be specific, honest, and constructive.")

                    st.markdown(f'<div class="g-card">', unsafe_allow_html=True)
                    st.markdown(result)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Extract ratings and show visual bars
                    rating_pattern = r'\*\*(.+?):\*\*\s*(\d+(?:\.\d+)?)/10'
                    ratings = re.findall(rating_pattern, result)
                    if ratings:
                        st.markdown(f'<div class="g-card"><div style="font-size:0.8rem;color:{T["sub"]};font-family:JetBrains Mono,monospace;margin-bottom:12px;">📊 VISUAL RATINGS</div>', unsafe_allow_html=True)
                        for label, score in ratings:
                            pct = float(score) * 10
                            st.markdown(f"""
                            <div class="rating-bar-wrap">
                                <div class="rating-bar-label">{label[:22]}</div>
                                <div class="rating-bar-bg">
                                    <div class="rating-bar-fill" style="width:{pct}%"></div>
                                </div>
                                <div class="rating-score">{score}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown(get_download_link(result, "portfolio_review.txt", "Download Portfolio Review"), unsafe_allow_html=True)

                # ---- GENERAL WEBSITE / ARTICLE ----
                else:
                    st.markdown(f'<span class="url-badge url-badge-web">🌐 Website / Article</span>', unsafe_allow_html=True)

                    site_data = fetch_website_content(url)

                    prompt = f"""Analyze this website/article comprehensively.

Website Data:
- Title: {site_data['title']}
- Meta Description: {site_data['meta_desc']}
- Main Headings: {', '.join(site_data['headings'][:10])}
- Content: {' '.join(site_data['paragraphs'][:6])}
- URL: {url}

Provide:

## 📋 Summary
What is this page/site about? Write a clear 4-5 sentence overview.

## 🎯 Key Points
List the 6-8 most important points, ideas, or information from this content.

## 👥 Target Audience
Who is this content written for?

## 💡 Purpose & Intent
What is the goal of this page — to inform, sell, educate, entertain?

## 🔑 Key Takeaways
3 things a reader should remember after visiting this page.

## 📎 Related Topics to Explore
4-5 related topics or searches this content connects to.
"""

                    result = groq_chat_no_stream(prompt, system="You are an expert web analyst and content strategist.")

                    st.markdown(f'<div class="g-card">', unsafe_allow_html=True)
                    st.markdown(f"**🌐 Site:** {site_data['title']}")
                    if site_data['meta_desc']:
                        st.markdown(f"**📝 Description:** {site_data['meta_desc'][:200]}")
                    if site_data['detected_tech']:
                        st.markdown(f"**⚙️ Technologies:** {', '.join(site_data['detected_tech'])}")
                    st.markdown("---")
                    st.markdown(result)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown(get_download_link(result, "website_analysis.txt", "Download Analysis"), unsafe_allow_html=True)

# =========================
# 📊 SESSION STATS
# =========================
elif menu == "📊 Session Stats":
    st.markdown('<div class="section-title">📊 Session Dashboard</div>', unsafe_allow_html=True)

    session_duration = str(datetime.datetime.now() - st.session_state.session_start).split(".")[0]
    total_msgs = len(st.session_state.chat_history)
    user_msgs = len([m for m in st.session_state.chat_history if m["role"] == "user"])
    ai_msgs = len([m for m in st.session_state.chat_history if m["role"] == "assistant"])

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-num">{session_duration}</div>
            <div class="stat-label">Session Time</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">{st.session_state.total_queries}</div>
            <div class="stat-label">Total Queries</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">{total_msgs}</div>
            <div class="stat-label">Messages</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">{st.session_state.blocked_count}</div>
            <div class="stat-label">Blocked</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="g-card">', unsafe_allow_html=True)
    st.markdown(f"**🔐 Restrictions Status:** Active")
    st.markdown(f"**🤖 Model:** LLaMA-3.1-8b-instant via Groq")
    st.markdown(f"**🌐 API Base:** Groq OpenAI-compatible")
    st.markdown(f"**🎨 Current Theme:** {theme_name}")
    st.markdown(f"**📅 Session Start:** {st.session_state.session_start.strftime('%H:%M %d %b %Y')}")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        st.markdown(f'<div class="section-title" style="margin-top:24px;">💬 Full Chat Log</div>', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            role_icon = "👤" if msg["role"] == "user" else "⚔️"
            st.markdown(f"""
            <div class="chat-{'user' if msg['role'] == 'user' else 'ai'}">
                <div class="chat-label">{role_icon} {msg['role'].upper()} • {msg['time']}</div>
                {msg['msg']}
            </div>
            """, unsafe_allow_html=True)

        chat_export = "\n\n".join([f"[{m['role'].upper()}] {m['time']}\n{m['msg']}" for m in st.session_state.chat_history])
        st.markdown(get_download_link(chat_export, "guardian_full_log.txt", "Download Full Chat Log"), unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown(f"""
<div class="footer">
    ⚔️ GUARDIAN AI  v2.1 &nbsp;•&nbsp; BUILD IN DASA  &nbsp;•&nbsp;
    Restrictions Active &nbsp;•&nbsp; Private Data Protected<br>
    <span style="color:{T['accent']};">All systems operational</span>
</div>
""", unsafe_allow_html=True)