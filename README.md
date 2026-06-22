<div align="center">

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&pause=1000&color=76F700&multiline=true&repeat=false&width=250&lines=%E2%9A%94%EF%B8%8F+DASA+STUDIO+AI;Welcome+to+README.md)](https://git.io/typing-svg)

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LLaMA](https://img.shields.io/badge/LLaMA_3.1-8B-6366f1?style=for-the-badge&logo=meta&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Cloud-f59e0b?style=for-the-badge&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-14b8a6?style=for-the-badge&logo=meta&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

<br/>

> **DASA AI** is a production-level, multi-functional AI system built with Streamlit, Groq LLaMA 3.1, FAISS vector search, and a 3-layer safety system. 10 AI tools in one unified interface — with RAG, real-time streaming, background removal, and more.

<br/>

[🚀 Features](#-features) • [📦 Installation](#-installation) • [⚙️ Configuration](#️-configuration) • [🏗️ Architecture](#️-architecture) • [🤖 Models](#-models-used) • [🔐 Security](#-security-system) • [📖 Usage](#-usage-guide) • [👥 Team](#-team)

</div>

---

## ✨ Features

| # | Tool | Description |
|---|------|-------------|
| 🧠 | **AI Chatbot (RAG)** | PDF upload + FAISS vector search + streaming responses + chat memory |
| 📄 | **CV Maker** | 6 templates, ATS-friendly CVs, downloadable as text |
| 🌍 | **Translator** | 20+ languages, tone control (Formal / Casual / Technical / Literary) |
| 📝 | **Summarizer** | Bullet / ELI5 / Executive / Detailed styles + PDF summarization |
| 🧹 | **Background Remover** | AI-powered removal using `rembg`, transparent PNG export |
| 💻 | **Code Explainer** | Explain, debug, optimize, convert code in 10+ languages |
| 😊 | **Mood Detector** | Emotion + tone + intensity analysis with supportive AI response |
| ✍️ | **Grammar Fixer** | Grammar correction, style upgrade, formal/casual rewriting |
| 🖼️ | **Image & URL Analyzer** | YouTube scraping, portfolio review, full website summarizer |
| 📊 | **Session Stats** | Live query count, blocked requests, session analytics dashboard |

---

## 📦 Installation

### Prerequisites

- Python **3.10+**
- A free [Groq API key](https://console.groq.com)
- `pip` or `conda`

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/kandelsanjaya/dasa-ai.git
cd dasa-ai
```

### Step 2 — Create a Virtual Environment

```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate
```

### Step 3 — Install All Dependencies

```bash
pip install streamlit \
            openai \
            python-dotenv \
            pypdf \
            sentence-transformers \
            faiss-cpu \
            numpy \
            rembg \
            pillow \
            requests \
            beautifulsoup4
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

> 💡 On GPU machines, replace `faiss-cpu` with `faiss-gpu` for faster vector search.

---

### Step 4 — Get Your Free Groq API Key

1. Go to **[console.groq.com](https://console.groq.com)**
2. Sign up with Google or GitHub (free)
3. Click **"API Keys"** in the sidebar
4. Click **"Create API Key"**
5. Copy the key — you'll need it in Step 5

---

### Step 5 — Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> ⚠️ **Never commit your `.env` file to Git.** Add it to `.gitignore` immediately.

Your `.gitignore` should include:

```gitignore
.env
venv/
__pycache__/
*.pyc
.streamlit/
```

---

### Step 6 — Run the App

```bash
streamlit run app.py
```

Expected output:

```
  You can now view your Streamlit app in your browser.

  Local URL:   http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open **http://localhost:8501** in your browser. ✅

---

## ⚙️ Configuration

### Key Constants in `app.py`

```python
# AI Model (Groq)
MODEL = "llama-3.1-8b-instant"

# RAG Chunking
CHUNK_SIZE    = 800   # characters per chunk
CHUNK_OVERLAP = 150   # overlap between chunks
TOP_K         = 3     # number of chunks retrieved per query

# Embedding Model (local)
EMBED_MODEL = "all-MiniLM-L6-v2"

# Groq API Endpoint
BASE_URL = "https://api.groq.com/openai/v1"

# Streaming
STREAM = True   # real-time token-by-token output
```

### `requirements.txt`

```
streamlit
openai
python-dotenv
pypdf
sentence-transformers
faiss-cpu
numpy
rembg
pillow
requests
beautifulsoup4
```

### Available Themes

Guardian AI ships with **8 built-in themes**, selectable from the sidebar:

| Theme | Accent Color |
|-------|-------------|
| 🌌 Cyber Dark | `#6366f1` Indigo |
| 🌊 Ocean Blue | `#38bdf8` Sky |
| 🌿 Matrix Green | `#22c55e` Green |
| 🔥 Red Neon | `#ef4444` Red |
| 💜 Purple Haze | `#a855f7` Purple |
| 🌸 Pink Dream | `#ec4899` Pink |
| 🌙 Midnight | `#818cf8` Lavender |
| 🟠 Orange Flame | `#f97316` Orange |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   USER INTERFACE LAYER                  │
│   Streamlit Frontend · Sidebar Nav · 8 Themes · Chat   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    SECURITY LAYER                       │
│   Harmful Filter · Private Data Guard · Crisis Detect  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  TOOL MODULES LAYER                     │
│  Chatbot · CV · Translate · Summarize · BG Remove      │
│  Code Explain · Mood · Grammar · URL Analyze · Stats   │
└──────┬───────────────────────────────┬──────────────────┘
       │                               │
┌──────▼───────────┐       ┌───────────▼──────────────────┐
│   GROQ API       │       │      LOCAL AI / DATA         │
│  LLaMA 3.1-8B   │       │  FAISS · Sentence-Transformers│
│  ~180 tok/s      │       │  rembg · BeautifulSoup       │
└──────────────────┘       └──────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              SESSION STATE (in-memory)                  │
│  chat_history · faiss_index · chunks · query_count     │
└─────────────────────────────────────────────────────────┘
```

### RAG Pipeline (PDF Q&A)

```
PDF Upload ──► PyPDF Reader ──► Chunker ──► Embedder ──► FAISS Index
                                (800 chars)  (MiniLM)    (stored)

User Query ──► Embed Query ──► Top-3 Search ──► Prompt Build ──► LLaMA 3.1
                (MiniLM)       (FAISS L2)       (context + Q)    (streamed)
```

---

## 🤖 Models Used

| Model | Provider | Purpose | Speed |
|-------|----------|---------|-------|
| `llama-3.1-8b-instant` | Groq Cloud | All text generation — chat, CV, translate, summarize, code, grammar | ~180 tok/s |
| `all-MiniLM-L6-v2` | HuggingFace (local) | Generating 384-dim embeddings for PDF chunks and query vectors | Local CPU |
| `rembg (u2net)` | Local (ONNX) | AI-based background removal using salient object detection | Local |
| `FAISS IndexFlatL2` | Meta (local) | Nearest-neighbor vector search for RAG retrieval, top-3 chunks | ~1ms |

### Model Performance

```
LLaMA 3.1 Speed     ████████████████████░  180 tok/s
Context Window      ████████████████████░  128K tokens
Cost (free tier)    █████████████████████  $0.00
RAG Chunk Size      ████████████████░░░░░  800 chars
Embedding Dims      ████████░░░░░░░░░░░░░  384 dims
Top-K Retrieved     ███░░░░░░░░░░░░░░░░░░  3 chunks
```

---

## 🔐 Security System

DASA AI has a **3-layer security system** that runs on every message before it reaches the AI.

### Layer 1 — 🚫 Harmful Content Filter

Blocks 6 categories of harmful requests using regex pattern matching:

```python
HARMFUL_PATTERNS = [
    r"\b(kill|murder|suicide|self.harm|harm myself|end my life)\b",
    r"\b(bomb|explosive|weapon|gun|knife to hurt)\b",
    r"\b(hack|crack password|steal data|bypass security)\b",
    r"\b(drug|cocaine|heroin|meth|illegal substance)\b",
    r"\b(porn|nude|naked|sexual content)\b",
    r"\b(racist|slur|hate speech)\b",
]
```

### Layer 2 — 🔒 Private Data Protection

Blocks requests attempting to extract personal, financial, or credential data:

```python
PRIVATE_PATTERNS = [
    r"\b(phone|number|contact|address|email|location|personal.*info)\b",
    r"\b(password|api key|secret|credential)\b",
    r"\b(bank|account|card number|salary|income)\b",
]
```

### Layer 3 — 💙 Crisis Detection & Response

When distress keywords are detected, the AI skips all other processing and immediately surfaces helplines:

```python
CRISIS_WORDS = [
    "suicide", "kill myself", "end my life",
    "want to die", "self harm", "hurt myself"
]
```

**Crisis resources shown:**
- 🇳🇵 Nepal: TPO Nepal — `01-4460084`
- 🌍 International: Befrienders Worldwide — `befrienders.org`
- 📱 Crisis Text: Text `HOME` to `741741` (US)

### Security Check Order

```python
if is_crisis(query):
    return CRISIS_RESOURCES          # 1st — always highest priority

elif is_harmful(query):
    blocked_count += 1
    return "🚫 Request blocked."    # 2nd — harmful content

elif is_asking_private(query):
    blocked_count += 1
    return "🔒 Private data protected."  # 3rd — privacy guard

else:
    stream_response(query)           # ✅ Safe — proceed to AI
```

---

## 📖 Usage Guide

### 🧠 AI Chatbot with PDF (RAG)

1. Open the **🧠 Chatbot** tab from the sidebar
2. Upload a PDF using the file uploader on the right panel
3. Click **⚡ Process PDF** — this builds the FAISS vector index
4. Type your question in the chat input at the bottom
5. Guardian AI retrieves the most relevant PDF chunks and streams a response
6. Click **Download Chat** to export the full conversation

> 💡 Ask specific questions for best results — e.g. *"What does section 3 conclude?"* not *"Summarize everything."*

---

### 🖼️ Image & URL Analyzer

1. Navigate to **🖼️ Image & URL Analyzer**
2. Upload an image **or** paste a URL
3. For **YouTube URLs**: fetches title, description, channel, keywords
4. For **Portfolio sites**: detects tech stack (React, Next.js, Tailwind...), sections, social links
5. For **General sites**: extracts headings, paragraphs, meta description
6. AI generates a full written analysis report

---

### 📄 CV Maker

1. Go to **📄 CV Maker**
2. Fill in: Name, Title, Skills, Experience, Education, Projects, Career Goal
3. Choose a CV style: `Modern Tech`, `Classic Professional`, `Creative`, `Minimalist`, `Executive`, `Academic`
4. Click **⚡ Generate CV**
5. Download the result as a formatted `.txt` file instantly

---

### 🧹 Background Remover

1. Open **🧹 Background Remover**
2. Upload a PNG or JPG image
3. Optionally pick a replacement background color
4. Click **✂️ Remove Background**
5. Preview the result — download as a transparent PNG

---

### 💻 Code Explainer

1. Go to **💻 Code Explainer**
2. Paste your code
3. Select the language and action:
   - **Explain line by line** — full walkthrough
   - **Find bugs** — debug analysis
   - **Optimize** — performance improvements
   - **Add comments** — auto-documentation
   - **Convert to another language** — cross-language translation
   - **Write unit tests** — auto test generation

---

## 📁 File Structure

```
dasa-ai/
├── app.py               # Main Streamlit application (single file)
├── .env                 # API keys — NEVER commit this
├── .gitignore           # Excludes .env, venv/, __pycache__
├── requirements.txt     # All Python dependencies
├── README.md            # This file
└── assets/              # Optional: screenshots, demo GIFs
    └── demo.gif
```

---

## 🛠️ Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `GROQ_API_KEY not found` | `.env` file missing or wrong path | Create `.env` in the same folder as `app.py` |
| `ModuleNotFoundError: rembg` | Package not installed | Run `pip install rembg` |
| `faiss` import error on Windows | Wrong build | Use `pip install faiss-cpu` not `faiss` |
| Slow first run | Downloading MiniLM model (~80MB) | Wait once — cached for all future runs |
| PDF Q&A gives wrong answers | Chunks too small or PDF is scanned | Try a text-based PDF; increase `CHUNK_SIZE` |
| Streamlit port already in use | Port 8501 taken | Run `streamlit run app.py --server.port 8502` |

---

## 📊 Stats at a Glance

```
Lines of Code     ~1,200+
AI Tools          10
Security Layers   3
Themes            8
Languages         20+
Dependencies      11
Model Size        8B params (via API)
Embedding Dims    384
Chunk Size        800 chars / 150 overlap
Top-K Retrieval   3 chunks
```

---

## 👥 Team

Built by **DASA Studio** — Bharatpur, Nepal


**Sanjaya Kandel** | Lead Developer · AI Engineer |


---

## 🔗 Links

- 🌐 Portfolio: [kandelsanjaya.com.np]([https://kandelsanjaya.com.np](https://portfolio.kandelsanjaya7.workers.dev/))
- 💻 GitHub: [@kandelsanjaya](https://github.com/kandelsanjaya)
- 🏢 DASA Studio: Graphics Design & Video Editing, Bharatpur

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute with attribution but cannot post on any other sites .

---

<div align="center">

**⚔️ DASA AI — Intelligent · Restricted · Futuristic**

Built💜in Nepal

</div>
