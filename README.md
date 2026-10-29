# TopicTrace 📚

> AI-Driven Automated Study Notes Generator built with Streamlit, Google Gemini, and Python.

TopicTrace extracts topics from your **Course Syllabus**, cross-references and maps them against your **Textbook PDF**, and generates comprehensive, hallucination-free study notes. If a syllabus topic is missing from the textbook, the notes will explicitly flag it.

---

## Features

- **PDF Ingestion:** Extracts clean, page-indexed text from both Syllabus and Textbook PDFs using `pypdf`.
- **Hierarchical Syllabus Parsing:** Uses Google Gemini (`gemini-2.5-flash` / `gemini-2.0-flash`) to parse syllabus content into structured units, topics, and subtopics (JSON).
- **Strict Anti-Hallucination Grounding:** Prompts ensure study notes are derived strictly from textbook content. Missing topics are prominently marked with `⚠️ Missing from Textbook`.
- **Interactive Human Review:** Live side-by-side Markdown editor with formatted preview.
- **Mandatory Warning Banner:** *"AI-generated content. Please verify with your textbook."*
- **Multi-Format Export:** One-click download as clean Markdown (`.md`) or formatted PDF (`.pdf`) using `fpdf2`.
- **100% Free API Tier:** Designed to run with zero cost using Google AI Studio's free tier.

---

## Tech Stack & Architecture

```
topictrace/
├── app.py             # Main Streamlit web application & user workflow
├── extractor.py       # PDF parsing & text extraction using pypdf
├── llm_service.py     # Gemini client, syllabus parsing & grounded note generation
├── exporter.py        # PDF & Markdown generation with fpdf2
├── requirements.txt   # Python package dependencies
├── .env.example       # Example environment configuration for GEMINI_API_KEY
├── .gitignore         # Git ignore rules for venv, cache, keys
└── instructions.md    # Original MVP specification
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 2. Setup Virtual Environment & Install Dependencies

```bash
# Clone the repository (if applicable)
cd topictrace

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. (Optional) Configure API Key in `.env`

Copy `.env.example` to `.env` and insert your Gemini API Key:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
GEMINI_API_KEY=AIzaSy...
```
*(You can also input your API key directly inside the Streamlit sidebar at runtime).*

### 4. Run the Streamlit Application

```bash
.venv/bin/streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Workflow Guide

1. **Upload:** In the left sidebar, upload your **Syllabus PDF** and **Textbook PDF**.
2. **Parse Topics:** Click **"🚀 Parse Syllabus Topics"** to generate the interactive topic tree.
3. **Select & Generate:** Check the topics you need notes for, then click **"📝 Generate Study Notes"**.
4. **Review & Edit:** Review the generated notes in the live Markdown editor. Add custom explanations, adjust formatting, or remove sections.
5. **Export:** Click **"📥 Download as Markdown"** or **"📄 Download as PDF"** to export your final notes.
