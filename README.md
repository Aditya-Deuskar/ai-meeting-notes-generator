# 🎙️ AI Meeting Notes Generator

Stop typing notes during meetings. Upload your recording and get back structured notes, action items, and an executive summary — powered by Whisper transcription and a 3-stage Llama pipeline, running entirely on your own machine.

---

## What It Does

Meetings happen everywhere — work standups, client calls, university lectures, interviews, team retrospectives. Nobody wants to be the person furiously typing while trying to stay present.

This tool solves that. Record your meeting, upload it, and walk away with:

- **Clean transcript** — filler words removed, grammar fixed
- **Action items** — every task, owner, and deadline extracted automatically
- **Executive summary** — key decisions, discussion points, and outcomes in one place
- **Downloadable notes** — export everything as a markdown file

All processing happens locally on your hardware. **No data ever leaves your machine.**

---

## 🧠 How It Works

Whisper handles transcription, then Llama runs three specialised tasks in sequence:

```
Audio Recording
      │
      ▼
Whisper (medium)       →  Transcribes speech to text
      │
      ▼
Llama 3.1 8B           →  Stage 1: Cleans up the raw transcript
      │
      ▼
Llama 3.1 8B           →  Stage 2: Extracts action items with owners and deadlines
      │
      ▼
Llama 3.1 8B           →  Stage 3: Generates executive summary
      │
      ▼
Structured Meeting Notes
```

---

## 💻 Requirements

- NVIDIA GPU with 8GB+ VRAM
- 16GB RAM
- CUDA 12.8+
- Windows 10/11 or Linux
- Python 3.10+

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Aditya-Deuskar/ai-meeting-notes-generator.git
cd ai-meeting-notes-generator
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install PyTorch with CUDA support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Ollama and pull Llama

Download from [ollama.com](https://ollama.com), then:

```bash
ollama pull llama3.1:8b
```

### 6. Install ffmpeg

```bash
# Windows
winget install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

---

## ▶️ Usage

Make sure Ollama is running in the background, then:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser, upload your recording, and click **Generate Meeting Notes**.

**Supported formats:** mp3, wav, m4a, ogg, flac

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Transcription | OpenAI Whisper |
| Language Model | Llama 3.1 8B via Ollama |
| UI | Streamlit |
| ML Framework | PyTorch (CUDA) |
| Language | Python 3.13 |

---

## 👤 Author

**Aditya Deuskar**  