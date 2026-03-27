import whisper
import ollama
import torch
import os

# ── Config ────────────────────────────────────────────────────────────────────

WHISPER_MODEL = "medium"       # Options: tiny, base, small, medium, large
LLAMA_MODEL   = "llama3.1:8b"  # Must match what you pulled with ollama pull

# ── Model 1: Whisper — Speech to Text ─────────────────────────────────────────

def transcribe_audio(audio_path: str, language: str = None) -> str:
    """Takes an audio file path, returns raw transcript text."""
    
    print("🎙️  Model 1: Transcribing audio with Whisper...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    model = whisper.load_model(WHISPER_MODEL, device=device)
    result = model.transcribe(audio_path, language=None if language == "Auto Detect" else language)
    
    transcript = result["text"]
    print(f"   ✅ Transcription done — {len(transcript)} characters")
    return transcript


# ── Model 2: Llama — Transcript Cleanup ───────────────────────────────────────

def clean_transcript(raw_transcript: str) -> str:
    """Takes messy raw transcript, returns cleaned readable version."""
    
    print("\n🧹  Model 2: Cleaning transcript with Llama...")
    
    prompt = f"""You are an expert meeting transcriptionist.

Clean up the following raw speech-to-text transcript:
- Fix grammar and punctuation
- Remove filler words (um, uh, like, you know)
- Fix obvious transcription errors
- Keep all the actual content and meaning intact
- Do NOT summarise or remove any information

Raw transcript:
{raw_transcript}

Cleaned transcript:"""

    response = ollama.chat(
        model=LLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    cleaned = response["message"]["content"]
    print(f"   ✅ Cleanup done — {len(cleaned)} characters")
    return cleaned


# ── Model 3: Llama — Action Item Extraction ───────────────────────────────────

def extract_action_items(cleaned_transcript: str) -> str:
    """Takes cleaned transcript, returns structured action items."""
    
    print("\n✅  Model 3: Extracting action items with Llama...")
    
    prompt = f"""You are an expert meeting analyst.

Extract all action items from the following meeting transcript.

Format each action item exactly like this:
- [ ] Action description | Owner: [person name or 'TBD'] | Due: [date or 'Not specified']

Rules:
- Only include concrete tasks and commitments
- If no clear owner is mentioned, write 'TBD'
- If no deadline is mentioned, write 'Not specified'
- Be specific and clear

Meeting transcript:
{cleaned_transcript}

Action items:"""

    response = ollama.chat(
        model=LLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    action_items = response["message"]["content"]
    print(f"   ✅ Action items extracted")
    return action_items


# ── Model 4: Llama — Executive Summary ────────────────────────────────────────

def generate_summary(cleaned_transcript: str, action_items: str) -> str:
    """Takes cleaned transcript + action items, returns executive summary."""
    
    print("\n📋  Model 4: Generating executive summary with Llama...")
    
    prompt = f"""You are an expert meeting summariser.

Write a concise executive summary of this meeting.

Structure your response exactly like this:

## Meeting Summary

**Key Discussion Points:**
- Point 1
- Point 2
- Point 3

**Decisions Made:**
- Decision 1
- Decision 2

**Next Steps:**
[Reference the action items below]

**Overall Outcome:**
[1-2 sentences on what was achieved]

Meeting transcript:
{cleaned_transcript}

Action items already extracted:
{action_items}

Executive summary:"""

    response = ollama.chat(
        model=LLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    summary = response["message"]["content"]
    print(f"   ✅ Summary generated")
    return summary


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(audio_path: str, language: str = None) -> dict:
    """
    Runs the full 4-model pipeline.
    Returns a dict with all outputs.
    """
    
    print("=" * 60)
    print("  AI MEETING NOTES GENERATOR")
    print("=" * 60)
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Run all 4 models in sequence
    raw_transcript  = transcribe_audio(audio_path)
    clean_text      = clean_transcript(raw_transcript)
    action_items    = extract_action_items(clean_text)
    summary         = generate_summary(clean_text, action_items)
    
    results = {
        "raw_transcript":  raw_transcript,
        "clean_transcript": clean_text,
        "action_items":    action_items,
        "summary":         summary
    }
    
    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETE")
    print("=" * 60)
    
    return results


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Change this to any audio file you have for testing
    test_audio = "test_audio.wav"
    
    results = run_pipeline(test_audio)
    
    print("\n── RAW TRANSCRIPT ──────────────────────────────────────")
    print(results["raw_transcript"])
    
    print("\n── CLEANED TRANSCRIPT ──────────────────────────────────")
    print(results["clean_transcript"])
    
    print("\n── ACTION ITEMS ────────────────────────────────────────")
    print(results["action_items"])
    
    print("\n── EXECUTIVE SUMMARY ───────────────────────────────────")
    print(results["summary"])