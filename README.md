# Lecture Notes AI

A browser app (Streamlit) that turns my Chinese classes into study notes.
I record each topic the teacher explains; the app transcribes the
Chinese audio (Whisper base), summarizes it into notes (Ollama qwen2.5),
and keeps every topic in a review library.

Runs 100% locally - no cloud, no API keys. My class audio never leaves my laptop.

## How it works
1. [Start] / [Stop] in the browser -> one WAV per topic (background thread)
2. Whisper base transcribes Mandarin -> Chinese text
3. qwen2.5 summarizes -> key points + likely exam focus
4. Every note saved to notes/ and shown in the study library

## Why per-topic recording
Whisper on CPU runs at ~0.5-1x real time, so a full 45-min lecture would
take ~30+ min to transcribe. Recording topic-by-topic means Topic 1 is
being transcribed while Topic 2 is being taught.

## Setup
1. `py -m pip install -r requirements.txt`
2. `ollama pull qwen2.5:3b`
3. `py -m streamlit run app.py` -> opens at http://localhost:8501

## Notes
- recordings/ and notes/ are gitignored: private class content
- Model choice: `base` over `tiny` because tiny mangles Chinese
- Summarizer reuses the proxy-proof session + robust answer extraction
  patterns from my voice-assistant project