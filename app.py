"""
Lecture Notes AI - Milestone 1: record class topics from the browser.

How the recording works (the one tricky concept of this milestone):

  Streamlit re-runs the whole script on every click, so we CANNOT just
  loop the microphone inside a button's if-block - the rerun would kill
  it. Instead, the recorder lives in a BACKGROUND THREAD:
    - click Start   -> spawn thread that keeps reading the mic into a list
    - click Stop    -> flip a flag, the thread sees it and exists,
                       we join the chunks and save one WAV file
  st.session_state keeps the Recorder object alive between reruns -
  that is Streamlit's "memory that survives clicks".

Usage in class: click Start when the teacher begins a topic,
click Stop when the topic ends. One file per topic.
"""

import os
import sys
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf
import streamlit as st

# 1. First: figure out where the project lives
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 2. Then: tell Python "our modules are in src/"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# 3. Only THEN: import our own modules (they need step 2 to work)
from transcriber import transcribe_chinese
from summarizer import summarize_chinese

# 4. Constants that use PROJECT_ROOT
SAMPLE_RATE = 16000
RECORDINGS = os.path.join(PROJECT_ROOT, "recordings")
NOTES = os.path.join(PROJECT_ROOT, "notes")


class Recorder:
    """A microphone recorder that runs in a background thread.

    recording - flag the thread checks every slice (our Stop signal)
    chunks    - every 32ms slice of audio the thread has captured
    """

    def __init__(self):
        self.chunks = []
        self.recording = False
        self.thread = None

    def _record_loop(self):
        """Runs INSIDE the background thread until stop() flips and flag."""
        self.recording = True
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="float32", blocksize=512) as stream:
            while self.recording:
                data, _ = stream.read(512)
                self.chunks.append(data.copy())

    def start(self):
        self.chunks = []
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()

    def stop(self, path):
        """Signal the thread to finish, wait for it, save the WAV."""
        self.recording = False
        self.thread.join(timeout=5)          # wait for the thread to exit
        audio = np.concatenate(self.chunks)
        sf.write(path, audio, SAMPLE_RATE)
        return len(audio) / SAMPLE_RATE      # recording Length in seconds


st.set_page_config(page_title="Lecture Notes AI")
st.title("Lecture Notes AI")
st.write("Record each topic your teacher explains. One Start-Stop = one note.")

# session_state = the app's memory that survives between clicks
if "recorder" not in st.session_state:
    st.session_state.recorder = None
if "last_clip" not in st.session_state:
    st.session_state.last_clip = None
if "last_text" not in st.session_state:
    st.session_state.last_text = None
if "last_summary" not in st.session_state:
    st.session_state.last_summary = None

col1, col2 = st.columns(2)

# disabled= makes the impossible clicks grey: you can't Start twice,
# you can't Stop before Starting
if col1.button("Start recording", disabled=st.session_state.recorder is not None):
    st.session_state.recorder = Recorder()
    st.session_state.recorder.start()
    st.success("Recording... press Stop when the topic ends.")

if col2.button("Stop recording", disabled=st.session_state.recorder is None):
    path = os.path.join(RECORDINGS, f"topic_{int(st.session_state.get('n', 0)) + 1:02d}.wav")
    seconds = st.session_state.recorder.stop(path)
    st.session_state.recorder = None
    st.session_state.last_clip = path
    st.session_state.n = st.session_state.get("n", 0) + 1
    st.info(f"Saved {seconds:.1f}s to {os.path.basename(path)}")
    with st.spinner("Transcribing... (CPU, this can take a moment)"):
        st.session_state.last_text = transcribe_chinese(path)
    with st.spinner("Summarizing into notes..."):
        st.session_state.last_summary = summarize_chinese(st.session_state.last_text)
    note_path = os.path.join(NOTES, f"topic_{st.session_state.n:02d}.md")
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(f"# Topic {st.session_state.n}\n\n")
        f.write("## Transcript\n\n" + st.session_state.last_text + "\n\n")
        f.write("## Summary\n\n" + st.session_state.last_summary + "\n")
        
# Playback so you can verify mic actually captured the topic
if st.session_state.last_clip:
    st.subheader("Last recording")
    st.audio(st.session_state.last_clip)
if st.session_state.get("last_text"):
    st.subheader("Transcript")
    st.write(st.session_state.last_text)

# ===================================================================
# Milestone 4 - Study library: every note saved in notes/,
# listed and expandable for review before the exam.
# ===================================================================
st.divider()
st.header("Study library")

if os.path.isdir(NOTES):
    note_files = sorted(f for f in os.listdir(NOTES) if f.endswith(".md"))
    if not note_files:
        st.info("No notes yet. Record your first topic above.")
    for fname in note_files:
        with open(os.path.join(NOTES, fname), encoding="utf-8") as f:
            content = f.read()
        # st.expander = a collapsible box; click to open a topic's notes
        with st.expander(f"📒 {fname}"):
            st.markdown(content)
