# src/transcriber.py
# Transcription with faster-whisper (CTranslate2 engine):
# same "base" model accuracy as openai-whisper, but 2-4x faster on CPU.
# The function signature is unchanged - app.py doesn't need to know

from faster_whisper import WhisperModel

_model = None

def load_model():
    global _model
    if _model is None:
        # device="cpu" + compute_types="int8" = optimized for laptops without GPU
        _model = WhisperModel("base", device="cpu", compute_type="int8")
    return _model


def transcribe_chinese(path):
    model = load_model()
    # faster-whisper returns a generator of segments, not one dict,
    # so we join the pieces ourselves
    segments, _ = model.transcribe(
        path,
        language="zh",
        beam_size=1,                            # greedy = faster, same setting as before
        condition_on_previous_text=False,       # stops repetition loops silence
    )
    return "".join(seg.text for seg in segments).strip()