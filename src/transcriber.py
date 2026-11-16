"""
Milestone 2 - Chinese transcription with Whisper

Model choice note: we use 'base', NOT 'tiny'.
Whisper tiny is fast but its Chinese is poor; base is the
smallest model that handles Mandarin reliably.

language="zh" forces Mandarin output - without it, Whisper
guesses the language per clip, and on short noisy clips
classroom audio can be guessed wrong.
"""

import whisper

_model = None

def load_model():
    """Load whisper-base once (lazy: only when first needed)."""
    global _model
    if _model is None:
        _model = whisper.load_model("base", device="cpu")
    return _model


def transcribe_chinese(path):
    """Return the Chinese transcript of an audio file."""
    model = load_model()
    result = model.transcribe(
        path,
        language="zh",
        beam_size=1,                        # greedy decoding: ~2-3x faster than
                                            # the default beam search, tiny quality loss
        condition_on_previous_text=False    # IMPORTANT for classrooms: without this,
                                            # whisper loops/repeats text during silences
    )
    return result["text"].strip()