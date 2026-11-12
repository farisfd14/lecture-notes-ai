"""
Milestone 3 - summarize each topic into study notes.

Sends the Chinese transcript to a local LLM through Ollama.
Model choice: qwen2.5 (alibaba) - built for Chinese, much
better Mandarin than llama3.2.

This file reuses two hard-won patterns from the voice-assistant
project (src/brain.py) - same bugs, same fixes:
  1. _session with trust_env=False -> the machine proxy must not
     touch localhost traffic
  2. pick_answer()                 -> never hardcode a JSON field
     name; the answer is the longest string in the response
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"

_session = requests.session()
_session.trust_env = False          # ignore system proxy for localhost


def pick_answer(data: dict) -> str:
    """The answer is the longest string in Ollama's JSON response.

    Assumption: the answer is a full text, always longer than
    metadata strings like the model name.
    """
    strings = [v for v in data.values() if isinstance(v, str)]
    return max(strings, key=len).strip()


def summarize_chinese(text):
    """Summarize transcript into concise Chinese study notes."""
    prompt = (
        "你是一名学习助手。请用英文总结下面的课堂内容，"
        "要求: \n"
        "1. 先写一段2-3句话的简短总结\n"
        "2. 然后列出3-5个要点\n"
        "3. 如果内容里有考试可能考的重点，请在最后标注出来\n\n"
        f"课堂内容: \n{text}"
    )
    resp = _session.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=300,                # Long transcripts take a while on CPU
    )
    return pick_answer(resp.json())


if __name__ == "__main__":
    demo = "细胞膜由磷脂双分子层构成，具有选择透过性，蛋白质镶嵌在其中负责运输物质。"
    print(summarize_chinese(demo))