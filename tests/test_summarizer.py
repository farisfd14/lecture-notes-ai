# Tests for the answer-extraction logic (the pattern reused from
# the voice-assistant project). Run: py -m pytest tests/ -v

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from summarizer import pick_answer


def test_picks_the_longest_string():
    fake = {"model": "qwen2.5:3b", "done": "true",
            "response": "细胞膜由磷脂双分子层构成。"}
    assert pick_answer(fake) == "细胞膜由磷脂双分子层构成。"


def test_works_with_renamed_field():
    fake = {"model": "qwen2.5:3b",
            "some_future_field": "这是未来版本才会出现的字段名。"}
    assert pick_answer(fake) == "这是未来版本才会出现的字段名。"


def test_ignores_non_strings():
    fake = {"model": "qwen2.5:3b", "total_duration": 999,
            "reply": "我已经完成了你的请求，请查收。"}
    assert pick_answer(fake) == "我已经完成了你的请求，请查收。"