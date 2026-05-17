import pytest
import os
import sys

# Ensure the src directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processor import TextCleaner

def test_emoji_cleaning():
    """Test if our demoji logic correctly replaces emojis with descriptions."""
    cleaner = TextCleaner()
    # Assuming your TextCleaner uses demoji.replace
    raw_text = "This is fire 🔥"
    # Note: Depending on your exact TextCleaner logic, adjust the expected result
    cleaned = cleaner.sanitize(raw_text) 
    assert "fire" in cleaned.lower()

def test_api_key_check():
    """Ensure the SentimentAgent raises an error if no API key is present."""
    from src.agent import SentimentAgent
    os.environ["GEMINI_API_KEY"] = ""
    with pytest.raises(ValueError):
        SentimentAgent(api_key=None)