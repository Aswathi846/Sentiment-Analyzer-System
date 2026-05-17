import os
import requests
import demoji
from PIL import Image
import io
import base64
from dotenv import load_dotenv

# =====================================================================
# BULLETPROOF PATH RESOLUTION FOR YOUR .ENV FILE
# =====================================================================
# This explicitly finds your project root folder and loads the .env file
current_file_path = os.path.abspath(__file__)                    # /src/agent.py
src_folder = os.path.dirname(current_file_path)                  # /src
project_root = os.path.abspath(os.path.join(src_folder, ".."))   # Project Root
env_path = os.path.join(project_root, ".env")

# Force load from the verified absolute path
load_dotenv(dotenv_path=env_path, override=True)
# =====================================================================

class SentimentAgent:
    def __init__(self, api_key=None):
        """
        Initializes the agent by targeting the raw HTTP endpoint directly.
        This completely bypasses Google's local SDK credential bugs.
        """
        raw_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not raw_key:
            raise ValueError(f"❌ GEMINI_API_KEY not found. Looked in: {env_path}")
        
        # Clean the key of any hidden spaces or accidental quotation marks
        self.api_key = raw_key.strip().strip("'").strip('"')

    def explain_sentiment(self, text, sentiment_label, image_file=None):
        """
        Calls Gemini 1.5 Flash using raw, pristine HTTP POST requests.
        """
        text_with_emoji_desc = demoji.replace(text, " ")

        prompt = f"""
        You are a UK-based AI Analyst specializing in Gen-Z communication and internet slang.
        
        Input Text: "{text_with_emoji_desc}"
        Local Model Prediction: {sentiment_label}
        
        Task:
        1. Identify and explain specific slang (e.g., 'mid', 'no cap', 'fire', 'banger').
        2. Justify why the sentiment is {sentiment_label}.
        3. If an image is provided, explain the 'visual vibe' and if it contains irony.
        4. Suggest a culturally relevant, professional response for a UK business.
        
        Keep the analysis concise and expert.
        """

        # 1. Structure the raw Gemini API request payload
        parts = [{"text": prompt}]

        # 2. Handle image conversion to base64 if provided
        if image_file:
            try:
                img = Image.open(image_file)
                if img.mode in ('RGBA', 'LA'):
                    img = img.convert('RGB')
                
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_str
                    }
                })
            except Exception as e:
                return f"❌ Image Processing Error: {str(e)}"

        # 3. Construct the clean, unified payload
        payload = {
            "contents": [{
                "parts": parts
            }]
        }

        # 4. Fire the direct HTTP post request to the Google Endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(url, json=payload, headers=headers)
            response_json = response.json()

            if response.status_code != 200:
                error_msg = response_json.get("error", {}).get("message", "Unknown Error")
                return f"❌ API Error ({response.status_code}): {error_msg}"

            return response_json['candidates'][0]['content']['parts'][0]['text']

        except KeyError:
            return f"❌ Parsing Error: Unexpected response structure format. Got: {response_json}"
        except Exception as e:
            return f"❌ Agent Network Error: {str(e)}"