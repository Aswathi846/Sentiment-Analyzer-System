import re 
from pydantic import BaseModel, field_validator #data validator =pydantic

class ReviewInput(BaseModel):
    text: str

    @field_validator('text')
    @classmethod
    def check_length(cls, v):
        if not v or len(v.strip()) < 5:
            raise ValueError('Review is too short!')
        return v

class TextCleaner:
    @staticmethod
    def sanitize(text: str) -> str:
        # If text is None or not a string, return empty string
        if not isinstance(text, str):
            return ""
        
        # 1. Remove HTML
        clean = re.sub(r'<[^>]+>', '', text)
        # 2. Normalize whitespace
        clean = ' '.join(clean.split())
        # 3. Reduce repeated characters
        clean = re.sub(r'(.)\1{2,}', r'\1\1', clean)
        
        return str(clean).strip()