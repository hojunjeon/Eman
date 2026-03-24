# request/response 모델

from pydantic import BaseModel

class SentimentRequest(BaseModel):
    text: str