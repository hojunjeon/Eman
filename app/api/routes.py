# API 엔드포인트

from fastapi import APIRouter
from app.model.inference import predict
from app.schemas.sentiment import SentimentRequest

router = APIRouter()

@router.post("/analyze")
def analyze(request: SentimentRequest):
    return predict(request.text)