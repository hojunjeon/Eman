# API 엔드포인트

from fastapi import APIRouter
from app.model.inference import predict

router = APIRouter()

@router.post("/analyze")
def analyze(text: str):
    return predict(text)