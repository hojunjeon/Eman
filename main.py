from fastapi import FastAPI
from transformers import pipeline
import torch

app = FastAPI()

# CORS 세팅
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 일단 전체 허용
    allow_credentials=True,
    allow_methods=["*"],   # OPTIONS 포함
    allow_headers=["*"],
)

# 1. gpu 세팅
device = 1 if torch.cuda.is_available() else 0

if device:
    print("GPU 세팅 성공!")

# 2. model load
pipe = pipeline("text-classification", 
                # model="j-hartmann/emotion-english-distilroberta-base", # 영어
                model="nlp04/korean_sentiment_analysis_kcelectra", # 한국어
                )

print("모델 로드 완료!")

# ------ api 를 하나 씩

@app.get("/")
def home():
    return {"메세지": "서버 정상적으로 실행 중!"}


@app.post("/predict")
def predict(data: dict):
    # 1. 사용자로부터 문장을 전달받음
    text = data.get("text", "")

    # 2. 문장의 감정을 분석
    raw_result = pipe(text)
    # [결과] raw_result = [{'label': 'sadness', 'score': 0.8056379556655884}]
    print(f"raw_result = {raw_result}")

    # 3. 분석 결과를 return    
    return {
        "메세지": text,
        "점수": float(raw_result[0]["score"]),
        "감정": raw_result[0]["label"]
            }