# 감정 분석 로직

import torch
from .loader import tokenizer, model

labels = ['angry', 'anxious', 'embarrassed', 'happy', 'heartache', 'sad']

def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    # 결과 딕셔너리 : 소수점 2자리 까지만
    result = {
        label: round(float(probs[i]), 2)
        for i, label in enumerate(labels)
    }

    # 값 기준 정렬
    sorted_result = sorted(result.items(), key=lambda x:x[1], reverse=True)

    return [
        {"감정" : label, "점수" : score}
        for label, score in sorted_result[:3] #if score > 0.3
    ]