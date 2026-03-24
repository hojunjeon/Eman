# 감정 분석 로직

import torch
from .loader import tokenizer, model

labels = ['angry', 'anxious', 'embarrassed', 'happy', 'heartache', 'sad']

def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    result = {
        label: float(probs[i])
        for i, label in enumerate(labels)
    }

    return result