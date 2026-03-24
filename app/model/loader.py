# 모델 로드 (초기 1회)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "Jinuuuu/KoELECTRA_fine_tunning_emotion"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)