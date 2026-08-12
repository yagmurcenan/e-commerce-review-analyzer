import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter

# =========================
# PATHS
# =========================
BASE_DIR = r"D:\cenann\guncel"

TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test")

MODEL_PATH = os.path.join(BASE_DIR, "models")

OUTPUT_PATH = os.path.join(BASE_DIR, "results", "analysis_results.json")

# =========================
# LOAD TEST DATA
# =========================

# CSV ise
if os.path.exists(TEST_PATH + ".csv"):
    df = pd.read_csv(TEST_PATH + ".csv")
    texts = df.iloc[:, 0].tolist()   # ilk kolon text varsayımı

# JSON ise
elif os.path.exists(TEST_PATH + ".json"):
    with open(TEST_PATH + ".json", "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [x["text"] for x in data]

# TXT ise
elif os.path.exists(TEST_PATH + ".txt"):
    with open(TEST_PATH + ".txt", "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines()]

else:
    raise FileNotFoundError("test dosyası bulunamadı (.csv / .json / .txt)")

# =========================
# MODEL LOAD
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

label_map = {0: "negative", 1: "neutral", 2: "positive"}

# =========================
# PREDICTION
# =========================
results = []

for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    pred = outputs.logits.argmax().item()
    sentiment = label_map[pred]

    results.append({
        "text": text,
        "sentiment": sentiment
    })

# =========================
# SAVE RESULTS
# =========================
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# =========================
# CHECK DISTRIBUTION
# =========================
counts = Counter(r["sentiment"] for r in results)

print("TEST TAMAMLANDI")
print("Dağılım:", counts)