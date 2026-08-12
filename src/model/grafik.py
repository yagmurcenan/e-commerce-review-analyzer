import os
import json
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter

# =========================
# PATHS
# =========================
BASE_DIR = r"D:\cenann\guncel"

CSV_PATH = os.path.join(BASE_DIR, "data", "sentiment_test_results.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models")

RESULT_PATH = os.path.join(BASE_DIR, "results", "analysis_results.json")

# =========================
# 1. LOAD CSV
# =========================
df = pd.read_csv(CSV_PATH)

# Eğer text kolonu varsa onu kullan, yoksa ilk sütun
if "text" in df.columns:
    texts = df["text"].astype(str).tolist()
else:
    texts = df.iloc[:, 0].astype(str).tolist()

print(f"Toplam veri: {len(texts)}")

# =========================
# 2. LOAD MODEL
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

label_map = {0: "negative", 1: "neutral", 2: "positive"}

# =========================
# 3. PREDICTION
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
# 4. SAVE RESULTS
# =========================
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

with open(RESULT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("RESULTS KAYDEDİLDİ")

# =========================
# 5. SENTIMENT DISTRIBUTION
# =========================
counts = Counter(r["sentiment"] for r in results)

labels = list(counts.keys())
values = list(counts.values())

# BAR CHART
plt.figure(figsize=(6,4))
plt.bar(labels, values)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")

bar_path = os.path.join(BASE_DIR, "poster_bar.png")
plt.savefig(bar_path)
plt.show()

# PIE CHART
plt.figure(figsize=(5,5))
plt.pie(values, labels=labels, autopct="%1.1f%%")
plt.title("Sentiment Ratio")

pie_path = os.path.join(BASE_DIR, "poster_pie.png")
plt.savefig(pie_path)
plt.show()

print("GRAFİKLER HAZIR!")
print("Bar:", bar_path)
print("Pie:", pie_path)