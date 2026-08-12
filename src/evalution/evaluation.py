import pandas as pd
import torch
from numpy.ma.core import negative
from torch.distributions.constraints import positive
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix
import numpy as np

from data_processing.clean_data import neutral

# =====================
# MODEL LOAD
# =====================
MODEL_PATH = r"D:\cenann\review\models"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# =====================
# DATA LOAD
# =====================
DATA_PATH = r"D:\cenann\review\data\processed\test.csv"

df = pd.read_csv(DATA_PATH)

df = df.rename(columns={"combined_text": "text"})
df = df[["text", "label"]].dropna()

df["label"] = df["label"].astype(int)

texts = df["text"].tolist()
labels = df["label"].tolist()

# =====================
# PREDICT
# =====================
y_true = []
y_pred = []

for text, label in zip(texts, labels):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=1).item()

    y_true.append(label)
    y_pred.append(pred)

# =====================
# CONFUSION MATRIX
# =====================
cm = confusion_matrix(y_true, y_pred)


print("\n===== CONFUSION MATRIX =====\n")
print(cm)

import matplotlib.pyplot as plt
from collections import Counter

# label dağılımı
counts = Counter(labels)

classes = ["Class 0", "Class 1", "Class 2"]  # istersen Positive/Neutral/Negative yaparız
values = [counts[0], counts[1], counts[2]]

plt.figure(figsize=(6,4))
plt.bar(classes, values)

plt.title("Dataset Label Distribution")
plt.xlabel("Classes")
plt.ylabel("Number of Samples")

# sayı yazdırma
for i, v in enumerate(values):
    plt.text(i, v, str(v), ha='center', va='bottom')

plt.show()