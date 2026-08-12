import os
import pickle
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# =========================
# 1. VERİ
# =========================
DATA_PATH = r"D:\cenann\review\data\processed\train.csv"
df = pd.read_csv(DATA_PATH)

texts = df["combined_text"].dropna().astype(str).tolist()

print("Veri sayısı:", len(texts))

# =========================
# 2. EMBEDDING MODEL
# =========================
embedding_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# =========================
# 3. BERTopic TRAIN
# =========================
topic_model = BERTopic(
    embedding_model=embedding_model,
    min_topic_size=40,
    nr_topics="auto",
    calculate_probabilities=True,
    verbose=True
)

topics, probs = topic_model.fit_transform(texts)

print("\n=== TOPIC INFO ===")
print(topic_model.get_topic_info())

# =========================
# 4. PKL KAYDET
# =========================
SAVE_PATH = r"D:\cenann\guncel\artifacts\bertopic_model.pkl"

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

with open(SAVE_PATH, "wb") as f:
    pickle.dump(topic_model, f)

print("\nMODEL KAYDEDİLDİ:", SAVE_PATH)

# =========================
# 5. MODEL LOAD (TEST)
# =========================
with open(SAVE_PATH, "rb") as f:
    model = pickle.load(f)

# =========================
# 6. TEST VERİLERİ
# =========================
print("\n=== TEST ===\n")

test_texts = [
    "kargo çok hızlı geldi paket sağlamdı",
    "ürün bozuk çıktı çalışmıyor",
    "müşteri hizmetleri çok yardımcı oldu",
    "fiyatına göre çok iyi ürün",
    "kargo geç geldi paket hasarlıydı",
    "ürün beklediğimden iyi çıktı",
    "beden küçük geldi değiştirmek zorunda kaldım",
    "sipariş iptali çok zor oldu sistem kötü"
]

for t in test_texts:
    topic_id, prob = model.transform([t])
    print(t)
    print("TOPIC ID:", topic_id[0])
    print("-" * 40)