import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# EMBEDDING MODEL (GÜÇLÜ)
# =========================
embedder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# =========================
# KATEGORİ TANIMLARI (EZBER DEĞİL, SEMANTIC ANCHOR)
# =========================
categories = {
    "Kargo / Teslimat": [
        "kargo hızlı geldi", "paket sağlamdı", "kargo gecikti", "teslimat geç oldu"
    ],

    "Ürün Kalitesi": [
        "ürün kaliteli", "çok sağlam", "bozuk geldi", "kırık ürün",
        "çalışmıyor", "dayanıklı malzeme", "çok kötü ürün", "rezalet kalite"
    ],

    "Fiyat / Performans": [
        "fiyatına göre iyi", "uygun fiyat", "ucuz ve kaliteli",
        "parasına değer", "f/p ürün", "çok pahalı değil"
    ],

    "Müşteri Hizmetleri": [
        "müşteri hizmetleri yardımcı oldu", "destek ekibi çözdü",
        "çağrı merkezi dönüş yaptı", "şikayet çözüldü", "geri dönüş yaptılar"
    ],

    "Sipariş / Sistem": [
        "sipariş iptali zor", "sistem hata verdi", "site çalışmıyor",
        "sipariş gecikti", "ödeme olmadı"
    ]
}

# =========================
# CATEGORY EMBEDDINGS (1 KEZ HESAPLANIR)
# =========================
category_embeddings = {}

for cat, samples in categories.items():
    emb = embedder.encode(samples)
    category_embeddings[cat] = np.mean(emb, axis=0)

# =========================
# COSINE SIMILARITY
# =========================
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =========================
# PREDICT
# =========================
def predict(text, threshold=0.45):
    text_emb = embedder.encode(text)

    best_cat = None
    best_score = -1

    for cat, cat_emb in category_embeddings.items():
        score = cosine_sim(text_emb, cat_emb)

        if score > best_score:
            best_score = score
            best_cat = cat

    if best_score < threshold:
        return "Diğer", best_score

    return best_cat, best_score

# =========================
# TEST
# =========================
test_texts = [
    "kargo çok hızlı geldi paket sağlamdı",
    "ürün bozuk çıktı çalışmıyor",
    "fiyatına göre çok iyi ürün",
    "müşteri hizmetleri çok yardımcı oldu",
    "sipariş iptali çok zor oldu sistem kötü",
    "baya uygun aldım çok memnunum",
    "bu kulaklık rezalet",
    "ürün aşırı kaliteli ve sağlam"
]

for t in test_texts:
    label, score = predict(t)
    print(t)
    print("LABEL:", label, "| SCORE:", round(score, 3))
    print("-" * 50)