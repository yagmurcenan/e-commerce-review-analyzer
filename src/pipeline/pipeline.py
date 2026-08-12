import torch
import numpy as np
from collections import Counter
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic


# ==================================================
# 1. SENTIMENT MODEL
# ==================================================
SENTIMENT_PATH = r"D:\cenann\guncel\models"

tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_PATH)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_PATH)
sentiment_model.eval()


def predict_sentiment_batch(texts, batch_size=32):
    sentiments = []
    confidences = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = sentiment_model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        preds = torch.argmax(probs, dim=1).cpu().numpy()
        conf = torch.max(probs, dim=1).values.cpu().numpy()

        label_map = {0: "negative", 1: "neutral", 2: "positive"}

        sentiments.extend([label_map[p] for p in preds])
        confidences.extend(conf.tolist())

    return sentiments, confidences


# ==================================================
# 2. EMBEDDING MODEL
# ==================================================
embedder = SentenceTransformer("intfloat/multilingual-e5-base")


# ==================================================
# 3. BERTOPIC
# ==================================================
BERTOPIC_PATH = r"D:\cenann\guncel\artifacts\bertopic_model.pkl"
topic_model = BERTopic.load(BERTOPIC_PATH)


def get_bertopic_features(texts):
    topics, probs = topic_model.transform(texts)
    return topics, probs


# ==================================================
# 4. CATEGORY RULES
# ==================================================
CATEGORIES = {
    "Kargo / Teslimat": [
        "kargo hızlı geldi",
        "kargo geç geldi",
        "kargo çok geç geldi",
        "teslimat gecikti",
        "teslimat çok hızlıydı",
        "ürün elime ulaştı",
        "paketleme sağlamdı",
        "paketleme çok iyiydi",
        "paket hasarlı geldi",
        "kutu ezilmişti",
        "kargo sorunsuzdu",
        "hızlı teslimat",
        "geç teslimat",
        "kargo firması kötüydü",
        "kargo şirketi sorunluydu",
        "ertesi gün teslim edildi",
        "çok hızlı ulaştı",
        "uzun sürede geldi"
    ],

    "Ürün Kalitesi": [
        "ürün kaliteli",
        "çok kaliteli",
        "malzeme kaliteli",
        "ürün kalitesiz",
        "çok kötü kalite",
        "dayanıklı ürün",
        "ürün kırık geldi",
        "ürün bozuk çıktı",
        "ürün çalışmıyor",
        "kusurlu ürün",
        "defolu geldi",
        "ürün sağlam değil",
        "beklediğimden kaliteli",
        "kalite çok iyi",
        "malzeme ince ve dayanıksız",
        "parça eksik çıktı",
    ],

    "Fiyat / Performans": [
        "fiyatına göre iyi",
        "fiyat performans ürünü",
        "çok uygun fiyat",
        "ucuz ve iyi",
        "pahalı ama değer",
        "fiyatı uygun",
        "fiyatı çok iyi",
        "f/p ürün",
        "parasına göre iyi",
        "değerini hak ediyor",
        "çok pahalı",
        "fiyat yüksek",
        "uygun fiyatlı",
        "performansına göre iyi",
        "beklentiyi karşılıyor fiyatına göre"
    ],

    "Müşteri Hizmetleri": [
        "müşteri hizmetleri kötü",
        "destek ekibi yardımcı oldu",
        "müşteri hizmetleri ilgisiz",
        "destek çok iyiydi",
        "yardımcı olmadılar",
        "iletişim kuramadım",
        "satıcı çok ilgiliydi",
        "satıcı cevap vermedi",
        "destek ekibi hızlı dönüş yaptı",
        "müşteri temsilcisi yardımcı oldu",
        "problem çözülmedi",
        "çok ilgililerdi",
        "satıcı ilgisizdi"
    ],

    "Sipariş / Sistem": [
        "sipariş iptal oldu",
        "sipariş gecikti",
        "sipariş hatalı geldi",
        "sistem hatası",
        "site çalışmıyor",
        "ödeme sorunu yaşadım",
        "sipariş onaylanmadı",
        "sipariş yanlış geldi",
        "kargo takip çalışmıyor",
        "site donuyor",
        "alışveriş tamamlanmadı",
        "açıklamalar eksik"
    ],

    "Genel Memnuniyet": [
        "çok memnunum",
        "memnun kaldım",
        "tekrar alırım",
        "yeniden satın alırım",
        "ürünü ikinci kez aldım",
        "ürünü ikinci kez satın alıyorum",
        "harika ürün",
        "mükemmel",
        "çok beğendim",
        "tavsiye ederim",
        "kesinlikle alınır",
        "çok iyi ürün",
        "çok başarılı",
        "gayet memnunum",
        "idare eder",
        "ortalama ürün",
        "beklentimi karşıladı",
        "güzel ürün"
    ]
}


category_embeddings = {
    cat: embedder.encode(samples)
    for cat, samples in CATEGORIES.items()
}


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def predict_topics_batch(texts, threshold=0.55):
    embeddings = embedder.encode(texts, batch_size=64)

    results = []

    for emb in embeddings:

        best_cat = None
        best_score = -1

        for cat, cat_vectors in category_embeddings.items():

            scores = [cosine_sim(emb, v) for v in cat_vectors]
            score = max(scores)

            if score > best_score:
                best_score = score
                best_cat = cat

        if best_score < threshold:
            results.append(("Genel Deneyim", float(best_score)))
        else:
            results.append((best_cat, float(best_score)))

    return results


# ==================================================
# 5. HELPERS
# ==================================================
def split_review(text):
    separators = [" ama ", " fakat ", " ancak "]

    parts = [text]

    for sep in separators:
        new_parts = []
        for p in parts:
            new_parts.extend(p.split(sep))
        parts = new_parts

    return [t.strip() for t in parts if t.strip()]


def sentiment_postprocess(text, sentiment):
    text = text.lower()

    positive_hints = [
        "gayet iyi", "beğendim", "çok beğendim",
        "memnun kaldım", "çok memnunum",
        "harika", "mükemmel", "tavsiye ederim",
        "alınır", "ipek gibi", "fiyat performans",
        "çok kaliteli", "güzel", "süper"
    ]

    negative_hints = [
        "berbat", "rezalet", "hayal kırıklığı",
        "çok kötü", "değmez",
        "kırık geldi", "bozuk çıktı", "çalışmıyor"
    ]

    if any(word in text for word in positive_hints):
        return "positive"

    if any(word in text for word in negative_hints):
        return "negative"

    return sentiment


# ==================================================
# 6. PIPELINE
# ==================================================
def analyze_reviews(reviews):

    cleaned = []
    for r in reviews:
        if isinstance(r, str) and len(r.strip()) > 0:
            cleaned.extend(split_review(r))

    reviews = cleaned

    sentiments, confidences = predict_sentiment_batch(reviews)
    topics = predict_topics_batch(reviews)
    bertopic_topics, _ = get_bertopic_features(reviews)

    results = []

    for i in range(len(reviews)):

        final_sentiment = sentiment_postprocess(
            reviews[i],
            sentiments[i]
        )

        results.append({
            "text": reviews[i],
            "sentiment": final_sentiment,
            "confidence": float(confidences[i]),
            "topic": topics[i][0],
            "topic_score": topics[i][1],
            "bertopic_cluster": int(bertopic_topics[i])
        })

    return results


# ==================================================
# 7. SCORE
# ==================================================
def compute_review_score(r):

    sentiment_weight = {
        "positive": 1.0,
        "neutral": 0.2,
        "negative": 0.0
    }

    return (
        sentiment_weight.get(r["sentiment"], 0.5) * 0.75 +
        r["confidence"] * 0.15 +
        r["topic_score"] * 0.10
    )


# ==================================================
# 8. DASHBOARD
# ==================================================
def build_dashboard(results):

    sentiment_counts = Counter(r["sentiment"] for r in results)
    topic_counts = Counter(r["topic"] for r in results)

    best_reviews = sorted(results, key=compute_review_score, reverse=True)[:5]
    worst_reviews = sorted(results, key=compute_review_score)[:5]

    return {
        "total_reviews": len(results),
        "sentiment_distribution": dict(sentiment_counts),
        "topic_distribution": dict(topic_counts),

        "best_reviews": [
            {"text": r["text"], "confidence": round(r["confidence"], 4)}
            for r in best_reviews
        ],

        "worst_reviews": [
            {"text": r["text"], "confidence": round(r["confidence"], 4)}
            for r in worst_reviews
        ]
    }


# ==================================================
# 9. TEST
# ==================================================
if __name__ == "__main__":

    reviews = [
        "kargo çok hızlı geldi paket sağlamdı",
        "kargo gecikti çok bekledim",
        "ürün kırık çıktı iade ettim",
        "çok memnunum tekrar alırım",
        "harika ürün kesinlikle tavsiye ederim",
        "kargo hızlı ama ürün kalitesiz",
        "ürün güzel ama kargo geç geldi"
    ]

    results = analyze_reviews(reviews)
    dashboard = build_dashboard(results)

    print(json.dumps(dashboard, indent=2, ensure_ascii=False))