from collections import Counter
from src.pipeline.pipeline import compute_review_score


def build_dashboard(results):

    # =========================
    # SCORE COMPUTATION
    # =========================
    for r in results:
        r["final_score"] = compute_review_score(r)

    total_reviews = len(results)

    # =========================
    # GLOBAL METRICS
    # =========================
    sentiment_counts = Counter(r.get("sentiment", "neutral") for r in results)
    topic_counts = Counter(r.get("topic", "Unknown") for r in results)

    sentiment_bar = {
        "positive": round(sentiment_counts.get("positive", 0) / total_reviews, 4) if total_reviews else 0,
        "neutral": round(sentiment_counts.get("neutral", 0) / total_reviews, 4) if total_reviews else 0,
        "negative": round(sentiment_counts.get("negative", 0) / total_reviews, 4) if total_reviews else 0
    }

    # =========================
    # GROUP BY TOPIC
    # =========================
    topics = {}
    for r in results:
        topic = r.get("topic", "Unknown")
        topics.setdefault(topic, []).append(r)

    topic_output = {}

    # =========================
    # PER TOPIC
    # =========================
    for topic_name, items in topics.items():

        total_topic_reviews = len(items)
        sent_counts = Counter(i.get("sentiment", "neutral") for i in items)

        topic_sentiment_bar = {
            "positive": round(sent_counts.get("positive", 0) / total_topic_reviews, 4) if total_topic_reviews else 0,
            "neutral": round(sent_counts.get("neutral", 0) / total_topic_reviews, 4) if total_topic_reviews else 0,
            "negative": round(sent_counts.get("negative", 0) / total_topic_reviews, 4) if total_topic_reviews else 0
        }

        positives = [i for i in items if i.get("sentiment") == "positive"]
        negatives = [i for i in items if i.get("sentiment") == "negative"]

        # =========================
        # BEST REVIEW
        # =========================
        if positives:
            best = max(positives, key=lambda x: x.get("final_score", 0))
        else:
            best = None

        # =========================
        # WORST REVIEW
        # =========================
        if negatives:
            worst = min(negatives, key=lambda x: x.get("final_score", 0))
        else:
            worst = None

        # =========================
        # FORMAT HELPERS
        # =========================
        def format_review(r, fallback_msg):
            if r is None:
                return {"message": fallback_msg}
            return {
                "text": r.get("text", ""),
                "sentiment": r.get("sentiment", "neutral"),
                "confidence": round(r.get("confidence", 0), 4),
                "topic_score": round(r.get("topic_score", 0), 4),
                "final_score": round(r.get("final_score", 0), 4)
            }

        # =========================
        # OUTPUT
        # =========================
        topic_output[topic_name] = {
            "count": total_topic_reviews,

            "sentiment_distribution": {
                "positive": sent_counts.get("positive", 0),
                "neutral": sent_counts.get("neutral", 0),
                "negative": sent_counts.get("negative", 0)
            },

            "sentiment_bar": topic_sentiment_bar,

            "best_review": format_review(
                best,
                "Bu konuya ait pozitif yorum bulunamadı"
            ),

            "worst_review": format_review(
                worst,
                "Bu konuya ait negatif yorum bulunamadı"
            )
        }

    # =========================
    # FINAL OUTPUT
    # =========================
    return {
        "total_reviews": total_reviews,

        "sentiment_distribution": {
            "positive": sentiment_counts.get("positive", 0),
            "neutral": sentiment_counts.get("neutral", 0),
            "negative": sentiment_counts.get("negative", 0)
        },

        "sentiment_bar": sentiment_bar,

        "topic_distribution": dict(topic_counts),

        "topics": topic_output
    }