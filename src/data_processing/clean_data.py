import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# 1. RAW DATA LOAD
# =========================
df = pd.read_csv("D:/cenann/review/data/raw/hb_data.csv")

df = df[["combined_text", "label"]].dropna()

print("RAW:", df.shape)

# =========================
# 2. (OPSİYONEL) NEUTRAL CLEAN
# =========================
strong_negative = ["berbat", "kötü", "iade", "rezalet", "çöp"]
strong_positive = ["mükemmel", "harika", "süper", "efsane"]

def is_clean_neutral(text):
    text = str(tegit add .gitignore app srcxt).lower()
    if any(w in text for w in strong_negative):
        return False
    if any(w in text for w in strong_positive):
        return False
    return True

neutral = df[df["label"] == 1]
others = df[df["label"] != 1]

neutral = neutral[neutral["combined_text"].apply(is_clean_neutral)]

df = pd.concat([others, neutral])

print("AFTER CLEAN:", df.shape)

# =========================
# 3. BALANCE (20K EACH CLASS)
# =========================
df_0 = df[df["label"] == 0].sample(20000, random_state=42)
df_1 = df[df["label"] == 1].sample(20000, random_state=42)
df_2 = df[df["label"] == 2].sample(20000, random_state=42)

df = pd.concat([df_0, df_1, df_2]).sample(frac=1, random_state=42)

print("\nBALANCED:")
print(df["label"].value_counts())

# =========================
# 4. SPLIT
# =========================
train_val, test = train_test_split(
    df,
    test_size=0.15,
    stratify=df["label"],
    random_state=42
)

train, val = train_test_split(
    train_val,
    test_size=0.1765,
    stratify=train_val["label"],
    random_state=42
)

# =========================
# 5. SAVE
# =========================
out_path = "D:/cenann/review/data/processed"

train.to_csv(f"{out_path}/train.csv", index=False)
val.to_csv(f"{out_path}/val.csv", index=False)
test.to_csv(f"{out_path}/test.csv", index=False)

print("\nDONE - FULL PIPELINE READY")