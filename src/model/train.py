import pandas as pd
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score

# =========================
# 1. LOAD DATA
# =========================
data_path = "/content/drive/MyDrive/processed"

train_df = pd.read_csv(f"{data_path}/train.csv")
val_df   = pd.read_csv(f"{data_path}/val.csv")
test_df  = pd.read_csv(f"{data_path}/test.csv")

print(train_df["label"].value_counts())

# =========================
# 2. MODEL + TOKENIZER
# =========================
model_name = "dbmdz/bert-base-turkish-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

# =========================
# 3. CONVERT TO HF DATASET
# =========================
train_ds = Dataset.from_pandas(train_df)
val_ds   = Dataset.from_pandas(val_df)
test_ds  = Dataset.from_pandas(test_df)

# =========================
# 4. TOKENIZATION
# =========================
def tokenize(batch):
    return tokenizer(
        batch["combined_text"],
        truncation=True,
        padding=True
    )

train_ds = train_ds.map(tokenize, batched=True)
val_ds   = val_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize, batched=True)

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# =========================
# 5. DATA COLLATOR
# =========================
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =========================
# 6. METRICS
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro")
    }

# =========================
# 7. TRAINING ARGS
# =========================
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/processed/bert_model",

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=3,
    weight_decay=0.01,

    load_best_model_at_end=True,
    metric_for_best_model="f1",

    logging_steps=50,
    report_to="none"
)

# =========================
# 8. TRAINER
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# =========================
# 9. TRAIN
# =========================
trainer.train()

# =========================
# 10. SAVE MODEL
# =========================
trainer.save_model("/content/drive/MyDrive/processed/bert_model")
tokenizer.save_pretrained("/content/drive/MyDrive/processed/bert_model")

print("DONE")