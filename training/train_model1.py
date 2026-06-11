import os
import json

import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score
)

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from torch.optim import AdamW
# 1. Load dataset
# -------------------
df = pd.read_csv("data/train.csv")

print("Total rows:", len(df))
print(df.head())

# -------------------
# 2. Encode labels
# -------------------
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

os.makedirs("models/model1", exist_ok=True)

label_map = {
    int(i): label
    for i, label in enumerate(label_encoder.classes_)
}

with open(
    "models/model1/label_map.json",
    "w"
) as f:
    json.dump(label_map, f, indent=4)

num_labels = len(label_encoder.classes_)
print("Classes:", label_encoder.classes_)

# -------------------
# 3. Train-test split
# -------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df[["drug_a", "drug_b"]],
    df["label"],
    test_size=0.2,
    random_state=42
)

# -------------------
# 4. Tokenizer
# -------------------
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# -------------------
# 5. Dataset class
# -------------------
class DrugDataset(Dataset):

    def __init__(self, texts, labels):

        self.texts = texts.values
        self.labels = labels.values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        drug_a, drug_b = self.texts[idx]

        encoding = tokenizer(
            drug_a,
            drug_b,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(
                self.labels[idx],
                dtype=torch.long
            )
        }
# -------------------
# 6. DataLoaders
# -------------------
train_dataset = DrugDataset(train_texts, train_labels)
val_dataset = DrugDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# -------------------
# 7. Model
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

model.to(device)

# -------------------
# 8. Optimizer
# -------------------
optimizer = AdamW(model.parameters(), lr=2e-5)

# -------------------
# 9. Training loop
# -------------------
os.makedirs("logs", exist_ok=True)

log_file = open(
    "logs/training_log.txt",
    "w",
    encoding="utf-8"
)

best_accuracy = 0

epochs = 5

for epoch in range(epochs):

    print(f"\nEpoch {epoch+1}/{epochs}")

    model.train()

    train_loss = 0

    for batch in train_loader:

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()

    val_loss = 0

    all_preds = []
    all_labels = []

    mistakes = []

    with torch.no_grad():

        for batch in val_loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            val_loss += outputs.loss.item()

            preds = torch.argmax(
                outputs.logits,
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            for i in range(len(preds)):

                if preds[i] != labels[i]:

                    mistakes.append({
                        "drug_a": batch["drug_a"][i],
                        "drug_b": batch["drug_b"][i],
                        "true":
                            label_encoder.inverse_transform(
                                [labels[i].cpu().item()]
                            )[0],
                        "pred":
                            label_encoder.inverse_transform(
                                [preds[i].cpu().item()]
                            )[0]
                    })

    avg_val_loss = val_loss / len(val_loader)

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    print(
        f"Train Loss: {avg_train_loss:.4f}"
    )

    print(
        f"Val Loss: {avg_val_loss:.4f}"
    )

    print(
        f"Val Accuracy: {accuracy:.4f}"
    )

    report = classification_report(
        all_labels,
        all_preds,
        target_names=label_encoder.classes_
    )

    print(report)

    log_file.write(
        f"\nEpoch {epoch+1}\n"
    )

    log_file.write(
        f"Train Loss: {avg_train_loss}\n"
    )

    log_file.write(
        f"Val Loss: {avg_val_loss}\n"
    )

    log_file.write(
        f"Accuracy: {accuracy}\n"
    )

    log_file.write(
        report + "\n"
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        model.save_pretrained(
            "models/model1"
        )

        tokenizer.save_pretrained(
            "models/model1"
        )

        print(
            f"BEST MODEL SAVED "
            f"(Accuracy={accuracy:.4f})"
        )

    print("\nMisclassified Samples:")

    for error in mistakes[:10]:

        print(error)

log_file.close()

print(
    f"\nTraining Complete. "
    f"Best Accuracy = {best_accuracy:.4f}"
)
# -------------------
# 10. Save model
# -------------------
import os
os.makedirs("models", exist_ok=True)

model.save_pretrained("models/model1")
tokenizer.save_pretrained("models/model1")

print("Model saved to models/model1")