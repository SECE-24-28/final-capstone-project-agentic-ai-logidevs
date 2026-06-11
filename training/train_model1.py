import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
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
epochs = 2

for epoch in range(epochs):  
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_loader):
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
        optimizer.step()

        total_loss += loss.item()

        if step % 50 == 0:
            print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item()}")

    print(f"Epoch {epoch+1} completed | Avg Loss: {total_loss/len(train_loader)}")

# -------------------
# 10. Save model
# -------------------
import os
os.makedirs("models", exist_ok=True)

model.save_pretrained("models/model1")
tokenizer.save_pretrained("models/model1")

print("Model saved to models/model1")