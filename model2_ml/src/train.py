import os
import joblib

from preprocess import (
    load_data,
    create_features
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

print("Loading dataset...")

df = load_data()

df = create_features(df)

# Binary label
# All known interactions = 1

df["label"] = 1

# Create negatives

sample_size = 50000

negative_df = df.sample(
    sample_size,
    random_state=42
).copy()

negative_df["label"] = 0

negative_df["text"] = (
    negative_df["Drug 2"]
    + " "
    + negative_df["Drug 1"]
)

final_df = df[
    ["text", "label"]
]

negative_df = negative_df[
    ["text", "label"]
]

final_df = final_df._append(
    negative_df,
    ignore_index=True
)

print(
    "Total Training Samples:",
    len(final_df)
)

X = final_df["text"]

y = final_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(
    max_features=10000
)

X_train_vec = vectorizer.fit_transform(
    X_train
)

X_test_vec = vectorizer.transform(
    X_test
)

model = LogisticRegression(
    max_iter=1000
)

print("Training...")

model.fit(
    X_train_vec,
    y_train
)

preds = model.predict(
    X_test_vec
)

print(
    "\nAccuracy:",
    accuracy_score(
        y_test,
        preds
    )
)

print(
    classification_report(
        y_test,
        preds
    )
)

os.makedirs(
    "../model",
    exist_ok=True
)

joblib.dump(
    model,
    "../model/model.pkl"
)

joblib.dump(
    vectorizer,
    "../model/vectorizer.pkl"
)

print(
    "\nModel saved successfully."
)