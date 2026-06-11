import os

import joblib

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.linear_model import (
    LogisticRegression
)

from preprocess import (
    load_data,
    create_features
)


df = load_data()

df = create_features(df)

X = df["text"]

y = df["label"]

vectorizer = TfidfVectorizer()

X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()

model.fit(X_vec, y)

os.makedirs(
    "model",
    exist_ok=True
)

joblib.dump(
    model,
    "model/model.pkl"
)

joblib.dump(
    vectorizer,
    "model/vectorizer.pkl"
)

print(
    "Training completed successfully"
)