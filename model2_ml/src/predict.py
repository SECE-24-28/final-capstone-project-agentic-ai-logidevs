import os
import joblib

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

model = joblib.load(
    os.path.join(
        BASE_DIR,
        "model",
        "model.pkl"
    )
)

vectorizer = joblib.load(
    os.path.join(
        BASE_DIR,
        "model",
        "vectorizer.pkl"
    )
)


def predict(drug1, drug2):

    text = (
        drug1
        + " "
        + drug2
    )

    vec = vectorizer.transform(
        [text]
    )

    probability = (
        model.predict_proba(vec)[0][1]
    )

    if probability > 0.7:

        risk = "High"

    elif probability > 0.5:

        risk = "Medium"

    else:

        risk = "Low"

    return {
        "result":
            "Possible Interaction"
            if probability > 0.5
            else "No Strong Interaction",

        "confidence":
            round(
                float(probability),
                3
            ),

        "risk":
            risk
    }