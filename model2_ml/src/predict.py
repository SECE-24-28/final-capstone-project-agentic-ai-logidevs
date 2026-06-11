import joblib


model = joblib.load(
    "model/model.pkl"
)

vectorizer = joblib.load(
    "model/vectorizer.pkl"
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

    if probability > 0.5:

        return {
            "result":
                "Possible Interaction",
            "confidence":
                round(
                    float(probability),
                    2
                )
        }

    return {
        "result":
            "No Strong Interaction",
        "confidence":
            round(
                float(probability),
                2
            )
    }