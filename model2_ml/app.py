from flask import (
    Flask,
    request,
    jsonify
)

from src.predict import (
    predict
)

app = Flask(__name__)


@app.route(
    "/predict",
    methods=["POST"]
)
def predict_api():

    data = request.json

    drug1 = data["drug1"]
    drug2 = data["drug2"]

    result = predict(
        drug1,
        drug2
    )

    return jsonify(result)


if __name__ == "__main__":

    app.run(
        debug=True
    )