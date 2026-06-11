"""
Future integration layer.

Do not connect this
to Model 1 now.
"""

from predict import predict


def model2_interface(
    drug1,
    drug2
):

    result = predict(
        drug1,
        drug2
    )

    return {
        "source": "model2",
        "result":
            result["result"],
        "confidence":
            result["confidence"]
    }