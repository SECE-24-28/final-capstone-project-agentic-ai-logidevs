import pandas as pd


def load_data():

    df = pd.read_csv("data/train.csv")

    df.dropna(inplace=True)

    return df


def create_features(df):

    df["text"] = (
        df["drug1"]
        + " "
        + df["drug2"]
    )

    return df