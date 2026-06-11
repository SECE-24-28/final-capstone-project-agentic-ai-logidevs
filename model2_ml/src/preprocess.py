import pandas as pd
import kagglehub
import os


def load_data():

    path = kagglehub.dataset_download(
        "mghobashy/drug-drug-interactions"
    )

    csv_file = os.path.join(
        path,
        "db_drug_interactions.csv"
    )

    df = pd.read_csv(csv_file)

    df.dropna(inplace=True)

    return df


def create_features(df):

    df["text"] = (
        df["Drug 1"].astype(str)
        + " "
        + df["Drug 2"].astype(str)
        + " "
        + df["Interaction Description"].astype(str)
    )

    return df