from datasets import load_dataset
import csv

ds = load_dataset("bigbio/ddi_corpus", "ddi_corpus_bigbio_kb")


def build_entity_map(sample):
    return {
        e["id"]: e["text"][0]
        for e in sample["entities"]
    }


rows = []

for sample in ds["train"]:

    if len(sample["relations"]) == 0:
        

        continue

    entity_map = build_entity_map(sample)

    for rel in sample["relations"]:

        drug1 = entity_map[rel["arg1_id"]]
        drug2 = entity_map[rel["arg2_id"]]
        label = rel["type"].lower()

        rows.append([drug1, drug2, label])


with open("data/train.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["drug_a", "drug_b", "label"])
    writer.writerows(rows)

print("DONE. Total rows:", len(rows))