from datasets import load_dataset

ds = load_dataset(
    "bigbio/ddi_corpus",
    "ddi_corpus_bigbio_kb"
)

for sample in ds["train"]:

    if len(sample["relations"]) > 0:

        print("\nFOUND RELATION\n")

        print("DOCUMENT ID:", sample["document_id"])

        print("\nENTITIES:")
        print(sample["entities"])

        print("\nRELATIONS:")
        print(sample["relations"])

        
        break