from transformers import pipeline
import pandas as pd

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

def get_model():
    model = pipeline(
        "summarization",
        model="google-t5/t5-base",
        device="cpu",
    )
    return model

def main():
    summarizer = get_model()
    df = pd.read_csv("summarization_dataset_test.csv")

    df["generated_summary"] = df["origin_text"].apply(
        lambda text: summarizer(text, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]
    )

    print(df)

if "__main__" == __name__:
    main()