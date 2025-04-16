from transformers import pipeline
import pandas as pd
from tqdm import tqdm

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # or "true" if you want parallel tokenization


def get_model():
    model = pipeline(
        "summarization",
        model="google-t5/t5-base",
        device=0
    )
    return model

def summarize_text(summarizer, text):
    # T5 needs "summarize: " prefix
    input_text = "summarize: " + text.strip()
    try:
        summary = summarizer(
            input_text,
            truncation=True,
            max_length=512,
            min_length=20,
            do_sample=False
        )[0]["summary_text"]
    except Exception as e:
        summary = f"[ERROR] {e}"
    return summary

def main():
    summarizer = get_model()
    df = pd.read_csv("test_cnn_daily_mail_data/test_summarization_batch_1.csv")

    batch_size = 50
    all_summaries = []

    for i in range(0, len(df), batch_size):
        print(f"Processing batch {i // batch_size + 1}...")
        batch_df = df.iloc[i:i+batch_size].copy()

        for text in tqdm(batch_df["article"], desc=f"Summarizing batch {i // batch_size + 1}"):
            summary = summarize_text(summarizer, text)
            all_summaries.append(summary)

    df["generated_summary"] = all_summaries
    print(df)
    df.to_csv("t5_summarization_1.csv", index=False)

if __name__ == "__main__":
    main()