from transformers import pipeline
import pandas as pd
import asyncio
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # or "true" if you want parallel tokenization


def get_model():
    model = pipeline(
        "summarization",
        model="google/pegasus-large", # download model: google-t5/t5-base, 
        tokenizer="google/pegasus-xsum",
        use_fast=False,
        device="cpu"
    )
    return model

# def get_model():
#     model = pipeline(
#         "summarization",
#         model="google-t5/t5-base",
#         device="cpu",
#     )
#     return model

# Async wrapper for the summarization call
def summarize_text(summarizer, text):
    try:
        summary = summarizer(text, max_length=512, min_length=20, do_sample=False)[0]["summary_text"]
    except Exception as e:
        summary = f"[ERROR] {e}"
    return summary

async def process_summaries(df, summarizer, max_workers=4):
    summaries = []
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            loop.run_in_executor(executor, summarize_text, summarizer, text)
            for text in df["article"]
        ]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Summarizing"):
            summaries.append(await f)
    return summaries

def main():
    summarizer = get_model()
    df = pd.read_csv("test_cnn_daily_mail_data/test_summarization_batch_1.csv")
    summaries = asyncio.run(process_summaries(df, summarizer))

    df["generated_summary"] = summaries

    print(df)
    df.to_csv("summarization_1.csv", index=False)

if __name__ == "__main__":
    main()