from utils import logger

from evaluate import load
from rouge_score import rouge_scorer
from rouge_score import scoring

import pandas as pd
import torch
from huggingface_hub import login
import os
import numpy as np

class Evaluation:
    def __init__(
            self,
            file_path: str,
            token_id: str
    ):
        self.file_path = file_path
        self.token_id = token_id

    def _login_hg(self):
        login(token=self.token_id)

    def _read_file(self):
        df = pd.DataFrame()

        if os.path.isdir(self.file_path) is True:
            logger.info("This is a folder.")
            csv_files = [f for f in os.listdir(self.file_path) if f.endswith('.csv')]
            df_list = [pd.read_csv(os.path.join(self.file_path, file)) for file in csv_files]
            df = pd.concat(df_list, ignore_index=True)
        elif self.file_path.lower().endswith(".csv"):
            logger.info("Confirmed: it's a CSV file.")
            df = pd.read_csv(self.file_path)
        else:
            logger.error("Path does not exist or is not a directory.")
    
        return df
    
    def rouge(self, 
            refference_col: str, 
            generated_summary: str
        ) -> str:
        df = self._read_file()
        if df is None:
            logger.error("No data to process in ROUGE.")
            return
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()

        # Store per-row F1 scores to calculate mean and median manually
        rouge_f1_scores = {
            "rouge1": [],
            "rouge2": [],
            "rougeL": []
        }

        logger.info(f"Starting evaluation with rouge1, rouge2, rougeL..")
        for idx, row in df.iterrows():
            scores = scorer.score(row[refference_col], row[generated_summary])
            aggregator.add_scores(scores)

            for key in rouge_f1_scores:
                rouge_f1_scores[key].append(scores[key].fmeasure)

        result = aggregator.aggregate()
        logger.info("Done.")

        for key in result:
            print(f"{key}:")
            print(f"  Precision:   {result[key].mid.precision:.4f}")
            print(f"  Recall:      {result[key].mid.recall:.4f}")
            print(f"  F1:          {result[key].mid.fmeasure:.4f}")
            print(f"  F1 - Mean:   {np.mean(rouge_f1_scores[key]):.4f}")
            print(f"  F1 - Median: {np.median(rouge_f1_scores[key]):.4f}")
            
    def bert_score(self,
            refference_col: str, 
            generated_summary: str,
            model_type: str = "distilbert-base-uncased",
            batch_size: int = 8
        ):
        self._login_hg()
        logger.info("Logged into HuggingFace. Ready for BERTScore.")

        df = self._read_file()
        if df is None:
            logger.error("No data to process in BERTscore.")
            return

        if torch.backends.mps.is_available():
            torch_device = "mps"
        else:
            torch_device = "cpu"
        logger.info(f"This BERTscore evaluation using {torch_device}.")
        
        logger.info("Starting evaluation with BERTscore..")
        bertscore = load("bertscore")
        results = bertscore.compute(
            predictions=df[generated_summary].tolist(),
            references=df[refference_col].tolist(),
            lang="en",
            model_type=model_type,  
            batch_size=batch_size,
            device=torch_device
        )
        logger.info("Done.")
        
        print(f"BERTscore ({model_type})")
        print(f"  Precision: {sum(results['precision']) / len(results['precision']):.4f}")
        print(f"  Recall:    {sum(results['recall']) / len(results['recall']):.4f}")
        print(f"  F1:        {sum(results['f1']) / len(results['f1']):.4f}")

    def meteor(self,
            refference_col: str, 
            generated_summary: str,
        ):
        self._login_hg()
        logger.info("Logged into HuggingFace. Ready for BERTScore.")

        df = self._read_file()
        if df is None:
            logger.error("No data to process in BERTscore.")
            return
        
        meteor = load("meteor")

        references = df[refference_col].tolist()
        predictions = df[generated_summary].tolist()

        results = meteor.compute(predictions=predictions, references=references)
        df["meteor"] = [results["meteor"]] * len(df)

        mean_meteor = df["meteor"].mean()
        median_meteor = df["meteor"].median()

        print(f"Mean METEOR score:   {mean_meteor:.4f}")
        print(f"Median METEOR score: {median_meteor:.4f}")