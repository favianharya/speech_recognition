import torch

from utils.evaluation import Evaluation


def main():
    eval = Evaluation(
        file_path="pegasus", 
        token_id="test")
    
    eval.rouge(refference_col="highlights", generated_summary="generated_summary")
    eval.bert_score(refference_col="highlights", generated_summary="generated_summary")
    eval.meteor(refference_col="highlights", generated_summary="generated_summary")

if __name__ == "__main__":
    main()


 