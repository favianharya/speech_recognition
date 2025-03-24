import nltk
from rouge_score import rouge_scorer, scoring
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt_tab')

def rouge_eval(summary, original_text):

    original_text = sent_tokenize(original_text)
    summary = sent_tokenize(summary)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for i, summary_sentence in enumerate(summary):
        scores = scorer.score(original_text[i], summary_sentence)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()

    return result['rougeL'].mid