from typing import Callable, Optional
from numpy import ndarray
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


def _default_preproc(text):
    new_text = []

    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


class SentimentAnalyzer:
    def __init__(self, preprocessor: Optional[Callable[[str], str]] = _default_preproc):
        self.preprocessor: Callable[[str], str] = preprocessor
        model = "cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.save_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.model.save_pretrained(model)

    def classify_sentiment(self, text) -> ndarray:
        if self.preprocessor is not None:
            text = self.preprocessor(text)

        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        return scores
