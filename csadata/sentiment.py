from abc import ABC, abstractmethod
from numpy import ndarray
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


class SentimentAnalyzer(ABC):
    @abstractmethod
    def classify_sentiment(self, text) -> ndarray:
        pass


class RoBERTaSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self):
        super().__init__()

        model = "cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)

    def classify_sentiment(self, text) -> ndarray:
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        return scores
