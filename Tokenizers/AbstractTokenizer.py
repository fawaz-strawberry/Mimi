from abc import ABC, abstractmethod

class AbstractTokenizer(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, text):
        """Fit the tokenizer on the text data."""
        pass

    @abstractmethod
    def tokenize(self, text):
        """Tokenize a string into tokens."""
        pass

    @abstractmethod
    def untokenize(self, tokens):
        """Untokenize a list of tokens into a string."""
        pass
