import re
from .AbstractTokenizer import AbstractTokenizer

class SimpleTokenizer(AbstractTokenizer):
    def __init__(self):
        super().__init__() 
        self.token_to_id = {"[PAD]": 0, "[UNK]": 1}
        self.num_tokens = 2
    
    # Fit the tokenizer onto the dataset
    def fit(self, texts):
        for text in texts:
            for word in re.findall(r'\w+', text):
                if word not in self.token_to_id:
                    self.token_to_id[word] = self.num_tokens
                    self.num_tokens += 1

    def tokenize(self, text):
        return_list = []
        for word in re.findall(r'\w+', text):
            return_list.append(self.token_to_id.get(word, 1))
        return return_list
    
    def untokenize(self, token_list):
        return_string = ""
        for token in token_list:
            return_string += list(self.token_to_id.keys())[list(self.token_to_id.values()).index(token)] + " "
        return return_string