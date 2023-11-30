import re
import pickle
from .AbstractTokenizer import AbstractTokenizer

class SimpleTokenizer(AbstractTokenizer):
    def __init__(self, filename=None):
        super().__init__() 
        if filename:
            tokens_to_id = pickle.load(open(filename, 'rb'))
            self.token_to_id = tokens_to_id
            self.num_tokens = len(tokens_to_id)
            self.isTokenized = True
        else:
            self.token_to_id = {"[PAD]": 0, "[UNK]": 1}
            self.num_tokens = 2
            self.isTokenized = False

        print("Tokenizer initialized")
    
    # Fit the tokenizer onto the dataset
    def fit(self, texts):
        i = 0
        for word in re.findall(r'\w+', texts):
            if word not in self.token_to_id:
                self.token_to_id[word] = self.num_tokens
                self.num_tokens += 1
    
        # Sample Tokens
        print(f"Sample Tokens: {list(self.token_to_id.keys())[:10]}")

        # Save tokenizer to file for later use with date generation
        with open('token_to_id.pkl', 'wb') as f:
            pickle.dump(self.token_to_id, f)

        self.isTokenized = True

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