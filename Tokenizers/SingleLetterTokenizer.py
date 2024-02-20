import os
import pickle
import torch
import time
from .AbstractTokenizer import AbstractTokenizer

class SingleLetterTokenizer(AbstractTokenizer):
    def __init__(self, file_path=None):
        super().__init__()
        self.isFit = False
        print(f"Checking for tokenizer file: {file_path}")
        if file_path and os.path.isfile(file_path):
            print(f"Found 🥒")
            tokens_to_id = pickle.load(open(file_path, 'rb'))
            self.token_to_id = tokens_to_id
            self.id_to_token = {id: token for token, id in self.token_to_id.items()}
            self.num_tokens = len(tokens_to_id)
            self.isTokenized = True
            self.isFit = True
        else:
            self.token_to_id = {"[PAD]": 0, "[UNK]": 1}
            self.id_to_token = {"0": "[PAD]", "1": "[UNK]"}
            self.num_tokens = 2
            self.isTokenized = False

        print("Tokenizer initialized")

    def fit(self, model_name, filename):
        print(f"Loading data from text file 📄: {filename}")
        with open(filename, 'r') as f:
            texts = f.read()

        # Print sample data
        print(f"Sample Texts: {texts[:100]}")

        # Go character by character
        for char in texts:
            if char not in self.token_to_id:
                self.token_to_id[char] = self.num_tokens
                self.id_to_token[int(self.num_tokens)] = char
                self.num_tokens += 1

        # Sample Tokens
        print(f"Sample Tokens: {list(self.token_to_id.keys())[:10]}")

        # Save tokenizer to file for later use with date generation
        os.makedirs("Pickels/" + model_name, exist_ok=True)
        with open('Pickels/' + model_name + '/token_to_id.pkl', 'wb') as f:
            pickle.dump(self.token_to_id, f)
        
        self.isFit = True
        self.isTokenized = True

    
    def verify_tokenizer(self):
        try:
            assert self.isFit, "Tokenizer is not fit"
            assert self.isTokenized, "Tokenizer is not tokenized"
            # print("Tokenizer verified")
            return True
        except AssertionError as e:
            print(e)
            return False
  
    def tokenize(self, list_of_texts, return_tensor=True):
        if not self.verify_tokenizer():
            return None
        else:
            return_list = []
            for text in list_of_texts:
                current_list = []
                for char in text:
                    current_list.append(self.token_to_id.get(char))
                return_list.append(current_list)

            if return_tensor:
                return torch.tensor(return_list, dtype=torch.long)
            else:
                return return_list

    # Expect token_list to come in as a tensor of shape [batch_size, context_len]
    # Return a list of strings
    def untokenize(self, token_tensor):
        if not self.verify_tokenizer():
            return None
        else:
            token_list = token_tensor.tolist()
            return ["".join([self.id_to_token[token] for token in tokens]) for tokens in token_list]

    def getVocabSize(self):
        return self.num_tokens