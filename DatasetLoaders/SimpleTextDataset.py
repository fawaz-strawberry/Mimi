import os
import torch
import pickle
from torch.utils.data import Dataset
from .AbstractDatasetLoader import AbstractDatasetLoader

class SimpleTextDataset(AbstractDatasetLoader, Dataset):
    def __init__(self, model_name, filename, is_multi_line=True, context_len=1024, tokenizer=None):
        
        self.tokenizer = tokenizer
        self.context_len = context_len
        
        self.data = []
        self.chunks = []

        file_path = 'Pickels/' + model_name + '/tokenized_data.pkl'
        if file_path.endswith('.pkl') and os.path.isfile(file_path):
            print("Loading tokenized data from pickle file 🥒")
            self.chunks = pickle.load(open(file_path, 'rb'))
        else:
        # Read the file and add everything to the data list in one string into self.data
        # Then run the tokenizer on the data converting it to a list of tokens in self.chunks
        # with chunk size of chunk_size
            print("Loading data from text file 📄")
            with open('Datasets/' + filename, 'r') as f:
                if is_multi_line:
                    self.data = f.readlines()
                else:
                    self.data = f.read()

            # Print sample data
            print(f"Sample Data: {self.data[:10]}")

            tokenized_characters = (tokenizer.tokenize([self.data], return_tensor=False))[0]

            print(f"Sample Tokens: {tokenized_characters[:10]}")

            # exit()

            # Split the tokenized data into chunks of size context_len
            for i in range(0, len(tokenized_characters), self.context_len + 1):
                my_chunk = tokenized_characters[i:i + self.context_len + 1]
                if len(my_chunk) == self.context_len + 1:
                    self.chunks.append(tokenized_characters[i:i + self.context_len + 1])

            print(f"Sample Chunk: {self.chunks[0]}")
            print(self.chunks[:10])
            print(f"Lenght of chunks: {len(self.chunks)}")

            # Save the tokenized data to a file
            os.makedirs("Pickels/" + model_name, exist_ok=True)   
            with open(file_path, 'wb') as f:
                pickle.dump(self.chunks, f)

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.chunks[idx][:-1], dtype=torch.long)
        y = torch.tensor(self.chunks[idx][1:], dtype=torch.long)

        return x, y