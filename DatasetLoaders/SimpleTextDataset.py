import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from .AbstractDatasetLoader import AbstractDatasetLoader

class SimpleTextDataset(AbstractDatasetLoader, Dataset):
    def __init__(self, filename, is_multi_line=True, context_len=1024, tokenizer=None):
        
        self.tokenizer = tokenizer
        self.context_len = context_len
        
        self.data = []
        self.chunks = []

        # check if filetype is a pickle file and if so, load it as a list into self.chunks
        if filename.endswith('.pkl'):
            print("Loading tokenized data from pickle file 🥒")
            self.chunks = pickle.load(open(filename, 'rb'))
        else:
        # Read the file and add everything to the data list in one string into self.data
        # Then run the tokenizer on the data converting it to a list of tokens in self.chunks
        # with chunk size of chunk_size
            print("Loading data from text file 📄")
            with open(filename, 'r') as f:
                if is_multi_line:
                    self.data = f.readlines()
                else:
                    self.data = f.read()

            # Print sample data
            print(f"Sample Data: {self.data[:100]}")

            tokenizer.fit(self.data)
            tokenized_characters = tokenizer.tokenize(self.data)

            # Split the tokenized data into chunks of size context_len
            for i in range(0, len(tokenized_characters), self.context_len + 1):
                self.chunks.append(tokenized_characters[i:i + self.context_len + 1])


            # Save the tokenized data to a file
            with open('tokenized_data.pkl', 'wb') as f:
                pickle.dump(self.chunks, f)

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.chunks[idx][:-1], dtype=torch.long)
        y = torch.tensor(self.chunks[idx][1:], dtype=torch.long)

        return x, y