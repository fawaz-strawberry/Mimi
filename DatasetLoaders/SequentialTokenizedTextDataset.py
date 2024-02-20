import os
import torch
from torch.utils.data import Dataset

class SequentialTokenizedTextDataset(Dataset):
    def __init__(self, directory, tokenizer, context_len=1024):
        self.directory = directory
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.file_list = self._get_file_list(directory)
        self.current_file_index = -1
        self.current_tokens = []
        self._load_next_file()

    def _get_file_list(self, directory):
        """Returns a list of file paths contained in the directory."""
        return [os.path.join(directory, filename) for filename in os.listdir(directory)]

    def _load_next_file(self):
        """Loads the next file from the file list and tokenizes its content."""
        self.current_file_index += 1
        if self.current_file_index < len(self.file_list):
            with open(self.file_list[self.current_file_index], 'r', encoding='utf-8') as file:
                text = file.read()
                self.current_tokens = self.tokenizer.tokenize([text])
        else:
            self.current_tokens = []  # No more files to process

    def __len__(self):
        # This can be adjusted to return a more precise length if needed
        return len(self.current_tokens) // self.context_len

    def __getitem__(self, idx):
        if idx*self.context_len >= len(self.current_tokens) and self.current_file_index + 1 < len(self.file_list):
            self._load_next_file()
            return self.__getitem__(0)  # Restart idx for the new file
        else:
            start = idx * self.context_len
            end = start + self.context_len
            tokens = self.current_tokens[start:end]
            x = torch.tensor(tokens[:-1], dtype=torch.long)
            y = torch.tensor(tokens[1:], dtype=torch.long)
            return x, y
