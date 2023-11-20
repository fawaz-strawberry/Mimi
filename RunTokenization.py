import pickle

from .Tokenizers import SimpleTokenizer



# Tokenize your dataset (assuming you have a tokenize function)
tokenized_data = tokenize_my_dataset(dataset)

# Save the tokenized data to a file
with open('tokenized_data.pkl', 'wb') as f:
    pickle.dump(tokenized_data, f)