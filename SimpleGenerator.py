import yaml
from Tokenizers.AbstractTokenizer import AbstractTokenizer
from DatasetLoaders.AbstractDatasetLoader import AbstractDatasetLoader
from Embeddings.AbstractEmbedding import AbstractEmbedding
from Architecture.AbstractArchitecture import AbstractArchitecture
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os

PRE_TRAIN = True

# Function to dynamically import a class from a module
def dynamic_import(module, class_name):
    module = __import__(module, fromlist=[class_name])
    return getattr(module, class_name)

# Load configuration
with open('ModelYAMLs/SimpleConfig.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Get the tokenizer class from the configuration
tokenizer_class_name = config['tokenizer']['type']
# tokenizer_params = config['tokenizer'].get('params', {})
dataloader_class_name = config['dataloader']['type']
embedding_0_class_name = config['embedding_0']['type']
embedding_1_class_name = config['embedding_1']['type']
architecture_class_name = config['architecture']['type']

tokenizer_file = config['tokenizer_filename']
tokenized_data = config['tokenized_dataset']

# Dynamically import the tokenizer class
TokenizerClass = dynamic_import(f'Tokenizers.{tokenizer_class_name}', tokenizer_class_name)
DatasetLoaderClass = dynamic_import(f'DatasetLoaders.{dataloader_class_name}', dataloader_class_name)
Embedding0Class = dynamic_import(f'Embeddings.{embedding_0_class_name}', embedding_0_class_name)
Embedding1Class = dynamic_import(f'Embeddings.{embedding_1_class_name}', embedding_1_class_name)
ArchitectureClass = dynamic_import(f'Architecture.{architecture_class_name}', architecture_class_name)

# Check if the class is a subclass of AbstractTokenizer
if not issubclass(TokenizerClass, AbstractTokenizer):
    raise ValueError(f"{tokenizer_class_name} is not a valid Tokenizer")

if not issubclass(DatasetLoaderClass, AbstractDatasetLoader):
    raise ValueError(f"{dataloader_class_name} is not a valid DataLoader")

if not issubclass(Embedding0Class, AbstractEmbedding):
    raise ValueError(f"{embedding_0_class_name} is not a valid Embedding")

if not issubclass(Embedding1Class, AbstractEmbedding):
    raise ValueError(f"{embedding_1_class_name} is not a valid Embedding")

if not issubclass(ArchitectureClass, AbstractArchitecture):
    raise ValueError(f"{architecture_class_name} is not a valid Architecture")

# Instantiate the tokenizer

context_len = 64
embeding_size = 64
heads = 4
dropout = 0.1

# Check if the tokenizer file exists
if tokenizer_file and os.path.isfile(tokenizer_file):
    tokenizer = TokenizerClass(tokenizer_file)
else:
    tokenizer = TokenizerClass()

# Check if the tokenized data file exists
if tokenized_data and os.path.isfile(tokenized_data):
    dataset = DatasetLoaderClass(tokenized_data, is_multi_line=False, context_len=context_len, tokenizer=tokenizer)
else:
    dataset = DatasetLoaderClass('datasets/MEGA_SCRIPT.txt', is_multi_line=False, context_len=context_len, tokenizer=tokenizer)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


position_embedding = Embedding0Class(embeding_size, context_len)
semantic_embedding = Embedding1Class(tokenizer.num_tokens, embeding_size)
model = ArchitectureClass(embeding_size, tokenizer.num_tokens, heads, dropout)

text = "Hello, World!"
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
untokenized = tokenizer.untokenize(tokens)
print(f"Untokenized: {untokenized}")

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_count = 0

# Create training loop
for epoch in range(1):

    for batch in dataloader:

        x, y = batch

        print(f"X shape: {x.shape}")
        print(f"Y shape: {y.shape}")

        semantic_embedding = semantic_embedding(x)

        final_embeddings = position_embedding(semantic_embedding)

        # Run through the model
        output = model(final_embeddings)
        
        if(PRE_TRAIN):
            train_count += 1

            # Calculate loss
            loss = F.cross_entropy(output.view(-1, tokenizer.num_tokens), y.view(-1))
            
            # Backpropagate
            loss.backward()

            # Update parameters
            optimizer.step()

            # Zero out the gradients
            optimizer.zero_grad()

            # Print loss
            if train_count % 100 == 0:
                print(f"Loss: {loss}")

        break

# Print a sample of the next 30 tokens output based on some input
sample_input = torch.tensor([[tokenizer.tokenize('This is a test about your mom')]])

# Print sample input shape
print(f"Sample input shape: {sample_input.shape}")

for i in range(30):
    x = semantic_embedding(sample_input)
    x = position_embedding(x)
    output = model(x)
    next_token = torch.argmax(output[0, -1, :])
    sample_input = torch.cat((sample_input, next_token.view(1, 1)), dim=1)
