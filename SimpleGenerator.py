import yaml
from Tokenizers.AbstractTokenizer import AbstractTokenizer
from DatasetLoaders.AbstractDatasetLoader import AbstractDatasetLoader
from Embeddings.AbstractEmbedding import AbstractEmbedding
from Architecture.AbstractArchitecture import AbstractArchitecture
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

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

tokenizer = TokenizerClass()
dataset = DatasetLoaderClass('datasets/iac_mini.txt', is_multi_line=False, chunk_size=context_len)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


texts = dataset.chunks
tokenizer.fit(texts)

position_embedding = Embedding0Class(embeding_size, context_len)
semantic_embedding = Embedding1Class(tokenizer.num_tokens, embeding_size)
model = ArchitectureClass(embeding_size, tokenizer.num_tokens, heads, dropout)

text = "Hello, World!"
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
untokenized = tokenizer.untokenize(tokens)
print(f"Untokenized: {untokenized}")

for batch in dataloader:
    
    print(batch[:1])

    # Tokenize the batch
    x = tokenizer.tokenize(batch[0][0])
    y = tokenizer.tokenize(batch[0][1])

    # Create a tensor from the batch but also add a dimension for the batch size
    batch = torch.tensor(x).unsqueeze(0)

    print(batch.shape)

    # Add padding to the batch
    batch = torch.nn.functional.pad(batch, (0, context_len - batch.shape[1]), 'constant', 0)

    # Run through the embeddings
    position_embeddings = position_embedding(batch)
    semantic_embeddings = semantic_embedding(batch)

    # Add the embeddings together
    embeddings = position_embeddings + semantic_embeddings

    # Run through the model
    output = model(embeddings)
    print(output.shape)
    print(output)
    # Apply softmax on the vocab_size dimension
    probabilities = F.softmax(output, dim=-1)
    print(probabilities.shape)
    
    predicted_tokens = torch.argmax(probabilities, dim=-1)
    print(predicted_tokens.shape)
    print(predicted_tokens)
    # Convert the predicted tokens to text after converting the tensor to a list
    predicted_tokens = predicted_tokens.squeeze(0).tolist()
    predicted_text = tokenizer.untokenize(predicted_tokens)
    print(predicted_text)

    break