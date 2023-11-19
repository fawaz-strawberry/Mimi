import yaml
from Tokenizers.AbstractTokenizer import AbstractTokenizer
from DatasetLoaders.AbstractDatasetLoader import AbstractDatasetLoader
import torch
from torch.utils.data import Dataset, DataLoader

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


# Dynamically import the tokenizer class
TokenizerClass = dynamic_import(f'Tokenizers.{tokenizer_class_name}', tokenizer_class_name)
DatasetLoaderClass = dynamic_import(f'DatasetLoaders.{dataloader_class_name}', dataloader_class_name)

# Check if the class is a subclass of AbstractTokenizer
if not issubclass(TokenizerClass, AbstractTokenizer):
    raise ValueError(f"{tokenizer_class_name} is not a valid Tokenizer")

if not issubclass(DatasetLoaderClass, AbstractDatasetLoader):
    raise ValueError(f"{dataloader_class_name} is not a valid DataLoader")

# Instantiate the tokenizer
tokenizer = TokenizerClass()
dataset = DatasetLoaderClass('datasets/iac_mini.txt', is_multi_line=False)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

texts = dataset.chunks
tokenizer.fit(texts)


text = "Hello, World!"
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
untokenized = tokenizer.untokenize(tokens)
print(f"Untokenized: {untokenized}")
