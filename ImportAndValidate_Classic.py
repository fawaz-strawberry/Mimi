from Tokenizers.AbstractTokenizer import AbstractTokenizer
from DatasetLoaders.AbstractDatasetLoader import AbstractDatasetLoader
from Embeddings.AbstractEmbedding import AbstractEmbedding
from Architecture.AbstractArchitecture import AbstractArchitecture
import yaml

# Function to dynamically import a class from a module
def dynamic_import(module, class_name):
    module = __import__(module, fromlist=[class_name])
    return getattr(module, class_name)

# Load configuration
yaml_file = 'ModelYAMLs/CharacterLevelTransformerConfig.yaml'
with open(yaml_file, 'r') as file:
    config = yaml.safe_load(file)
    print(f"Loading Model: {config['model_identifier']}")
    
# Get the tokenizer class from the configuration
tokenizer_class_name = config['tokenizer']['type']
dataloader_class_name = config['dataloader']['type']
embedding_0_class_name = config['embedding_0']['type']
embedding_1_class_name = config['embedding_1']['type']
architecture_class_name = config['architecture']['type']


# Get the pretrained tokenizers if availalbe
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

# Instantiate some default variables
MODEL_NAME = config["model_identifier"]
BATCH_SIZE = config["batch_size"]
CONTEXT_LEN = config["context_length"]
EMBEDDING_SIZE = config["embedding_size"]
HEADS = config["num_heads"]
DROPOUT = config["dropout"]
LEARNING_RATE = config["learning_rate"]
LR_DECAY = config["lr_decay"]
EPOCHS = config["num_epochs"]
IMPORTED_DATSET = config["dataset"]
TRAIN_SPLIT = config["train_split"]
VAL_SPLIT = config["val_split"]
