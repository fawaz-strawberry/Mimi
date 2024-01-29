import yaml
import os
import sys

from Tokenizers.AbstractTokenizer import AbstractTokenizer
from DatasetLoaders.AbstractDatasetLoader import AbstractDatasetLoader
from Embeddings.AbstractEmbedding import AbstractEmbedding
from Architecture.AbstractArchitecture import AbstractArchitecture

# Function to dynamically import a class from a module
def dynamic_import(module, class_name):
    module = __import__(module, fromlist=[class_name])
    return getattr(module, class_name)

def partitioned_dataset(filename, partitionAmt=1000000):
    print(f"Splitting dataset into sizes of {partitionAmt}")
    with open(filename) as f:
        while chunk := f.read(partitionAmt):
            yield chunk

myArgs = sys.argv
assert (len(myArgs) >= 2 and os.path.isfile(myArgs[1])), "Please enter a valid config yaml path"
myYamlPath = myArgs[1]

with open(myYamlPath, 'r') as f:
    config = yaml.safe_load(f)
    print(f"Loading model: {config['MODEL']}")

# Create our model and validate it
model_class_name = config['MODEL_SETUP']['ARCHITECTURE']
model_class = dynamic_import(f'Architecture.{model_class_name}', model_class_name)
# model = model_class(config)

# For epoch
for epoch in range(config["TRAINING_PARAMS"]["NUM_EPOCHS"]):

    # Import Dataset
    datasetPath = "Datasets/" + config['DATASET']
    stats = os.stat(datasetPath)
    
    # Breakup Dataset Depending on size
    splitDataset = partitioned_dataset(datasetPath, int(stats[6] / 5.0))
    
    # For each portion
    for chunk in splitDataset:
        print(len(chunk))
        print(chunk[:100])
        print("\n\n")

        # LOSER IDEOLOGY --> Tokenize Portioned Dataset
        # TURN EVERYTHING INTO BYTES AND PREDICT BYTES!!!!
        # Convert chunk into bytes

        chunk = chunk.encode('utf-8')

        # Pull in byte tokenizer?  

        # Create a dataset loader based on the PD
    
        # For each batch size of examples

            # Run data through model

            # Calculate loss
    
            # Optimize
    
            # If counter hit, run approximation test for Train and Val Losses
            # Update learning rate
            # Save values into an array to be plotted later
