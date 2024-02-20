import yaml
import os
import sys
import torch

# Function to dynamically import a class from a module
def dynamic_import(module, class_name):
    module = __import__(module, fromlist=[class_name])
    return getattr(module, class_name)

def partitioned_dataset(filename, partitionAmt=1000000):
    print(f"Splitting dataset into sizes of {partitionAmt}")
    with open(filename) as f:
        while chunk := f.read(partitionAmt):
            yield chunk

# Create our tokenizer and validate it

tokenizer_class_name = config['TOKENIZER']
if(tokenizer_class_name == "None"):
    print("No tokenizer specified, skipping tokenizer import")
    tokenizer_class = None
else:
    tokenizer_class = dynamic_import(f'Tokenizers.{tokenizer_class_name}', tokenizer_class_name)

tokenizer = tokenizer_class()
if(tokenizer.verify_tokenizer()):
    print("Tokenizer verified")
else:
    print("Tokenizer has not been fit... please fit the tokenizer")
    exit()