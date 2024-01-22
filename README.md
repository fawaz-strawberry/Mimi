# Fancy Schmancy Super AGI
Mimi is a framework for generating continually better AI models. It consists of several components:

## Config YAMLS
Config YAMLs are used to provide necessary information for any model to be used by the training files. These YAMLs contain information such as the number of context, number of predictions, number of attention mechanisms, and more.


## INPUTS
- Dataset: The set of data you want to train on
- DatasetLoader: How the data will be accessed and then obtained
- Preprocessing: Run any preprocessing steps such as tokenization of the examples, or pre-embeddings
- Model: The model to actually process the data
- Loss: The method to calculate the loss based on the expected outcome
- Optimizer: The optimizer to step in the right direction, this is influenced by the learning rate.


## pre_training.py
`pre_training.py` is a file that performs the bare minimum training of a next token predictor on a given dataset. It takes arguments such as the model to train and the dataset to train on.

## fine_tuning.py
`fine_tuning.py` is used to fine-tune a model on a smaller dataset. It employs similar methods as pre-training but is better designed to generate responses, such as with GPT.

## generate.py
`generate.py` takes an input in the form of a string or file, along with the model to use, and outputs a response within a maximum length.

## models/model_x.py
`models/model_x.py` is a folder that holds different architectural models that can be trained and tested. The purpose of this folder is to make it easy to build and experiment with different architectures.

## artifacts/artifact_x.xxx
`artifacts/artifact_x.xxx` is a folder that holds the artifacts required to run a model. The configuration YAMLs specify which model goes where. The training Python scripts should update the configs to represent the latest checkpoints of the models.