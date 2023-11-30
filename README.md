Fancy Schmancy Super AGI

Mimi is not just a single AI, but rather a process to generate continually better AI models

Config JSONs will be used to provide the neccessary information for any model to be taken in
by the training files as it should provide information such as
num_context
num_predictions
num_attention_mechs
.... so on as I learn

pre_training.py will be a file to do the bare minimum and train a next token predictor on any
given dataset. Arguments should include the model to train and the dataset to train on.

fine_tuning.py will be used to fine tune a model on a certain (typically smaller dataset) through
similar methods as pretraining but better designed to get responses out like GPT

generate.py will take in an input in the form of a string or file, along with the model to use,
and output a response within a max length.

models/model_x.py will be a folder to hold the hopefully many different architectural models that 
I plan on training on. The point of this is to make it as easy as possible to build different archs
and train/test them.

artifacts/artifact_x.xxx will be a folder to hold the artifacts in order to get a model to run, this should
be handled by the config JSONs as to which model goes where. Our training python scripts should modify the
configs to represent the latest ckpts of such models 