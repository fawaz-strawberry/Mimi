import time
import yaml
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

# Function to dynamically import a class from a module
def dynamic_import(module, class_name):
    module = __import__(module, fromlist=[class_name])
    return getattr(module, class_name)

def partitioned_dataset(filename, partitionAmt=1000000):
    print(f"Splitting dataset into sizes of {partitionAmt}")
    with open(filename) as f:
        while chunk := f.read(partitionAmt):
            yield chunk

# Create a test input function to test the model and monitor outputs
@torch.no_grad()
def test_input(context_len):
    # Print a sample of the next 30 tokens output based on some input
    input_text = "This is a test"
    sample_input = tokenizer.tokenize([input_text]).to(device)
    print("Sample Output: " + input_text, end="")
    for i in range(100):
        output = model(x)
        next_token = torch.argmax(output[0, -1, :])
        # Untokenize the next token
        nt = tokenizer.untokenize([next_token])
        print(f"{nt}", end="")
        # Add the next token to the sample input but remove the first token if the sample input would be longer than the context length
        if sample_input.shape[1] >= context_len:
            sample_input = sample_input[:, 1:]
        sample_input = torch.cat((sample_input, torch.tensor([[next_token]]).to(device)), dim=1)

    print("\n")


# Get the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

myArgs = sys.argv
assert (len(myArgs) >= 2 and os.path.isfile(myArgs[1])), "Please enter a valid config yaml path"
myYamlPath = myArgs[1]

with open(myYamlPath, 'r') as f:
    config = yaml.safe_load(f)
    print(f"Loading model: {config['MODEL']}")

# Create our model and validate it
model_class_name = config['MODEL_SETUP']['ARCHITECTURE']
if(model_class_name == "None"):
    print("No model specified, skipping model import")
    model_class = None
else:
    model_class = dynamic_import(f'Architecture.{model_class_name}', model_class_name)

# Create our tokenizer and validate it
tokenizer_class_name = config['TOKENIZER']
if(tokenizer_class_name == "None"):
    print("No tokenizer specified, skipping tokenizer import")
    tokenizer_class = None
else:
    tokenizer_class = dynamic_import(f'Tokenizers.{tokenizer_class_name}', tokenizer_class_name)

# Create DatasetLoader and validate it
datasetloader_class_name = config['DATASET_LOADER']
if(datasetloader_class_name == "None"):
    print("No dataset loader specified, skipping dataset loader import")
    datasetloader_class = None
else:
    datasetloader_class = dynamic_import(f'DatasetLoaders.{datasetloader_class_name}', datasetloader_class_name)

potential_tokenizer_file = "Pickels/" + config['MODEL'] + "/token_to_id.pkl"
tokenizer = tokenizer_class(potential_tokenizer_file)

if(tokenizer.verify_tokenizer()):
    print(f"Tokenizer contains a vocab size of {tokenizer.getVocabSize()}")
else:
    print(f"Fitting tokenizer onto YAML specified dataset: {config['DATASET']}")
    tokenizer.fit(config['MODEL'], config['DATASET'])
    print(f"Tokenizer generated, with a vocab size of {tokenizer.getVocabSize()}")

model_params = config['MODEL_PARAMS']
model = model_class(model_params, tokenizer.getVocabSize(), device)

dataset = datasetloader_class(config["MODEL"], "shakespeare.txt", False, model_params['CONTEXT_LENGTH'], tokenizer)
dataloader = DataLoader(dataset, batch_size=model_params["BATCH_SIZE"], shuffle=True)
# exit()


# Run a sample through the model.
sample_text = "This is a test"
tokenized_batch = tokenizer.tokenize([sample_text])
tokenized_batch = tokenized_batch.to(device)

print(f"Sample batch shape: {tokenized_batch.shape}")
print(tokenized_batch)

untokenized_batch = tokenizer.untokenize(tokenized_batch)
print(untokenized_batch)

learning_rate = config["TRAINING_PARAMS"]["LEARNING_RATE"]
learning_rate_decay = config["TRAINING_PARAMS"]["LR_DECAY"]


# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

MODEL_NAME = config['MODEL']
train_count = 0
train_start_time = int(time.time())

print(f"Training model: {MODEL_NAME}")
print(f"Training for {config['TRAINING_PARAMS']['NUM_EPOCHS']} epochs")
# exit()

for epoch in range(config["TRAINING_PARAMS"]["NUM_EPOCHS"]):
    print(f"Epoch {epoch + 1}")
    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        prediction = model(x)

        # Calculate loss
        loss = F.cross_entropy(prediction.view(-1, tokenizer.num_tokens), y.view(-1))

        # Backpropagate
        loss.backward()

        # Update parameters
        optimizer.step()

        # Zero out the gradients
        optimizer.zero_grad()

        # Print loss
        if train_count % 100 == 0:
            learning_rate = learning_rate * ( learning_rate_decay ** (train_count // 1000))
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            print(f"Train Loss: {loss} at {train_count} with lr={learning_rate}") 
        if train_count % 1000 == 0 and train_count != 0:
            # print(f"Train Loss: {estimateLoss(train_dataloader)} and Val Loss: {estimateLoss(val_dataloader)} at {train_count} with lr={learning_rate}")
            test_input(model_params["CONTEXT_LENGTH"])
            os.makedirs("Checkpoints/" + MODEL_NAME, exist_ok=True)
            torch.save(model.state_dict(), f"Checkpoints/{MODEL_NAME}/{MODEL_NAME}_{train_start_time}_{train_count}.pt")

        train_count += 1