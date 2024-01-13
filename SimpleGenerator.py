import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import time
from ImportAndValidate_Classic import TokenizerClass, DatasetLoaderClass, Embedding0Class, \
    Embedding1Class, ArchitectureClass, tokenizer_file, tokenized_data, BATCH_SIZE, CONTEXT_LEN, \
    EMBEDDING_SIZE, HEADS, DROPOUT, MODEL_NAME


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("My device is: " + str(device))

# Check if the tokenizer file exists
tokenizer_file = "Pickels/" + MODEL_NAME + "/" + tokenizer_file
if tokenizer_file and os.path.isfile(tokenizer_file):
    tokenizer = TokenizerClass(tokenizer_file)
else:
    tokenizer = TokenizerClass() 

# Check if the tokenized data file exists
if tokenized_data and os.path.isfile(tokenized_data):
    dataset = DatasetLoaderClass(MODEL_NAME, tokenized_data, is_multi_line=False, context_len=CONTEXT_LEN, tokenizer=tokenizer)
else:
    dataset = DatasetLoaderClass(MODEL_NAME, 'shakespeare.txt', is_multi_line=False, context_len=CONTEXT_LEN, tokenizer=tokenizer)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


position_embedding = Embedding0Class(BATCH_SIZE, EMBEDDING_SIZE, CONTEXT_LEN)
semantic_embedding = Embedding1Class(tokenizer.num_tokens, EMBEDDING_SIZE, device)
model = ArchitectureClass(EMBEDDING_SIZE, tokenizer.num_tokens, HEADS, DROPOUT, device)

model = model.to(device)
position_embedding = position_embedding.to(device)
semantic_embedding = semantic_embedding.to(device)

text = "Hello, World!"
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
untokenized = tokenizer.untokenize(tokens)
print(f"Untokenized: {untokenized}")

def test_input():
    # Print a sample of the next 30 tokens output based on some input
    sample_input = torch.tensor([tokenizer.tokenize('This is a test')]).to(device)


    for i in range(30):
        x = semantic_embedding(sample_input)
        x = position_embedding(x)
        output = model(x)
        next_token = torch.argmax(output[0, -1, :])
        # Untokenize the next token
        nt = tokenizer.untokenize([next_token])
        print(f"{nt}", end=" ")
        # Add the next token to the sample input but remove the first token if the sample input would be longer than the context length
        if sample_input.shape[1] >= CONTEXT_LEN:
            sample_input = sample_input[:, 1:]
        sample_input = torch.cat((sample_input, torch.tensor([[next_token]]).to(device)), dim=1)

    print("\n")

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train_count = 0

train_start_time = int(time.time())
# Create training loop
for epoch in range(10):

    for batch in dataloader:

        x, y = batch

        x = x.to(device)
        y = y.to(device)

        x = semantic_embedding(x)
        x = position_embedding(x)
        output = model(x)

        # Calculate loss
        loss = F.cross_entropy(output.view(-1, tokenizer.num_tokens), y.view(-1))

        # Backpropagate
        loss.backward()

        # Update parameters
        optimizer.step()

        # Zero out the gradients
        optimizer.zero_grad()

        # Print loss
        if train_count % 10 == 0:
            print(f"Loss: {loss} at {train_count} with lr={0.0001/((train_count/100)+1)}")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001/((train_count/100)+1))
            test_input()
        if train_count % 100 == 0 and train_count != 0:
            os.makedirs("Checkpoints/" + MODEL_NAME, exist_ok=True)
            torch.save(model.state_dict(), f"Checkpoints/{MODEL_NAME}/{MODEL_NAME}_{train_start_time}_{train_count}.pt")

        train_count += 1