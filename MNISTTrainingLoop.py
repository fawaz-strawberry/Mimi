import torch
import torch.nn.functional as F

from Architecture.RawMnistSolver import RawMnistSolver
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_path = "/home/fawaz/Documents/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte"
labels_path = "/home/fawaz/Documents/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte"

# Show a sample of the data --------------------------------------------
example_val = 42

imagefile = data_path
labelfile = labels_path
imagearray = idx2numpy.convert_from_file(imagefile)
labelsarray = idx2numpy.convert_from_file(labelfile)

# print(imagearray[example_val])
# print(labelsarray[example_val])

# plt.imshow(imagearray[example_val], cmap=plt.cm.binary)
# plt.show()

# 10000 examples

BATCH_SIZE = 64


from torch.utils.data import Dataset, DataLoader

class MNIST_DATASET(Dataset):
    def __init__(self, dataset, labels):
        self.chunks = dataset
        self.labels = labels

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.chunks[idx], dtype=torch.float)
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return x, y
    
dataset = MNIST_DATASET(imagearray, labelsarray)
dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True)

model = RawMnistSolver(len(imagearray[0]) * len(imagearray[0][0]), 10)
model = model.float()

optim = torch.optim.Adam(model.parameters(), lr=.00005)

train_count = 0

for epoch in range(2):
    for batch in dataloader:

        # Split the input and the labels
        x, y = batch
        
        # Run the input through the model
        output = model.forward(x)

        # print(f"Output Shape: {output.shape} --- Labels Shape: {y.shape}")

        # calculate the loss
        loss = F.cross_entropy(output, y.view(-1))

        # backpropagate
        loss.backward()

        optim.step()

        optim.zero_grad()

        # Print loss
        if train_count % 10 == 0:
            print(f"Loss: {loss} at {train_count} with lr={0.00005/((train_count/100)+1)}")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00005/((train_count/100)+1))
        if train_count % 100 == 0 and train_count != 0:
            torch.save(model.state_dict(), f"models/mnist/model_{train_count}.pt")

        train_count += 1


# Test the model ------------------------------------------------------
test_data_path = "/home/fawaz/Documents/MNIST/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
test_labels_path = "/home/fawaz/Documents/MNIST/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"

test_imagearray = idx2numpy.convert_from_file(test_data_path)
test_labelsarray = idx2numpy.convert_from_file(test_labels_path)

test_dataset = MNIST_DATASET(test_imagearray, test_labelsarray)
test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)

correct = 0
total = 0

print("\n\n\nTesting...")

with torch.no_grad():
    for batch in test_dataloader:
        x, y = batch
        output = model.forward(x)
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print(f"Accuracy: {round(correct/total, 3)}")

test_dataloader = DataLoader(test_dataset, 1, shuffle=True)

for batch in test_dataloader:

    input, label = batch

    output = model.forward(input)
    print(output.shape)

    print(f"Label For Image is... {label}")
    print(f"Prediction... {torch.argmax(output[0])}")
    plt.imshow((input.numpy())[0], cmap=plt.cm.binary)
    plt.show()
    
    # print(f"Prediction is ... {torch.argmax((model.forward(input))[0])}")
