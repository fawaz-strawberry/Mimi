'''
The following code will take a text file full of text and convert it into a file of bytes.
'''

filename = "Datasets/shakespeare.txt"

with open(filename) as f:
    text = f.read()

print_amt = 5

print(text[:print_amt])

# Convert text to bytes
text_as_bytes = text.encode("utf-8")

print(int.from_bytes(text_as_bytes[:print_amt], "little"))

