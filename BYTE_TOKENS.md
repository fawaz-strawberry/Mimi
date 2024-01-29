## What does it mean to have a byte tokenizer
A byte tokenizer gives the model flexibility to learn the core fundamentals and nature of the problem. It's like a brain purely getting the chemical activations.
The downside is that so much information needs to be encompassed into a singular byte to represent so much information.

## How do bytes get fed into the network
So we can convert a string of data into bytes, but do we convert this into integers to be fed into the network? Yes, the answer to that question is indeed yes.
Otherwise we can't really do regular operation of multiplications on the bytes unless we want to go down the and xor nor type of stuff and do hardware level
multiplications. We can convert everything into a byte, then feed it into our "tokenizer" which can now be predefined considering it's a simple operation and will
return back a list of integers for each byte given a chunk of text.

Thus

"I am a super cool person" => [49 20 61 6d 20 61 20 73 75 70 65 72 20 63 6f 6f 6c 20 70 65 72 73 6f 6e 0a]

Where now we can feed in this chain of bytes into the transformer. Now while this is a lot of tokens, it should be doable.