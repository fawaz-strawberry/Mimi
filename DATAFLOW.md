Assume the following example as your dataset

'"When thou impressest, what are precepts worth
  Of stale example? When thou wilt inflame,
  How coldly those impediments stand forth,
  Of wealth, of filial fear, law, kindred, fame!
  Love's arms are peace, 'gainst rule, 'gainst sense, 'gainst shame.
  And sweetens, in the suff'ring pangs it bears,
  The aloes of all forces, shocks and fears.

If you want to break this down into manageable chunks for the dataloader to bring in
then you have to simulatenously have a way to tokenize it and still maintain the same
amount of characters. Otherwise, the better way is to pretokenize your dataset and
basically store a seperate copy.... actually tbh, that is the only way to properly
store your data. But that means that you can no longer dynamically load in data? right?
This is false, you can still fit your tokenizer over everything and just add in new characters
by going large chunk by chunk. So for example, if you didn't want to tokenize the whole wikipedia
dataset but still wanted to have the whole thing available to train then you would instead
take a chunk for wikipedia, download it, run the tokenizer to make sure it's up to date and
ensure that the tokenizer doesn't overwrite previous tokens. 2 epochs will be enough because by the
end of the first you will have all tokens in your dataset. Of course unique tokens will be hard
but that was always going to be the case. The point is to most efficiently store the most frequently
found tokens. Utilizing BPE tokenizers I think this is possible pull off


Now this gets to this specific projects setup, how does on integrate into the trianing loop

The first step after a "dataset load" is to tokenize it, generate a file and then feed that into
your dataset loader class. You DO NOT feed in the nontokenized data into the datsetLoader class otherwise
it defeats the purpose of fast data loading. While the model is training, you can possibly do work
to begin loading in your next dataloader if neccessary but tbh, it's probably a relatively short time
to reload and rebuild the dataloader compared to the raw training time. This results in a fast and
consistent training process.