# RNN training example

Train RNNs to recognize a sequence `abba` in a random `ababbaabbbaaababab` sequence of two characters.

Version 1. `main.lua`: Send sequecne of 4 symbols to train but test on each character at a time.

```bash
th main.lua
```

The results will be not optimal (by default you're using `1` hidden layer with `d = 2` neurons), but you'll be able to appreciate a wide variety of combinations.
True positives, true negatives, false positives and false negatives.
Try to use more neurons or more hidden layers to improve performace (switch from `d = 2` neurons to `3`, and all will work perfectly, even with just one hidden layer).
(Type `th main.lua -h` to see all the available options.)

There are three different models available to train which can be selected by modifying variable `mode`:

 + Simple [RNN](RNN.lua)
 + Gated Recurrent Unit ([GRU](GRU.lua))
 + Fast Weights ([FW](FW.lua))

Version 2. `train-er-rnn.lua`: is based on Element Research [rnn package](https://github.com/Element-Research/rnn/blob/master/examples/sequence-to-one.lua).

```
th train-er-rnn.lua
```

## Train on longer sequences

To make the task more challenging, we can increase the number of symbols that make up our `key` sequence.
To experiment with longer random sequences, use the optionr:

 + `-ds randomSeq`,
 + `-S [int]` to specify the key sequence length,
 + `-T [int]` to set the unrolling through time steps,
 + `-mode [RNN|GRU|FW]` to select a specific model,
 + `-d [int]` for the hidden unit dimension,
 + `-nHL [int]` to set the number of hidden layers
