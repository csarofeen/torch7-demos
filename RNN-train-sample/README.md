# RNN training example

Train an RNN to recognize a sequence abba in a random ababbaabbbaaababab sequence of two characters.

Version 1. `main.lua`: Send sequecne of 4 to train but test on each character at a time.

```
th main.lua
```

Version 2. `train-er-rnn.lua`: is based on Element Research [rnn package](https://github.com/Element-Research/rnn/blob/master/examples/sequence-to-one.lua).

```
th train-er-rnn.lua
```
