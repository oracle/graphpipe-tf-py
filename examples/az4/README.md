# AZ Four inference

This directory contains a few versions of a convolutional model trained with
the AlphaZero algorithm to evaluate four-in-a-row positions. The model expects
inputs of the shape [?, 2, 6, 7] where each [2, 6, 7] represents a single board
position. The first [6, 7] array is a binary representation of the pieces for
the current player, and the second [6,7] is a binary representation of the
pieces for the opposing player. In this binary representation, an empty square
is represented by a 0.0, and a piece is represented by a 1.0.

The model returns two outputs. The first output, called the policy output, is a
[?, 7] array representing the softmax representation of the value of a move in
each of the seven columns starting from the left for each of the input
positions. For example, an evaluation might look like the following:

    [0.0, 0.0, 0.1, 0.8, 0.1, 0.0, 0.0]

This means that the agent things going in the center is the optimal move, and
column 3 and 5 are also reasonable choices.

The second output, called the value output, is a single floating point value
representing the expectation of the result of the game from the perspective of
the current player. A value of 1.0 means it is a certainty the player will win,
and a value -1.0 means it is a certainty the player will lose. A value close to
0.0 represents a draw.

See [az4.py](az4.py) for an example of how to query the model using GraphPipe.
