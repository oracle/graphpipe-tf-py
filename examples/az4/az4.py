#!/usr/bin/env python
"""Example of making requests to the AlphaZero four model

The model convolutions are in channels first order. This
means they must be served with the gpu version of libtensorflow
or one that was built with MKL. If you have one, you can
serve the model using model_server.py:

    ../model_server.py --model az4.000050.pb

Or you can serve it using a community provided docker container:

    docker run -it --rm \
        -v "$PWD:/models/" \
        -p 9000:9000 \
        sleepsonthefloor/graphpipe-tf:cpu \
        --model=/models/az4.000050.pb \
        --listen=0.0.0.0:9000
"""

from graphpipe import remote

import numpy as np


def parse_board(s):
    s = s.strip()
    boardx = np.zeros((1, 6, 7), dtype=np.float32)
    boardo = np.zeros((1, 6, 7), dtype=np.float32)
    empty = 0
    for x, line in enumerate(s.split("\n")):
        line = line.replace(" ", "")
        for y, char in enumerate(line):
            if char == 'X':
                boardx[0, x, y] = 1
            elif char == 'O':
                boardo[0, x, y] = 1
            else:
                empty += 1
    if empty % 2 == 1:
        return np.concatenate((boardo, boardx))
    return np.concatenate((boardx, boardo))


def print_board(board):
    x, o = 0, 1
    if (np.count_nonzero(board) % 2) == 1:
        x, o = o, x
    display = np.chararray((6, 7), 3)
    display[:] = ' - '
    display[board[x] == 1] = ' X '
    display[board[o] == 1] = ' O '

    dtype = 'S' + str(7 * 3)
    strs = display.transpose().view(dtype).ravel()
    print(b'\n'.join(strs).decode("utf8"))


def print_weights(w):
    print("{:02.0f} {:02.0f} {:02.0f} {:02.0f} {:02.0f} {:02.0f} {:02.0f}".format(
        w[0]*100, w[1]*100, w[2]*100, w[3]*100, w[4]*100, w[5]*100, w[6]*100))


MODEL = "http://127.0.0.1:9000"


def evaluate(boards):

    x = np.concatenate([parse_board(board)[np.newaxis, :] for board in boards])
    weights, values = remote.execute(MODEL, x)
    for i in range(len(weights)):
        weight = weights[i]
        value = values[i][0]
        move = int(x[i].sum())
        player = move % 2
        print("Move {}: {} to play".format(move, "O" if player else "X"))
        print_board(x[i])
        print_weights(weight)
        if player:
            value = -value
        if value > 0:
            print("Evaluation: X wins {:02.0f}%".format(value * 100))
        else:
            print("Evaluation: O wins {:02.0f}%".format(value * -100))
        print()


boards = []
boards.append("""
- - - - - - -
- - - - - - -
- - - - - - -
- - - - - - -
- - - - - - -
- - - - - - -
""")

boards.append("""
- - - - - - -
- - - - - - -
- - - - - - -
- - - - - - -
- - - - - - -
- - - X - - -
""")

boards.append("""
- - - - - - -
- - - - - - -
- - - - - - -
- - - - - - -
- - - - X X -
X - - - O O -
""")

evaluate(boards)
