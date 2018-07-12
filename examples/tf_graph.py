#!/usr/bin/env python
"""
This example illustrates how to write a tensorflow graph as a constantized
graph_def protobuf so it can be served by ./model_server.py
"""

import os
import tensorflow as tf

from tensorflow.python.framework import graph_io
from tensorflow.python.framework import graph_util

g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32)

    mul = tf.multiply(x, x)

    fname = "tf_graph.pb"

    graph_def = g.as_graph_def(add_shapes=True)

    # this conversion is not necessary because there are no trainable
    # parameters, but it is included because it is important for more
    # complex models
    output_names = [mul.op.name]
    graph_def = graph_util.convert_variables_to_constants(
        tf.Session(), graph_def, output_names)

    d, f = os.path.split(os.path.abspath(fname))
    graph_io.write_graph(graph_def, d, f, as_text=False)
