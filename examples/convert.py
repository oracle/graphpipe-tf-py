#!/usr/bin/env python

import collections
import os.path

import tensorflow as tf

from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras import models
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import graph_util

def write_graph(graph, fname):
    d, f = os.path.split(os.path.abspath(fname))
    graph_io.write_graph(graph, d, f, as_text=False)


def constantize(fname):
    K.clear_session()
    tf.reset_default_graph()
    K.set_learning_phase(False)
    mod = models.load_model(fname)
    outputs = mod.output
    if not isinstance(outputs, collections.Sequence):
        outputs = [outputs]
    output_names = []
    for output in outputs:
        output_names.append(output.name.split(':')[0])
    sess = K.get_session()
    cg = graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(add_shapes=True), output_names)
    K.clear_session()
    return cg


def h5_to_pb(h5, pb):
    write_graph(constantize(h5), pb)

if __name__ == "__main__":
    # disable gpu for conversion
    config = tf.ConfigProto(allow_soft_placement=True,
                            device_count={'CPU': 1, 'GPU': 0})
    session = tf.Session(config=config)
    K.set_session(session)

    import sys
    if len(sys.argv) < 3:
        print('usage: {} <src_fname> <dst_fname>'.format(sys.argv[0]))
        sys.exit(1)
    h5_to_pb(sys.argv[1], sys.argv[2])
    print('saved the constant graph (ready for inference) at: ', sys.argv[2])

