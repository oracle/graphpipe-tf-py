import os

import tensorflow as tf

from graphpipe import remote


def load_op_library():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return tf.load_op_library(os.path.join(dir_path, 'remote_op.so'))


_remote_op = load_op_library()


def remote_op(uri, inp, input_name=None, output_name=None, output_type=None,
              output_shape=None, config=""):
    res = remote_op_multi(uri,
                          [inp],
                          None if input_name is None else [input_name],
                          None if output_name is None else [output_name],
                          None if output_type is None else [output_type],
                          None if output_shape is None else [output_shapes],
                          config)
    if len(res) == 1:
        res = res[0]
    return res


def remote_op_multi(uri, inputs, input_names, output_names, output_types=None,
                    output_shapes=None, config=""):
    if not output_types or not output_shapes:
        all_names = remote.get_output_names(uri)
        all_types = remote.get_output_types(uri)
        all_shapes = remote.get_output_shapes(uri)
        if not output_names:
            output_names = [all_names[-1]]
        actual_types = []
        actual_shapes = []
        for i in range(len(output_names)):
            for j in range(len(all_names)):
                if all_names[j] == output_names[i]:
                    actual_types.append(all_types[j])
                    actual_shapes.append(all_shapes[j])
        if not output_types:
            output_types = actual_types
        if not output_shapes:
            output_shapes = actual_shapes
    return _remote_op.remote(uri,
                             config,
                             inputs,
                             input_names,
                             output_names,
                             output_types,
                             output_shapes)
