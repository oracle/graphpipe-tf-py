from tensorflow.python.keras._impl.keras import engine

from . import ops


class Remote(engine.Layer):
    def __init__(self, name=None, uri=None, input_name=None, output_name=None,
                 output_type=None, output_shape=None, config=None, **kwargs):
        self.uri = uri
        self.iname = input_name or ""
        self.oname = output_name or ""
        self.otype = output_type
        self.oshape = output_shape
        self.config = config or ""
        super(Remote, self).__init__(name=name, trainable=False, **kwargs)

    def call(self, x):
        return ops.remote_op(self.uri,
                             x,
                             self.iname,
                             self.oname,
                             self.otype,
                             self.oshape,
                             self.config)
