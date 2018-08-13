from tensorflow.python.keras._impl.keras import engine

from . import ops


class Remote(engine.Layer):
    """Remote Layer allows a Keras model to include a remote GraphPipe model.

    The keras layer uses a tensorflow remote_op plugin to make a request to an
    external model. The only required parameter to the constructor is the uri
    of the remote model. If output_type and and output_shape are not specified,
    they will be inferred using the GraphPipe metadata api.

    The layer is created with trainable=False, because it is not possible to
    train the remote model. It is possible to use a remote layer with
    additional trainable layers on top to fine-tune the output of a remote
    model or to create an ensemble.
    """

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
