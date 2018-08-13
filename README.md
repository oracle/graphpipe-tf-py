# GraphPipe helpers for TensorFlow

This package contains helpers and examples for using GraphPipe with tensorflow.
It contains a new plug-in operation for tensorflow that makes a call to a
GraphPipe remote model from within a local tensorflow graph.  The new operation
is called remote_op and communicates with the remote model using libcurl and
the GraphPipe protocol.

Additionaly, a new keras layer is included based on the
remote operation.  This allaws you to include a layer in a keras model that
makes a remote call.

Finally, various examples are included of serving tensorflow models in python.
For production, a more performant server like
[`graphpipe-tf`](https://github.com/oracle/graphpipe-go/cmd/graphpipe-tf) is
recommended, but the python server is useful for experimentation.

## List Of Examples

 * [Jupyter Notebook: serving and querying VGG with
   GraphPipe](examples/RemoteModelWithGraphPipe.ipynb)
 * [Complete client/server example](examples/simple_request.py)
 * [Simple tensorflow model server](examples/model_server.py)
 * [Keras to GraphDef](examples/convert.py)
 * [Using a remote operation](examples/call_remote_op.py)
 * [Tensorflow graph to GraphDef](examples/tf_graph.py)

## Build

Building manually requires a few libraries to be installed, but the Makefile
will happily run a build for you in a docker container.
```
  make build
```

See `manually.sh` for the additional headers besides libcurl that you will
need to build the C library. (From tensorflow and flatbuffers)

If you've successfully built the C library, to build installation packages:

    python setup.py bdist_wheel

Note that these are not manylinux wheels and depend on libcurl being installed
