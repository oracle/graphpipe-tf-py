# graphpipe python helper for TensorFlow

This package contains a new operation for tensorflow that allows you to
make a call to a graphpipe remote model inside a local tensorflow graph.
The new operation is called remote_op and communicates with the remote
model using libcurl and the graphpipe protocol. A complete example is
provided in the examples subdirectory.

Included also, are examples of serving a model.

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
