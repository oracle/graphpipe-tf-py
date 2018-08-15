#!/bin/bash
set -ex

brew install curl openssl

TF_DIR=/usr/local
TF_TYPE=cpu

if [ ! -e "/usr/local/lib/libtensorflow.so" ]; then
    curl -L "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-darwin-x86_64-1.8.0.tar.gz" | sudo tar -C ${TF_DIR} -xz
fi

pushd remote_op

FB_VERSION=1.9.0
if [ ! -d "flatbuffers" ]; then
    curl -L https://github.com/google/flatbuffers/archive/v${FB_VERSION}.tar.gz | tar -xz
    mv flatbuffers-${FB_VERSION} flatbuffers
fi

export EXTRA_FLAGS="-L/usr/local/opt/openssl/lib -I/usr/local/opt/openssl/include"
make clean; make

popd


pip3 install -r test-requirements.txt
python setup.py bdist_wheel
