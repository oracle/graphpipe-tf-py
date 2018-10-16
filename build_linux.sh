#!/bin/bash
#
# Copyright (c) 2018, Oracle and/or its affiliates. All rights reserved.
#
# Licensed under the Universal Permissive License v 1.0 as shown at
# http://oss.oracle.com/licenses/upl.

set -ex


apt-get update
apt-get install -y libcurl4-openssl-dev curl

TF_DIR=/usr/local
TF_TYPE=gpu

if [ ! -e "/usr/local/lib/libtensorflow.so" ]; then
    curl -L "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-linux-x86_64-1.8.0.tar.gz" | tar -C ${TF_DIR} -xz
fi

pushd remote_op

FB_VERSION=1.9.0
if [ ! -d "flatbuffers" ]; then
    curl -L https://github.com/google/flatbuffers/archive/v${FB_VERSION}.tar.gz | tar -xz
    mv flatbuffers-${FB_VERSION}/ flatbuffers
fi

make clean; make

popd

pip3 install -r test-requirements.txt
python setup.py bdist_wheel

#exec gosu tox tox
