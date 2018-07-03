#!/bin/bash
set -ex


apt-get update
apt-get install -y libcurl4-openssl-dev curl

TF_DIR=/usr/local
TF_TYPE=gpu

curl -L "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-linux-x86_64-1.8.0.tar.gz" | tar -C ${TF_DIR} -xz; \

FB_VERSION=1.9.0
curl -L https://github.com/google/flatbuffers/archive/v${FB_VERSION}.tar.gz | tar -C /tmp -xz

ls /tmp
ls /tmp/flatbuffers-${FB_VERSION}
cp -r /tmp/flatbuffers-${FB_VERSION}/include remote_op/flatbuffers/


pip3 install -r test-requirements.txt

make -C remote_op

python setup.py bdist_wheel

chown -R $1 build dist graphpipe_tf.egg-info

#exec gosu tox tox
