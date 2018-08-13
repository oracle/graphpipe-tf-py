set -o errexit

# VENV=venv3
#
# if [ ! -d "${VENV}" ]; then
#     virtualenv -p python3 ${VENV}
# fi
#
# source ${VENV}/bin/activate
#

pushd remote_op

FB_VERSION=1.9.0

if [ ! -d "flatbuffers" ]; then
    curl -L https://github.com/google/flatbuffers/archive/v${FB_VERSION}.tar.gz | tar -xz
    mv flatbuffers-${FB_VERSION} flatbuffers
fi

make clean; make

popd


pip3 install -r test-requirements.txt
python setup.py bdist_wheel
