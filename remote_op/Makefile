TARGET = ../graphpipe_tf/remote_op.so

PYTHON = python

TF_LIB = $(shell $(PYTHON) -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_INC = $(shell $(PYTHON) -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

CC = g++

INCLUDES = -Igraphpipe/cpp -Iflatbuffers/include -I$(TF_INC) -I$(TF_INC)/external/nsync/public

LIBS = -ltensorflow_framework -lcurl -lssl -lcrypto

FLAGS = -std=c++11 \
		-shared \
		-fPIC \
		-O2 \
		-L$(TF_LIB) \
		$(INCLUDES) \
		$(LIBS) \
		-D_GLIBCXX_USE_CXX11_ABI=0

FILES = remote_op.cc

all: $(TARGET)

$(TARGET) : $(FILES) graphpipe/cpp/graphpipe_generated.h
	$(CC) -o $(TARGET) $(FILES) $(FLAGS)

%.so : %.cc
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
