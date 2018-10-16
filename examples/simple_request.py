#!/usr/bin/env python
#
# Copyright (c) 2018, Oracle and/or its affiliates. All rights reserved.
#
# Licensed under the Universal Permissive License v 1.0 as shown at
# http://oss.oracle.com/licenses/upl.

"""
Serves a simple model with model_server.py and makes a request to it
"""

import subprocess
import time

import numpy as np

from graphpipe import remote

# create a simple protobuf model that squares its input
process = subprocess.Popen(["./tf_graph.py"])
process.wait()

port = "4242"
process = subprocess.Popen(["./model_server.py", "--port", port, "--model", "tf_graph.pb"])
# make sure the process has time to start
time.sleep(3)

# send a request
x = np.array(0.42)
y = remote.execute("http://127.0.0.1:" + port, x)
# print the response
print(y)

# kill the server
process.kill()
