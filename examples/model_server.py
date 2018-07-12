#!/usr/bin/env python
"""
Extremely simple tensorflow graphpipe model server
"""

import argparse
from http import server

import numpy as np
import tensorflow as tf

from graphpipe import convert
from graphpipe.graphpipefb.Type import Type


def serve(host, port, model):
    with tf.Graph().as_default() as graph:
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(model, "rb") as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def)
        g = graph

    inputs = []
    outputs = []
    ops = g.get_operations()
    for op in ops:
        for tensor in op.outputs:
            shape = tensor.get_shape()
            if shape._dims is None:
                shape = []
            else:
                shape = [-1 if x is None else x for x in shape.as_list()]
            try:
                typ = convert.to_type(tensor.dtype.as_numpy_dtype)
            except KeyError:
                typ = Type.Null
            t = {
             "name": tensor.name,
             "shape": shape,
             "type": typ,
            }
            inputs.append(t)
            outputs.append(t)

    metadata = {
        "name": model,
        "version": "1.2",
        "server": "example python model server",
        "inputs": inputs,
        "outputs": outputs,
    }


    class MyHandler(server.BaseHTTPRequestHandler):
        def do_POST(self):
            req_enc = self.rfile.read(int(self.headers['Content-Length']))

            req = convert.deserialize_request(req_enc)
            if req is None:
                # Return metadata response
                output_enc = convert.serialize_metadata_response(metadata)
            else:
                feed_dict = {}
                y = []
                if req.input_names == []:
                    req.input_names = [b""]
                if req.output_names == []:
                    req.output_names = [b""]
                for name in req.output_names:
                    name = name.decode()
                    # default output_name to the last op
                    if name == "":
                        name = ops[-1].name
                    if ':' not in name:
                        name += ':0'
                    y.append(g.get_tensor_by_name(name))
                if not y:
                    y = [g.get_operations()[-1].outputs[0]]
                for i, t in enumerate(req.input_tensors):
                    name = req.input_names[i].decode()
                    # default input name to the first op
                    if name == "":
                        name = ops[0].name
                    if ':' not in name:
                        name += ':0'
                    x = g.get_tensor_by_name(name)
                    feed_dict[x] = t
                outputs = sess.run(y, feed_dict=feed_dict)
                output_enc = convert.serialize_infer_response(outputs)

            self.send_response(200)
            self.end_headers()
            self.wfile.write(output_enc)

    server_address = (host, port)
    httpd = server.HTTPServer(server_address, MyHandler)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=g, config=config) as sess:
        print('Starting httpd on %s:%d...' % (host, port))
        httpd.serve_forever()



if __name__ == "__main__":
    # Parse CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default='', help="Serving host/ip")
    parser.add_argument("--port", default=9000, help="Serving port", type=int)
    parser.add_argument("--model", required=True, help="Model file (.pb)")
    args = parser.parse_args()
    serve(args.host, args.port, args.model)

