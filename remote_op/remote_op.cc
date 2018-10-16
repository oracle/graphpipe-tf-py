// Copyright (c) 2018, Oracle and/or its affiliates. All rights reserved.
//
// Licensed under the Universal Permissive License v 1.0 as shown at
// http://oss.oracle.com/licenses/upl.

#include <curl/curl.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "flatbuffers/flatbuffers.h"
#include "graphpipe_generated.h"

using namespace tensorflow;
using graphpipe::CreateInferRequest;
using graphpipe::Req_InferRequest;
using graphpipe::CreateRequest;
using graphpipe::CreateTensor;
using graphpipe::InferResponse;

REGISTER_OP("Remote")
    .Input("uri: string")
    .Input("config: string")
    .Input("inputs: input_types")
    .Input("input_names: string")
    .Input("output_names: string")
    .Output("outputs: output_types")
    .Attr("input_types: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      std::vector<PartialTensorShape> output_shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("output_shapes", &output_shapes));
      if (output_shapes.size() != c->num_outputs()) {
        return errors::InvalidArgument(
            "`output_shapes` must be the same length as `output_types` (",
            output_shapes.size(), " vs. ", c->num_outputs());
      }
      for (size_t i = 0; i < output_shapes.size(); ++i) {
        shape_inference::ShapeHandle output_shape_handle;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(
            output_shapes[i], &output_shape_handle));
        c->set_output(static_cast<int>(i), output_shape_handle);
      }
      return Status::OK();
    })
    .Doc(R"doc(
Execute a remote model.
)doc");


struct Reader {
  const char * pos;
  const char * end;
};

static size_t ReadCallback(void *dest, size_t size, size_t nmemb, void *userp)
{
      auto r = (Reader*)userp;
      size_t to_read = std::min(size * nmemb, (size_t)(r->end - r->pos));
      if (to_read) {
        memcpy(dest, r->pos, to_read);
        r->pos += to_read;
      }
      return to_read;
}

static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
      ((std::string*)userp)->append((char*)contents, size * nmemb);
          return size * nmemb;
}

static constexpr graphpipe::Type conv2flat[] = {
  graphpipe::Type_Null, // DT_INVALID = 0;
  graphpipe::Type_Float32, // DT_FLOAT = 1;
  graphpipe::Type_Float64, // DT_DOUBLE = 2;
  graphpipe::Type_Int32, // DT_INT32 = 3;
  graphpipe::Type_Uint8, // DT_UINT8 = 4;
  graphpipe::Type_Int16, // DT_INT16 = 5;
  graphpipe::Type_Int8, // DT_INT8 = 6;
  graphpipe::Type_String, // DT_STRING = 7;
  graphpipe::Type_Null, // DT_COMPLEX64 = 8;  // Single-precision complex
  graphpipe::Type_Int64, // DT_INT64 = 9;
  graphpipe::Type_Null, // DT_BOOL = 10;
  graphpipe::Type_Null, // DT_QINT8 = 11;     // Quantized int8
  graphpipe::Type_Null, // DT_QUINT8 = 12;    // Quantized uint8
  graphpipe::Type_Null, // DT_QINT32 = 13;    // Quantized int32
  graphpipe::Type_Null, // DT_BFLOAT16 = 14;  // Float32 truncated to 16 bits.  Only for cast ops.
  graphpipe::Type_Null, // DT_QINT16 = 15;    // Quantized int16
  graphpipe::Type_Null, // DT_QUINT16 = 16;   // Quantized uint16
  graphpipe::Type_Uint16, // DT_UINT16 = 17;
  graphpipe::Type_Null, // DT_COMPLEX128 = 18;  // Double-precision complex
  graphpipe::Type_Float16, // DT_HALF = 19;
  graphpipe::Type_Null, // DT_RESOURCE = 20;
  graphpipe::Type_Null, // DT_VARIANT = 21;  // Arbitrary C++ data types
  graphpipe::Type_Uint32, // DT_UINT32 = 22;
  graphpipe::Type_Uint64, // DT_UINT64 = 23;
};

static constexpr DataType conv2tf[] = {
  DT_INVALID, // Type_Null = 0,
  DT_UINT8, // Type_Uint8 = 1,
  DT_INT8, // Type_Int8 = 2,
  DT_UINT16, // Type_Uint16 = 3,
  DT_INT16, // Type_Int16 = 4,
  DT_UINT32, // Type_Uint32 = 5,
  DT_INT32, // Type_Int32 = 6,
  DT_UINT64, // Type_Uint64 = 7,
  DT_INT64, // Type_Int64 = 8,
  DT_HALF, // Type_Float16 = 9,
  DT_FLOAT, // Type_Float32 = 10,
  DT_DOUBLE, // Type_Float64 = 11,
  DT_STRING, // Type_String = 12,
};

inline graphpipe::Type to_flat_dtype(DataType dt) {
  return conv2flat[dt];
}

inline DataType to_tf_dtype(graphpipe::Type dt) {
  return conv2tf[dt];
}

flatbuffers::Offset<graphpipe::Tensor> to_flat(flatbuffers::FlatBufferBuilder &builder, const Tensor *tensor) {
  std::vector<int64_t> s;
  for (int d = 0; d < tensor->dims(); d++) {
    s.push_back(tensor->dim_size(d));
  }
  auto shape = builder.CreateVector<int64_t>(s);
  auto type = to_flat_dtype(tensor->dtype());

  flatbuffers::Offset<flatbuffers::Vector<uint8_t>> data = 0;
  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> string_val = 0;
  if (type == graphpipe::Type_String) {
    auto num_vals = tensor->NumElements();
    auto vals = tensor->flat<string>();
    std::vector<std::string> valvec;
    for (int i = 0; i < num_vals; ++i) {
      valvec.push_back(vals(i));
    }
    string_val = builder.CreateVectorOfStrings(valvec);
  } else {
    data = builder.CreateVector<uint8_t>((const uint8_t*)tensor->tensor_data().data(), tensor->tensor_data().size());
  }
  return CreateTensor(builder, type, shape, data, string_val);
}

bool from_flat(Tensor *tensor, const graphpipe::Tensor *flat_tensor) {
  TensorShape shape;
  for (int i = 0; i < flat_tensor->shape()->Length(); ++i) {
    shape.AddDim(flat_tensor->shape()->Get(i));
  }
  auto type = to_tf_dtype(flat_tensor->type());
  *tensor = Tensor(type, shape);
  if (type == DT_STRING) {
    auto dstarray = tensor->flat<string>();
    for (int i = 0; i < flat_tensor->string_val()->Length(); ++i) {
      dstarray(i).assign(flat_tensor->string_val()->Get(i)->c_str());
    }
  } else {
    memcpy((void *)tensor->tensor_data().data(), flat_tensor->data()->data(), tensor->tensor_data().size());
  }
  return true;
}

class RemoteOp : public OpKernel {
 public:
  explicit RemoteOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* tmp;
    OP_REQUIRES_OK(context, context->input("uri", &tmp));
    const string uri = tmp->scalar<string>()();
    OP_REQUIRES_OK(context, context->input("config", &tmp));
    const string config = tmp->scalar<string>()();
    OP_REQUIRES_OK(context, context->input("input_names", &tmp));
    int num_inputs = tmp->NumElements();
    auto input_names = tmp->flat<string>();
    OP_REQUIRES_OK(context, context->input("output_names", &tmp));
    int num_outputs = tmp->NumElements();
    auto output_names = tmp->flat<string>();

    flatbuffers::FlatBufferBuilder builder(1024);

    int start, stop;
    context->op_kernel().InputRange("inputs", &start, &stop);
    std::vector<flatbuffers::Offset<graphpipe::Tensor>> inputs_vector;
    for (int i = start; i < stop; ++i) {
      inputs_vector.push_back(to_flat(builder, &context->input(i)));
    }

    std::vector<std::string> invec;
    std::vector<std::string> outvec;
    for (int i = 0; i < num_inputs; ++i) {
      invec.push_back(input_names(i));
    }
    for (int i = 0; i < num_outputs; ++i) {
      outvec.push_back(output_names(i));
    }
    auto inputs = builder.CreateVector(inputs_vector);
    auto inp = builder.CreateVectorOfStrings(invec);
    auto outp = builder.CreateVectorOfStrings(outvec);
    auto conf = builder.CreateString(config);
    auto infer_req = CreateInferRequest(builder, conf, inp, inputs, outp);
    auto req = CreateRequest(builder, Req_InferRequest, infer_req.Union());
    builder.Finish(req);
    const char *buf = (const char *)builder.GetBufferPointer();
    int size = builder.GetSize();

    std::string out;

    CURL *curl;
    CURLcode response;
    long http_code = 0;

    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    if(curl) {
      curl_easy_setopt(curl, CURLOPT_URL, uri.c_str());
      curl_easy_setopt(curl, CURLOPT_POST, 1);
      Reader r = Reader{buf, buf + size};
      curl_easy_setopt(curl, CURLOPT_READDATA, &r);
      curl_easy_setopt(curl, CURLOPT_READFUNCTION, ReadCallback);
      curl_easy_setopt(curl, CURLOPT_WRITEDATA, &out);
      curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
      std::ostringstream clen;
      clen << "Content-Length: " << size;
      struct curl_slist *headerlist = NULL;
      headerlist = curl_slist_append(headerlist, clen.str().c_str());
      curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerlist);
      response = curl_easy_perform(curl);
      if (response != CURLE_OK) {
        LOG(ERROR) << "curl error: " << curl_easy_strerror(response);
      }
      curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
      curl_easy_cleanup(curl);
    }
    curl_global_cleanup();
    if (http_code != 200) {
        LOG(ERROR) << "Request failed with code " << http_code <<  ": " << out;
        return;
    }

    auto res = flatbuffers::GetRoot<InferResponse>(out.c_str());
    // iterate through res->output_tensors() and convert
    auto output_tensors = res->output_tensors();
    for (int i = 0; i < output_tensors->Length(); ++i) {
      Tensor tensor;
      if (!from_flat(&tensor, output_tensors->Get(i))) {
        LOG(ERROR) << "Tensor parsing error";
        return;
      }
      context->set_output(i, tensor);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Remote").Device(DEVICE_CPU), RemoteOp);
