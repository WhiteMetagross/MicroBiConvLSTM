/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstring>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

namespace {

constexpr int kOutputTensor = 0;

size_t ElementSize(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return sizeof(float);
    case kTfLiteInt8:
      return sizeof(int8_t);
    case kTfLiteInt16:
      return sizeof(int16_t);
    case kTfLiteInt32:
      return sizeof(int32_t);
    case kTfLiteInt64:
      return sizeof(int64_t);
    default:
      return 0;
  }
}

TfLiteStatus PackEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLitePackParams* data =
      reinterpret_cast<TfLitePackParams*>(node->builtin_data);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  const size_t element_size = ElementSize(output->type);
  if (element_size == 0) {
    MicroPrintf("Type '%s' is not supported by pack.",
                TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }

  const TfLiteEvalTensor* input0 =
      tflite::micro::GetEvalInput(context, node, 0);
  const int dimensions = output->dims->size;
  int axis = data->axis;
  if (axis < 0) {
    axis += dimensions;
  }

  const TfLiteIntArray* input_dims = input0->dims;
  const TfLiteIntArray* output_dims = output->dims;

  int outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= output_dims->data[i];
  }

  int copy_size = 1;
  for (int i = axis + 1; i < dimensions; ++i) {
    copy_size *= output_dims->data[i];
  }

  int input_size = 1;
  for (int i = 0; i < input_dims->size; ++i) {
    input_size *= input_dims->data[i];
  }

  TFLITE_DCHECK_EQ(input_size, copy_size * outer_size);

  uint8_t* output_data = tflite::micro::GetTensorData<uint8_t>(output);
  const size_t chunk_bytes = static_cast<size_t>(copy_size) * element_size;

  for (int i = 0; i < data->values_count; ++i) {
    const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, i);
    const uint8_t* input_data = tflite::micro::GetTensorData<uint8_t>(input);
    for (int outer = 0; outer < outer_size; ++outer) {
      const uint8_t* input_ptr =
          input_data + static_cast<size_t>(outer) * chunk_bytes;
      const int out_index = outer * data->values_count * copy_size + i * copy_size;
      uint8_t* output_ptr =
          output_data + static_cast<size_t>(out_index) * element_size;
      std::memcpy(output_ptr, input_ptr, chunk_bytes);
    }
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_PACK() {
  return tflite::micro::RegisterOp(nullptr, nullptr, PackEval);
}

}  // namespace tflite
