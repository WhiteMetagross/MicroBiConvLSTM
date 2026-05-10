#include <cstring>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

namespace {

constexpr int kInputTensor = 0;
constexpr int kAxisTensor = 1;
constexpr int kOutputTensor = 0;
constexpr int kMaxDimensions = 8;

int NumTensorElements(const TfLiteEvalTensor* tensor) {
  if (tensor == nullptr || tensor->dims == nullptr) {
    return 0;
  }
  int count = 1;
  for (int i = 0; i < tensor->dims->size; ++i) {
    count *= tensor->dims->data[i];
  }
  return count;
}

template <typename TAxis>
TfLiteStatus PopulateReverseMask(const TfLiteEvalTensor* axis_tensor,
                                 int dims_count,
                                 bool* reverse_dims) {
  const int axis_count = NumTensorElements(axis_tensor);
  const TAxis* axis_data = tflite::micro::GetTensorData<TAxis>(axis_tensor);
  for (int i = 0; i < axis_count; ++i) {
    int axis = static_cast<int>(axis_data[i]);
    if (axis < 0) {
      axis += dims_count;
    }
    if (axis < 0 || axis >= dims_count) {
      MicroPrintf("REVERSE_V2 axis %d out of range for %d dims.", axis,
                  dims_count);
      return kTfLiteError;
    }
    reverse_dims[axis] = true;
  }
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = tflite::GetMicroContext(context);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TfLiteTensor* axis =
      micro_context->AllocateTempInputTensor(node, kAxisTensor);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);

  TF_LITE_ENSURE(context, input != nullptr);
  TF_LITE_ENSURE(context, axis != nullptr);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), NumDimensions(output));
  TF_LITE_ENSURE_MSG(context, NumDimensions(input) <= kMaxDimensions,
                     "REVERSE_V2 supports at most 8 dimensions.");
  TF_LITE_ENSURE(context,
                 axis->type == kTfLiteInt32 || axis->type == kTfLiteInt64);

  for (int i = 0; i < NumDimensions(input); ++i) {
    TF_LITE_ENSURE_EQ(context, input->dims->data[i], output->dims->data[i]);
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(axis);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* axis_tensor =
      tflite::micro::GetEvalInput(context, node, kAxisTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  const int dims_count = input->dims->size;
  if (dims_count > kMaxDimensions) {
    MicroPrintf("REVERSE_V2 dims %d exceed max %d.", dims_count,
                kMaxDimensions);
    return kTfLiteError;
  }

  const int element_size = TfLiteTypeGetSize(input->type);
  if (element_size <= 0) {
    MicroPrintf("REVERSE_V2 unsupported tensor type %s.",
                TfLiteTypeGetName(input->type));
    return kTfLiteError;
  }

  bool reverse_dims[kMaxDimensions] = {false};
  if (axis_tensor->type == kTfLiteInt32) {
    TF_LITE_ENSURE_STATUS(PopulateReverseMask<int32_t>(axis_tensor, dims_count,
                                                       reverse_dims));
  } else if (axis_tensor->type == kTfLiteInt64) {
    TF_LITE_ENSURE_STATUS(PopulateReverseMask<int64_t>(axis_tensor, dims_count,
                                                       reverse_dims));
  } else {
    MicroPrintf("REVERSE_V2 axis tensor type %s is unsupported.",
                TfLiteTypeGetName(axis_tensor->type));
    return kTfLiteError;
  }

  bool needs_reverse = false;
  int dims[kMaxDimensions];
  int strides[kMaxDimensions];
  int total_elements = 1;
  for (int i = 0; i < dims_count; ++i) {
    dims[i] = input->dims->data[i];
    total_elements *= dims[i];
    needs_reverse = needs_reverse || reverse_dims[i];
  }

  if (!needs_reverse) {
    std::memcpy(output->data.raw, input->data.raw,
                static_cast<size_t>(total_elements) *
                    static_cast<size_t>(element_size));
    return kTfLiteOk;
  }

  if (dims_count > 0) {
    strides[dims_count - 1] = 1;
    for (int i = dims_count - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * dims[i + 1];
    }
  }

  const char* input_bytes = input->data.raw;
  char* output_bytes = output->data.raw;

  for (int output_index = 0; output_index < total_elements; ++output_index) {
    int remainder = output_index;
    int input_index = 0;
    for (int dim = 0; dim < dims_count; ++dim) {
      const int coord = remainder / strides[dim];
      remainder %= strides[dim];
      const int input_coord =
          reverse_dims[dim] ? (dims[dim] - 1 - coord) : coord;
      input_index += input_coord * strides[dim];
    }
    std::memcpy(output_bytes + (output_index * element_size),
                input_bytes + (input_index * element_size), element_size);
  }

  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_REVERSE_V2() {
  return tflite::micro::RegisterOp(nullptr, Prepare, Eval);
}

}  // namespace tflite
