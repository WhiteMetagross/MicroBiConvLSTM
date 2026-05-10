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

#include "tensorflow/lite/micro/kernels/fully_connected.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

void* FullyConnectedInit(TfLiteContext* context, const char* buffer,
                         size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context,
                                           sizeof(OpDataFullyConnected));
}

TfLiteStatus FullyConnectedPrepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);

  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* data = static_cast<OpDataFullyConnected*>(node->user_data);
  const auto params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter = micro_context->AllocateTempInputTensor(
      node, kFullyConnectedWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  TfLiteTensor* bias =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedBiasTensor);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(
      node, kFullyConnectedOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  if ((input->type == kTfLiteFloat32 && filter->type != kTfLiteFloat32) ||
      (input->type == kTfLiteInt8 &&
       (filter->type != kTfLiteInt8 && filter->type != kTfLiteInt4)) ||
      (input->type == kTfLiteInt16 && filter->type != kTfLiteInt8)) {
    MicroPrintf("Input type: %s with filter type: %s not supported.",
                TfLiteTypeGetName(input->type),
                TfLiteTypeGetName(filter->type));
    return kTfLiteError;
  }

  if (filter->type == kTfLiteInt4) {
    int filter_size =
        RuntimeShape(filter->dims->size,
                     reinterpret_cast<const int32_t*>(filter->dims->data))
            .FlatSize();
    context->RequestScratchBufferInArena(context, filter_size,
                                         &data->filter_buffer_index);
  }

  TfLiteTensor input_snapshot = *input;
  TfLiteTensor filter_snapshot = *filter;
  TfLiteTensor output_snapshot = *output;
  TfLiteTensor bias_snapshot;
  TfLiteTensor* bias_for_prepare = nullptr;
  if (bias != nullptr) {
    bias_snapshot = *bias;
    bias_for_prepare = &bias_snapshot;
  }

  // FC prepare allocates persistent per-channel metadata. Release temp tensor
  // wrappers before those allocations so the arena allocator can safely reset
  // temp state between node preparations on ESP32.
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }
  micro_context->DeallocateTempTfLiteTensor(output);

  TF_LITE_ENSURE_OK(context, CalculateOpDataFullyConnected(
                                 context, params->activation,
                                 input_snapshot.type, &input_snapshot,
                                 &filter_snapshot, bias_for_prepare,
                                 &output_snapshot, data));

#ifdef USE_TFLM_COMPRESSION

  // Compression scratch buffers.
  // These will only be allocated if the tensor is compressed.
  if (micro_context->IsTensorCompressed(node, kFullyConnectedWeightsTensor) &&
      filter->type == kTfLiteInt4) {
    MicroPrintf("Compression not supported with INT4 tensors");
    return kTfLiteError;
  }
  data->weights_scratch_index =
      micro_context->AllocateDecompressionScratchBuffer(
          node, kFullyConnectedWeightsTensor);
  data->bias_scratch_index = micro_context->AllocateDecompressionScratchBuffer(
      node, kFullyConnectedBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  return kTfLiteOk;
}

TfLiteStatus FullyConnectedEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto* params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedBiasTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kFullyConnectedOutputTensor);

#ifdef USE_TFLM_COMPRESSION

  MicroContext* micro_context = GetMicroContext(context);

  const CompressionTensorData* weights_comp_td =
      micro_context->GetTensorCompressionData(node,
                                              kFullyConnectedWeightsTensor);
  const CompressionTensorData* bias_comp_td =
      micro_context->GetTensorCompressionData(node, kFullyConnectedBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data =
      *(static_cast<const OpDataFullyConnected*>(node->user_data));
  int32_t* per_channel_output_multiplier = nullptr;
  int32_t* per_channel_output_shift = nullptr;
  if (data.is_per_channel) {
    const int channel_count = data.per_channel_quantization_size;
    per_channel_output_multiplier = reinterpret_cast<int32_t*>(
        __builtin_alloca(channel_count * sizeof(int32_t)));
    per_channel_output_shift = reinterpret_cast<int32_t*>(
        __builtin_alloca(channel_count * sizeof(int32_t)));
    for (int i = 0; i < channel_count; ++i) {
      const double effective_output_scale =
          static_cast<double>(data.input_scale) *
          static_cast<double>(data.per_channel_scales[i]) /
          static_cast<double>(data.output_scale);
      int channel_shift = 0;
      QuantizeMultiplier(effective_output_scale,
                         &per_channel_output_multiplier[i], &channel_shift);
      per_channel_output_shift[i] = channel_shift;
    }
  }

  // Checks in Prepare ensure input, output and filter types are all the same.
  switch (input->type) {
    case kTfLiteFloat32: {
      tflite::reference_ops::FullyConnected(
          FullyConnectedParamsFloat(params->activation),
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<float>(micro_context, filter,
                                              weights_comp_td,
                                              data.weights_scratch_index),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<float>(
              micro_context, bias, bias_comp_td, data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<float>(bias),
#endif  // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
      break;
    }

    case kTfLiteInt8: {
      switch (filter->type) {
        case kTfLiteInt4: {
          int8_t* unpacked_filter_data = static_cast<int8_t*>(
              context->GetScratchBuffer(context, data.filter_buffer_index));
          tflite::tensor_utils::UnpackDenseInt4IntoInt8(
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(filter).FlatSize(),
              unpacked_filter_data);
          tflite::reference_integer_ops::FullyConnected(
              FullyConnectedParamsQuantized(data),
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorShape(filter), unpacked_filter_data,
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int8_t>(output));
          break;
        }
        case kTfLiteInt8: {
          data.is_per_channel
              ? tflite::reference_integer_ops::FullyConnectedPerChannel(
                    FullyConnectedParamsQuantized(data),
                    per_channel_output_multiplier,
                    reinterpret_cast<const int*>(per_channel_output_shift),
                    tflite::micro::GetTensorShape(input),
                    tflite::micro::GetTensorData<int8_t>(input),
                    tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
                    tflite::micro::GetTensorData<int8_t>(
                        micro_context, filter, weights_comp_td,
                        data.weights_scratch_index),
                    tflite::micro::GetTensorShape(bias),
                    tflite::micro::GetOptionalTensorData<int32_t>(
                        micro_context, bias, bias_comp_td,
                        data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
                    tflite::micro::GetTensorData<int8_t>(filter),
                    tflite::micro::GetTensorShape(bias),
                    tflite::micro::GetOptionalTensorData<int32_t>(bias),
#endif  // USE_TFLM_COMPRESSION
                    tflite::micro::GetTensorShape(output),
                    tflite::micro::GetTensorData<int8_t>(output))
              : tflite::reference_integer_ops::FullyConnected(
                    FullyConnectedParamsQuantized(data),
                    tflite::micro::GetTensorShape(input),
                    tflite::micro::GetTensorData<int8_t>(input),
                    tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
                    tflite::micro::GetTensorData<int8_t>(
                        micro_context, filter, weights_comp_td,
                        data.weights_scratch_index),
                    tflite::micro::GetTensorShape(bias),
                    tflite::micro::GetOptionalTensorData<int32_t>(
                        micro_context, bias, bias_comp_td,
                        data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
                    tflite::micro::GetTensorData<int8_t>(filter),
                    tflite::micro::GetTensorShape(bias),
                    tflite::micro::GetOptionalTensorData<int32_t>(bias),
#endif  // USE_TFLM_COMPRESSION
                    tflite::micro::GetTensorShape(output),
                    tflite::micro::GetTensorData<int8_t>(output));
          break;
        }
        default: {
          MicroPrintf("Filter type %s (%d) not supported.",
                      TfLiteTypeGetName(filter->type), input->type);
          return kTfLiteError;
        }
      }
      break;
    }

    case kTfLiteInt16: {
      switch (filter->type) {
        case kTfLiteInt8: {
          if (bias == nullptr || bias->type == kTfLiteInt32) {
            data.is_per_channel
                ? tflite::reference_integer_ops::FullyConnectedPerChannel(
                      FullyConnectedParamsQuantized(data),
                      per_channel_output_multiplier,
                      reinterpret_cast<const int*>(per_channel_output_shift),
                      tflite::micro::GetTensorShape(input),
                      tflite::micro::GetTensorData<int16_t>(input),
                      tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorData<int8_t>(
                          micro_context, filter, weights_comp_td,
                          data.weights_scratch_index),
                      tflite::micro::GetTensorShape(bias),
                      tflite::micro::GetOptionalTensorData<int32_t>(
                          micro_context, bias, bias_comp_td,
                          data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorData<int8_t>(filter),
                      tflite::micro::GetTensorShape(bias),
                      tflite::micro::GetOptionalTensorData<int32_t>(bias),
#endif  // USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<int16_t>(output))
                : tflite::reference_integer_ops::FullyConnected(
                      FullyConnectedParamsQuantized(data),
                      tflite::micro::GetTensorShape(input),
                      tflite::micro::GetTensorData<int16_t>(input),
                      tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorData<int8_t>(
                          micro_context, filter, weights_comp_td,
                          data.weights_scratch_index),
                      tflite::micro::GetTensorShape(bias),
                      tflite::micro::GetOptionalTensorData<int32_t>(
                          micro_context, bias, bias_comp_td,
                          data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorData<int8_t>(filter),
                      tflite::micro::GetTensorShape(bias),
                      tflite::micro::GetOptionalTensorData<int32_t>(bias),
#endif  // USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<int16_t>(output));
          } else if (bias->type == kTfLiteInt64) {
            data.is_per_channel
                ? tflite::reference_integer_ops::FullyConnectedPerChannel(
                      FullyConnectedParamsQuantized(data),
                      per_channel_output_multiplier,
                      reinterpret_cast<const int*>(per_channel_output_shift),
                      tflite::micro::GetTensorShape(input),
                      tflite::micro::GetTensorData<int16_t>(input),
                      tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorData<int8_t>(
                          micro_context, filter, weights_comp_td,
                          data.weights_scratch_index),
                      tflite::micro::GetTensorShape(bias),
                      tflite::micro::GetOptionalTensorData<int64_t>(
                          micro_context, bias, bias_comp_td,
                          data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorData<int8_t>(filter),
                      tflite::micro::GetTensorShape(bias),
                      tflite::micro::GetOptionalTensorData<int64_t>(bias),
#endif  // USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<int16_t>(output))
                : tflite::reference_integer_ops::FullyConnected(
                      FullyConnectedParamsQuantized(data),
                      tflite::micro::GetTensorShape(input),
                      tflite::micro::GetTensorData<int16_t>(input),
                      tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorData<int8_t>(
                          micro_context, filter, weights_comp_td,
                          data.weights_scratch_index),
                      tflite::micro::GetTensorShape(bias),
                      tflite::micro::GetOptionalTensorData<int64_t>(
                          micro_context, bias, bias_comp_td,
                          data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorData<int8_t>(filter),
                      tflite::micro::GetTensorShape(bias),
                      tflite::micro::GetOptionalTensorData<int64_t>(bias),
#endif  // USE_TFLM_COMPRESSION
                      tflite::micro::GetTensorShape(output),
                      tflite::micro::GetTensorData<int16_t>(output));
          }
          break;
        }
        default: {
          MicroPrintf("Filter type %s (%d) not supported.",
                      TfLiteTypeGetName(filter->type), input->type);
          return kTfLiteError;
        }
      }
      break;
    }

    default: {
      MicroPrintf("Input type %s (%d) not supported.",
                  TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_FULLY_CONNECTED() {
  return tflite::micro::RegisterOp(FullyConnectedInit, FullyConnectedPrepare,
                                   FullyConnectedEval);
}

TFLMInferenceRegistration RegisterInference_FULLY_CONNECTED() {
  return tflite::micro::RegisterOp(FullyConnectedEval);
}

}  // namespace tflite
