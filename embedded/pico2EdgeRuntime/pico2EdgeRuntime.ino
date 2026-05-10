#include <Arduino.h>
#include "edge_fixture.h"
#include "edge_model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace {

constexpr uint32_t kBaudRate = 115200;
constexpr uint32_t kSerialAttachWaitMs = 8000;
constexpr int kWarmupRuns = 1;
constexpr int kMeasuredRuns = 10;

#ifndef EDGE_TENSOR_ARENA_SIZE
#define EDGE_TENSOR_ARENA_SIZE 425984
#endif
constexpr size_t kTensorArenaSize = EDGE_TENSOR_ARENA_SIZE;

const tflite::Model* g_model = nullptr;
tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_output = nullptr;
tflite::AllOpsResolver* g_resolver = nullptr;
uint8_t* g_tensor_arena_raw = nullptr;
uint8_t* g_tensor_arena = nullptr;
int8_t g_last_quantized_output[kFixtureNumClasses];
float g_last_dequantized_output[kFixtureNumClasses];
uint32_t g_last_latency_us = 0;

float absf(float value) {
  return value < 0.0f ? -value : value;
}

float compute_rmse(const float* reference, const float* candidate,
                   size_t count) {
  double sum = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double diff =
        static_cast<double>(reference[i]) - static_cast<double>(candidate[i]);
    sum += diff * diff;
  }
  return static_cast<float>(sqrt(sum / static_cast<double>(count)));
}

float compute_max_abs_error(const float* reference, const float* candidate,
                            size_t count) {
  float max_error = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    const float err = absf(reference[i] - candidate[i]);
    if (err > max_error) {
      max_error = err;
    }
  }
  return max_error;
}

float compute_cosine_similarity(const float* a, const float* b, size_t count) {
  double dot = 0.0;
  double norm_a = 0.0;
  double norm_b = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double da = static_cast<double>(a[i]);
    const double db = static_cast<double>(b[i]);
    dot += da * db;
    norm_a += da * da;
    norm_b += db * db;
  }
  const double denom = sqrt(norm_a) * sqrt(norm_b);
  if (denom <= 1e-12) {
    return 1.0f;
  }
  return static_cast<float>(dot / denom);
}

float compute_parity_percent(const float* reference, const float* candidate,
                             size_t count) {
  float max_abs_reference = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    const float mag = absf(reference[i]);
    if (mag > max_abs_reference) {
      max_abs_reference = mag;
    }
  }
  const float rmse = compute_rmse(reference, candidate, count);
  const float scale = max_abs_reference + 1e-8f;
  float parity = 1.0f - (rmse / scale);
  if (parity < 0.0f) {
    parity = 0.0f;
  }
  return parity * 100.0f;
}

int argmax(const float* values, size_t count) {
  size_t best = 0;
  for (size_t i = 1; i < count; ++i) {
    if (values[i] > values[best]) {
      best = i;
    }
  }
  return static_cast<int>(best);
}

void print_logits(const char* label, const float* values, size_t count) {
  Serial.print(label);
  Serial.print(": [");
  for (size_t i = 0; i < count; ++i) {
    Serial.print(values[i], 6);
    if (i + 1 < count) {
      Serial.print(", ");
    }
  }
  Serial.println("]");
}

void quantize_input() {
  if (g_input->type == kTfLiteInt8) {
    const float scale = g_input->params.scale;
    const int zero_point = g_input->params.zero_point;
    for (size_t i = 0; i < kFixtureInputElementCount; ++i) {
      const float scaled =
          (kFixtureInput[i] / scale) + static_cast<float>(zero_point);
      int value = static_cast<int>(roundf(scaled));
      if (value < -128) {
        value = -128;
      } else if (value > 127) {
        value = 127;
      }
      g_input->data.int8[i] = static_cast<int8_t>(value);
    }
  } else {
    for (size_t i = 0; i < kFixtureInputElementCount; ++i) {
      g_input->data.f[i] = kFixtureInput[i];
    }
  }
}

bool run_inference_once() {
  quantize_input();
  const uint32_t start_us = micros();
  const TfLiteStatus invoke_status = g_interpreter->Invoke();
  g_last_latency_us = micros() - start_us;
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed.");
    return false;
  }

  if (g_output->type == kTfLiteInt8) {
    const float scale = g_output->params.scale;
    const int zero_point = g_output->params.zero_point;
    for (size_t i = 0; i < kFixtureNumClasses; ++i) {
      const int8_t raw = g_output->data.int8[i];
      g_last_quantized_output[i] = raw;
      g_last_dequantized_output[i] =
          (static_cast<float>(raw) - static_cast<float>(zero_point)) * scale;
    }
  } else {
    for (size_t i = 0; i < kFixtureNumClasses; ++i) {
      g_last_quantized_output[i] = 0;
      g_last_dequantized_output[i] = g_output->data.f[i];
    }
  }
  return true;
}

void print_reference_summary() {
  Serial.println("=== Pico 2 TFLite Micro HAR Runtime ===");
  Serial.print("Fixture model: ");
  Serial.println(kFixtureModelName);
  Serial.print("Fixture dataset: ");
  Serial.println(kFixtureDatasetName);
  Serial.print("Fixture sample index: ");
  Serial.println(kFixtureSampleIndex);
  Serial.print("Model bytes: ");
  Serial.println(edge_model_len);
  Serial.print("Tensor arena size: ");
  Serial.println(kTensorArenaSize);
  Serial.print("Input scale/zp: ");
  Serial.print(edge_model_input_scale, 9);
  Serial.print(" / ");
  Serial.println(edge_model_input_zero_point);
  Serial.print("Output scale/zp: ");
  Serial.print(edge_model_output_scale, 9);
  Serial.print(" / ");
  Serial.println(edge_model_output_zero_point);
  Serial.print("Expected label: ");
  Serial.print(kFixtureExpectedLabel);
  Serial.print(" (");
  Serial.print(kFixtureClassNames[kFixtureExpectedLabel]);
  Serial.println(")");
  Serial.print("Desktop INT8 parity (%): ");
  Serial.println(kFixtureDesktopInt8ParityPercent, 6);
  Serial.print("Heap total bytes: ");
  Serial.println(rp2040.getTotalHeap());
}

bool setup_model() {
  tflite::InitializeTarget();
  g_model = tflite::GetModel(edge_model);
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Schema mismatch: ");
    Serial.print(g_model->version());
    Serial.print(" vs ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    return false;
  }

  g_resolver = new tflite::AllOpsResolver();
  const int free_heap_before_arena = rp2040.getFreeHeap();
  g_tensor_arena_raw =
      static_cast<uint8_t*>(malloc(kTensorArenaSize + 16));
  if (g_tensor_arena_raw == nullptr) {
    Serial.println("Arena allocation failed.");
    return false;
  }
  uintptr_t aligned = (reinterpret_cast<uintptr_t>(g_tensor_arena_raw) + 15u) &
                      ~static_cast<uintptr_t>(15u);
  g_tensor_arena = reinterpret_cast<uint8_t*>(aligned);
  const int free_heap_after_arena = rp2040.getFreeHeap();

  g_interpreter = new tflite::MicroInterpreter(
      g_model, *g_resolver, g_tensor_arena, kTensorArenaSize);
  const int free_heap_after_interpreter = rp2040.getFreeHeap();

  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed.");
    return false;
  }
  const int free_heap_after_allocate = rp2040.getFreeHeap();

  g_input = g_interpreter->input(0);
  g_output = g_interpreter->output(0);

  Serial.print("Heap free before arena: ");
  Serial.println(free_heap_before_arena);
  Serial.print("Heap free after arena alloc: ");
  Serial.println(free_heap_after_arena);
  Serial.print("Heap free after interpreter alloc: ");
  Serial.println(free_heap_after_interpreter);
  Serial.print("Heap free after AllocateTensors: ");
  Serial.println(free_heap_after_allocate);
  Serial.print("Heap delta total: ");
  Serial.println(free_heap_before_arena - free_heap_after_allocate);
  Serial.print("Interpreter arena used bytes: ");
  Serial.println(g_interpreter->arena_used_bytes());
  return true;
}

void run_measurements() {
  for (int i = 0; i < kWarmupRuns; ++i) {
    if (!run_inference_once()) {
      return;
    }
  }

  uint64_t latency_sum = 0;
  float parity_vs_pytorch_sum = 0.0f;
  float parity_vs_tflite_int8_sum = 0.0f;
  float rmse_vs_pytorch_sum = 0.0f;
  float max_error_vs_pytorch_sum = 0.0f;

  print_logits("Reference PyTorch logits", kFixtureExpectedPyTorchLogits,
               kFixtureNumClasses);
  print_logits("Reference TFLite INT8 logits", kFixtureExpectedTfliteInt8Logits,
               kFixtureNumClasses);

  for (int run = 0; run < kMeasuredRuns; ++run) {
    if (!run_inference_once()) {
      return;
    }

    const float parity_vs_pytorch = compute_parity_percent(
        kFixtureExpectedPyTorchLogits, g_last_dequantized_output,
        kFixtureNumClasses);
    const float parity_vs_tflite_int8 = compute_parity_percent(
        kFixtureExpectedTfliteInt8Logits, g_last_dequantized_output,
        kFixtureNumClasses);
    const float rmse_vs_pytorch = compute_rmse(
        kFixtureExpectedPyTorchLogits, g_last_dequantized_output,
        kFixtureNumClasses);
    const float max_error_vs_pytorch = compute_max_abs_error(
        kFixtureExpectedPyTorchLogits, g_last_dequantized_output,
        kFixtureNumClasses);

    latency_sum += g_last_latency_us;
    parity_vs_pytorch_sum += parity_vs_pytorch;
    parity_vs_tflite_int8_sum += parity_vs_tflite_int8;
    rmse_vs_pytorch_sum += rmse_vs_pytorch;
    max_error_vs_pytorch_sum += max_error_vs_pytorch;

    Serial.print("RUN ");
    Serial.print(run + 1);
    Serial.print(" | latency_us=");
    Serial.print(g_last_latency_us);
    Serial.print(" | free_heap=");
    Serial.print(rp2040.getFreeHeap());
    Serial.print(" | used_heap=");
    Serial.print(rp2040.getUsedHeap());
    Serial.print(" | arena_used=");
    Serial.print(g_interpreter->arena_used_bytes());
    Serial.print(" | top1_pico=");
    Serial.print(argmax(g_last_dequantized_output, kFixtureNumClasses));
    Serial.print(" | top1_ref=");
    Serial.print(kFixtureExpectedLabel);
    Serial.print(" | parity_pytorch_pct=");
    Serial.print(parity_vs_pytorch, 6);
    Serial.print(" | parity_tflite_int8_pct=");
    Serial.print(parity_vs_tflite_int8, 6);
    Serial.print(" | rmse_pytorch=");
    Serial.print(rmse_vs_pytorch, 6);
    Serial.print(" | max_abs_err_pytorch=");
    Serial.println(max_error_vs_pytorch, 6);
  }

  const float avg_latency =
      static_cast<float>(latency_sum) / static_cast<float>(kMeasuredRuns);
  const float avg_parity_vs_pytorch =
      parity_vs_pytorch_sum / static_cast<float>(kMeasuredRuns);
  const float avg_parity_vs_tflite_int8 =
      parity_vs_tflite_int8_sum / static_cast<float>(kMeasuredRuns);
  const float avg_rmse_vs_pytorch =
      rmse_vs_pytorch_sum / static_cast<float>(kMeasuredRuns);
  const float avg_max_error_vs_pytorch =
      max_error_vs_pytorch_sum / static_cast<float>(kMeasuredRuns);
  const float cosine_vs_pytorch = compute_cosine_similarity(
      kFixtureExpectedPyTorchLogits, g_last_dequantized_output,
      kFixtureNumClasses);

  print_logits("Last Pico logits", g_last_dequantized_output,
               kFixtureNumClasses);

  Serial.println("=== SUMMARY ===");
  Serial.print("avg_latency_us=");
  Serial.println(avg_latency, 3);
  Serial.print("min_flash_model_bytes=");
  Serial.println(edge_model_len);
  Serial.print("heap_total_bytes=");
  Serial.println(rp2040.getTotalHeap());
  Serial.print("heap_free_bytes=");
  Serial.println(rp2040.getFreeHeap());
  Serial.print("heap_used_bytes=");
  Serial.println(rp2040.getUsedHeap());
  Serial.print("tensor_arena_alloc_bytes=");
  Serial.println(kTensorArenaSize);
  Serial.print("tensor_arena_used_bytes=");
  Serial.println(g_interpreter->arena_used_bytes());
  Serial.print("avg_parity_vs_pytorch_pct=");
  Serial.println(avg_parity_vs_pytorch, 6);
  Serial.print("avg_parity_vs_tflite_int8_pct=");
  Serial.println(avg_parity_vs_tflite_int8, 6);
  Serial.print("avg_rmse_vs_pytorch=");
  Serial.println(avg_rmse_vs_pytorch, 6);
  Serial.print("avg_max_abs_err_vs_pytorch=");
  Serial.println(avg_max_error_vs_pytorch, 6);
  Serial.print("cosine_vs_pytorch=");
  Serial.println(cosine_vs_pytorch, 9);
  Serial.print("predicted_class=");
  const int predicted = argmax(g_last_dequantized_output, kFixtureNumClasses);
  Serial.print(predicted);
  Serial.print(" (");
  Serial.print(kFixtureClassNames[predicted]);
  Serial.println(")");
  Serial.print("expected_class=");
  Serial.print(kFixtureExpectedLabel);
  Serial.print(" (");
  Serial.print(kFixtureClassNames[kFixtureExpectedLabel]);
  Serial.println(")");
}

}  // namespace

void setup() {
  Serial.begin(kBaudRate);
  delay(1000);
  const uint32_t attach_start = millis();
  while (!Serial && (millis() - attach_start) < kSerialAttachWaitMs) {
    delay(10);
  }
  delay(750);

  print_reference_summary();
  if (!setup_model()) {
    Serial.println("Model setup failed.");
    return;
  }
}

void loop() {
  static bool has_run = false;
  if (has_run) {
    delay(1000);
    return;
  }
  has_run = true;
  run_measurements();
  Serial.println("=== DONE ===");
  Serial.flush();
  delay(1500);
  rp2040.rebootToBootloader();
}
