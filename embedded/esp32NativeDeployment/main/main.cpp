#include <cmath>
#include <cstdint>
#include <cstdio>
#include <new>

#include "edge_fixture.h"
#include "edge_model_data.h"
#include "edge_ops_config.h"

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_rom_sys.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#undef printf
#define printf esp_rom_printf

namespace {

constexpr char kTag[] = "microbi_native";
constexpr int kWarmupRuns = 1;
constexpr int kMeasuredRuns = 10;

#ifndef EDGE_TENSOR_ARENA_SIZE
#define EDGE_TENSOR_ARENA_SIZE (128 * 1024)
#endif

constexpr size_t kTensorArenaSize = EDGE_TENSOR_ARENA_SIZE;

#ifndef EDGE_USE_STATIC_ARENA
#define EDGE_USE_STATIC_ARENA 0
#endif

#if EDGE_USE_STATIC_ARENA
static uint8_t g_static_tensor_arena[kTensorArenaSize];
#endif

uint8_t* g_tensor_arena = nullptr;
const char* g_tensor_arena_heap = "unallocated";

const tflite::Model* g_model = nullptr;
tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_output = nullptr;

float g_last_output[kFixtureNumClasses];
int8_t g_last_output_int8[kFixtureNumClasses];
int64_t g_last_latency_us = 0;

uint32_t internal_heap_total_bytes() {
  return heap_caps_get_total_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
}

uint32_t internal_heap_free_bytes() {
  return heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
}

uint32_t internal_heap_used_bytes() {
  return internal_heap_total_bytes() - internal_heap_free_bytes();
}

size_t largest_internal_8bit_block_bytes() {
  return heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
}

size_t largest_iram_8bit_block_bytes() {
  return heap_caps_get_largest_free_block(MALLOC_CAP_IRAM_8BIT);
}

float absf(float value) {
  return value < 0.0f ? -value : value;
}

const char* format_float(char* buffer, size_t buffer_size, float value) {
  snprintf(buffer, buffer_size, "%.6f", static_cast<double>(value));
  return buffer;
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
  printf("%s: [", label);
  for (size_t i = 0; i < count; ++i) {
    char value_buffer[32];
    printf("%s", format_float(value_buffer, sizeof(value_buffer), values[i]));
    if (i + 1 < count) {
      printf(", ");
    }
  }
  printf("]\n");
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
  const int64_t start_us = esp_timer_get_time();
  const TfLiteStatus status = g_interpreter->Invoke();
  g_last_latency_us = esp_timer_get_time() - start_us;
  if (status != kTfLiteOk) {
    printf("Invoke failed.\n");
    return false;
  }

  if (g_output->type == kTfLiteInt8) {
    const float scale = g_output->params.scale;
    const int zero_point = g_output->params.zero_point;
    for (size_t i = 0; i < kFixtureNumClasses; ++i) {
      const int8_t raw = g_output->data.int8[i];
      g_last_output_int8[i] = raw;
      g_last_output[i] =
          (static_cast<float>(raw) - static_cast<float>(zero_point)) * scale;
    }
  } else {
    for (size_t i = 0; i < kFixtureNumClasses; ++i) {
      g_last_output_int8[i] = 0;
      g_last_output[i] = g_output->data.f[i];
    }
  }
  return true;
}

bool setup_model() {
  tflite::InitializeTarget();

  printf("=== ESP32 Native TFLite Micro HAR Runtime ===\n");
  printf("Mode: Native ESP-IDF\n");
  printf("Fixture model: %s\n", kFixtureModelName);
  printf("Fixture dataset: %s\n", kFixtureDatasetName);
  printf("Fixture sample index: %d\n", kFixtureSampleIndex);
  printf("Model bytes: %u\n", edge_model_len);
  printf("Tensor arena bytes (%s internal SRAM): %u\n",
         EDGE_USE_STATIC_ARENA ? "static" : "dynamic",
         static_cast<unsigned>(kTensorArenaSize));
  printf("Heap total internal bytes before setup: %u\n",
         static_cast<unsigned>(internal_heap_total_bytes()));
  printf("Heap free internal bytes before setup: %u\n",
         static_cast<unsigned>(internal_heap_free_bytes()));
  printf("Largest internal 8-bit block before setup: %u\n",
         static_cast<unsigned>(largest_internal_8bit_block_bytes()));
  printf("Largest IRAM 8-bit block before setup: %u\n",
         static_cast<unsigned>(largest_iram_8bit_block_bytes()));

  const esp_err_t uart_release = uart_driver_delete(UART_NUM_0);
  printf("UART driver delete status: %d\n", static_cast<int>(uart_release));
  printf("Heap free internal bytes after UART release: %u\n",
         static_cast<unsigned>(internal_heap_free_bytes()));
  printf("Largest internal 8-bit block after UART release: %u\n",
         static_cast<unsigned>(largest_internal_8bit_block_bytes()));
  printf("Largest IRAM 8-bit block after UART release: %u\n",
         static_cast<unsigned>(largest_iram_8bit_block_bytes()));

  const esp_err_t task_wdt_release = esp_task_wdt_deinit();
  printf("Task WDT deinit status: %d\n", static_cast<int>(task_wdt_release));

  #if EDGE_USE_STATIC_ARENA
    g_tensor_arena = g_static_tensor_arena;
    g_tensor_arena_heap = "static_internal";
  #else
    g_tensor_arena = reinterpret_cast<uint8_t*>(
        heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    g_tensor_arena_heap = "dram_8bit";
    if (g_tensor_arena == nullptr) {
      g_tensor_arena = reinterpret_cast<uint8_t*>(
          heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_IRAM_8BIT));
      g_tensor_arena_heap = "iram_8bit";
    }
    if (g_tensor_arena == nullptr) {
      printf("Failed to allocate tensor arena from internal SRAM.\n");
      printf("Largest internal 8-bit block at allocation failure: %u\n",
             static_cast<unsigned>(largest_internal_8bit_block_bytes()));
      printf("Largest IRAM 8-bit block at allocation failure: %u\n",
             static_cast<unsigned>(largest_iram_8bit_block_bytes()));
      return false;
    }
  #endif
  printf("Tensor arena heap source: %s\n", g_tensor_arena_heap);
  printf("Heap free internal bytes after arena alloc: %u\n",
         static_cast<unsigned>(internal_heap_free_bytes()));
  printf("Largest internal 8-bit block after arena alloc: %u\n",
         static_cast<unsigned>(largest_internal_8bit_block_bytes()));
  printf("Largest IRAM 8-bit block after arena alloc: %u\n",
         static_cast<unsigned>(largest_iram_8bit_block_bytes()));
  printf("Tensor arena pointer alignment mod 16: %u\n",
         static_cast<unsigned>(reinterpret_cast<uintptr_t>(g_tensor_arena) & 0xF));

  g_model = tflite::GetModel(edge_model);
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    printf("Schema mismatch: %u vs %d\n",
           static_cast<unsigned>(g_model->version()), TFLITE_SCHEMA_VERSION);
    return false;
  }

  static tflite::MicroMutableOpResolver<kEdgeResolverOpCount> resolver;
  static bool resolver_initialized = false;
  if (!resolver_initialized) {
    RegisterEdgeOps(resolver);
    resolver_initialized = true;
  }

  const uint32_t heap_free_before_allocate = internal_heap_free_bytes();
  g_interpreter =
      new tflite::MicroInterpreter(g_model, resolver, g_tensor_arena,
                                   kTensorArenaSize);
  if (g_interpreter == nullptr) {
    printf("Failed to construct MicroInterpreter.\n");
    return false;
  }

  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    printf("AllocateTensors failed.\n");
    return false;
  }

  g_input = g_interpreter->input(0);
  g_output = g_interpreter->output(0);

  char input_scale_buffer[32];
  char output_scale_buffer[32];
  char desktop_parity_buffer[32];
  printf("Input scale/zp: %s / %d\n",
         format_float(input_scale_buffer, sizeof(input_scale_buffer),
                      g_input->params.scale),
         static_cast<int>(g_input->params.zero_point));
  printf("Output scale/zp: %s / %d\n",
         format_float(output_scale_buffer, sizeof(output_scale_buffer),
                      g_output->params.scale),
         static_cast<int>(g_output->params.zero_point));
  printf("Expected label: %d (%s)\n", kFixtureExpectedLabel,
         kFixtureClassNames[kFixtureExpectedLabel]);
  printf("Desktop INT8 parity (%%): %s\n",
         format_float(desktop_parity_buffer, sizeof(desktop_parity_buffer),
                      kFixtureDesktopInt8ParityPercent));
  printf("Heap free internal bytes after AllocateTensors: %u\n",
         static_cast<unsigned>(internal_heap_free_bytes()));
  printf("Heap delta internal bytes: %u\n",
         static_cast<unsigned>(heap_free_before_allocate -
                               internal_heap_free_bytes()));
  printf("Largest internal 8-bit block after AllocateTensors: %u\n",
         static_cast<unsigned>(largest_internal_8bit_block_bytes()));
  printf("Interpreter arena used bytes: %u\n",
         static_cast<unsigned>(g_interpreter->arena_used_bytes()));
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
        kFixtureExpectedPyTorchLogits, g_last_output, kFixtureNumClasses);
    const float parity_vs_tflite_int8 = compute_parity_percent(
        kFixtureExpectedTfliteInt8Logits, g_last_output, kFixtureNumClasses);
    const float rmse_vs_pytorch = compute_rmse(
        kFixtureExpectedPyTorchLogits, g_last_output, kFixtureNumClasses);
    const float max_error_vs_pytorch = compute_max_abs_error(
        kFixtureExpectedPyTorchLogits, g_last_output, kFixtureNumClasses);

    latency_sum += static_cast<uint64_t>(g_last_latency_us);
    parity_vs_pytorch_sum += parity_vs_pytorch;
    parity_vs_tflite_int8_sum += parity_vs_tflite_int8;
    rmse_vs_pytorch_sum += rmse_vs_pytorch;
    max_error_vs_pytorch_sum += max_error_vs_pytorch;

    char parity_pytorch_buffer[32];
    char parity_tflite_buffer[32];
    char rmse_buffer[32];
    char max_error_buffer[32];
    printf(
        "RUN %d | latency_us=%lld | heap_free_internal=%u | heap_used_internal=%u | "
        "arena_used=%u | top1_esp32=%d | top1_ref=%d | parity_pytorch_pct=%s | "
        "parity_tflite_int8_pct=%s | rmse_pytorch=%s | "
        "max_abs_err_pytorch=%s\n",
        run + 1, static_cast<long long>(g_last_latency_us),
        static_cast<unsigned>(internal_heap_free_bytes()),
        static_cast<unsigned>(internal_heap_used_bytes()),
        static_cast<unsigned>(g_interpreter->arena_used_bytes()),
        argmax(g_last_output, kFixtureNumClasses), kFixtureExpectedPyTorchTop1,
        format_float(parity_pytorch_buffer, sizeof(parity_pytorch_buffer),
                     parity_vs_pytorch),
        format_float(parity_tflite_buffer, sizeof(parity_tflite_buffer),
                     parity_vs_tflite_int8),
        format_float(rmse_buffer, sizeof(rmse_buffer), rmse_vs_pytorch),
        format_float(max_error_buffer, sizeof(max_error_buffer),
                     max_error_vs_pytorch));
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
      kFixtureExpectedPyTorchLogits, g_last_output, kFixtureNumClasses);
  const int predicted_class = argmax(g_last_output, kFixtureNumClasses);
  char avg_latency_buffer[32];
  char avg_parity_pytorch_buffer[32];
  char avg_parity_tflite_buffer[32];
  char avg_rmse_buffer[32];
  char avg_max_error_buffer[32];
  char cosine_buffer[32];

  print_logits("Last ESP32 logits", g_last_output, kFixtureNumClasses);
  printf("=== SUMMARY ===\n");
  printf("avg_latency_us=%s\n",
         format_float(avg_latency_buffer, sizeof(avg_latency_buffer),
                      avg_latency));
  printf("mode=native_esp_idf\n");
  printf("heap_total_internal_bytes=%u\n",
         static_cast<unsigned>(internal_heap_total_bytes()));
  printf("heap_free_internal_bytes=%u\n",
         static_cast<unsigned>(internal_heap_free_bytes()));
  printf("heap_used_internal_bytes=%u\n",
         static_cast<unsigned>(internal_heap_used_bytes()));
  printf("largest_internal_block_bytes=%u\n",
         static_cast<unsigned>(largest_internal_8bit_block_bytes()));
  printf("tensor_arena_alloc_bytes=%u\n",
         static_cast<unsigned>(kTensorArenaSize));
  printf("tensor_arena_used_bytes=%u\n",
         static_cast<unsigned>(g_interpreter->arena_used_bytes()));
  printf("avg_parity_vs_pytorch_pct=%s\n",
         format_float(avg_parity_pytorch_buffer,
                      sizeof(avg_parity_pytorch_buffer),
                      avg_parity_vs_pytorch));
  printf("avg_parity_vs_tflite_int8_pct=%s\n",
         format_float(avg_parity_tflite_buffer,
                      sizeof(avg_parity_tflite_buffer),
                      avg_parity_vs_tflite_int8));
  printf("avg_rmse_vs_pytorch=%s\n",
         format_float(avg_rmse_buffer, sizeof(avg_rmse_buffer),
                      avg_rmse_vs_pytorch));
  printf("avg_max_abs_err_vs_pytorch=%s\n",
         format_float(avg_max_error_buffer, sizeof(avg_max_error_buffer),
                      avg_max_error_vs_pytorch));
  printf("cosine_vs_pytorch=%s\n",
         format_float(cosine_buffer, sizeof(cosine_buffer),
                      cosine_vs_pytorch));
  printf("predicted_class=%d (%s)\n", predicted_class,
         kFixtureClassNames[predicted_class]);
  printf("expected_class=%d (%s)\n", kFixtureExpectedLabel,
         kFixtureClassNames[kFixtureExpectedLabel]);
  printf("=== DONE ===\n");
  fflush(stdout);
}

}  // namespace

extern "C" void app_main(void) {
  esp_log_level_set("*", ESP_LOG_INFO);
  vTaskDelay(pdMS_TO_TICKS(1200));

  if (!setup_model()) {
    printf("=== FAILED ===\n");
    fflush(stdout);
    return;
  }

  run_measurements();
}
