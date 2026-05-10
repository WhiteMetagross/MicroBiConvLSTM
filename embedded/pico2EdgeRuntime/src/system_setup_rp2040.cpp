#include <Arduino.h>

#include "tensorflow/lite/micro/system_setup.h"

extern "C" void DebugLog(const char* s) { Serial.print(s); }

namespace tflite {

void InitializeTarget() {}

}  // namespace tflite
