#include <Arduino.h>

#include <limits>

#include "peripherals/utility.h"

namespace peripherals {

void DelayMicroseconds(uint32_t delay) {
  constexpr uint16_t kArduinoAccurateDelay = 16383;
  while (delay > kArduinoAccurateDelay) {
    delayMicroseconds(kArduinoAccurateDelay);
    delay -= kArduinoAccurateDelay;
  }
  delayMicroseconds(delay);
}

void DelayMilliseconds(uint32_t amount) {
  constexpr uint32_t kArduinoMaxMilliseconds =
      std::numeric_limits<uint32_t>::max() / 1000u;
  while (amount > kArduinoMaxMilliseconds) {
    DelayMicroseconds(kArduinoMaxMilliseconds * 1000u);
    amount -= kArduinoMaxMilliseconds;
  }
  DelayMicroseconds(amount * 1000u);
}

uint32_t MicrosecondsCounter() { return micros(); }

uint32_t MillisecondsCounter() { return millis(); }

void DebugOutput(const char* s) { Serial.println(s); }

TimestampBuffer::TimestampBuffer() : insert_index_(0), show_index_(0), entries_{} {}

TimestampBuffer& TimestampBuffer::Instance() {
  static TimestampBuffer instance;
  return instance;
}

void TimestampBuffer::Insert(char c) {
  size_t next_index = (insert_index_ + 1u) % kNumEntries;
  if (next_index == show_index_) {
    if (entries_[insert_index_].c_ != '\0') {
      entries_[insert_index_].c_ = '\0';
      entries_[insert_index_].timestamp_us_ = micros();
    }
  } else {
    entries_[insert_index_].c_ = c;
    entries_[insert_index_].timestamp_us_ = micros();
    insert_index_ = next_index;
  }
}

void TimestampBuffer::Show() {
  while (show_index_ != insert_index_) {
    if (entries_[show_index_].c_ == '\0') {
      Serial.print(entries_[show_index_].timestamp_us_);
      Serial.println(": *** TimestampBuffer Overflow ***");
      break;
    }
    Serial.print(entries_[show_index_].timestamp_us_);
    Serial.print(": ");
    Serial.println(entries_[show_index_].c_);
    show_index_ = (show_index_ + 1u) % kNumEntries;
  }
  show_index_ = insert_index_;
}

}  // namespace peripherals
