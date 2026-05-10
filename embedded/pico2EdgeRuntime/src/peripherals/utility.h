#ifndef PERIPHERALS_UTILITY_H_
#define PERIPHERALS_UTILITY_H_

#include <cstddef>
#include <cstdint>

namespace peripherals {

void DelayMicroseconds(uint32_t delay);
void DelayMilliseconds(uint32_t delay);

uint32_t MicrosecondsCounter();
uint32_t MillisecondsCounter();

void DebugOutput(const char* s);

class TimestampBuffer {
 public:
  static TimestampBuffer& Instance();

  void Insert(char c);
  void Show();

 private:
  TimestampBuffer();

  size_t insert_index_;
  size_t show_index_;

  static constexpr size_t kNumEntries = 100;
  struct Entry {
    uint32_t timestamp_us_;
    char c_;
  } entries_[kNumEntries];
};

}  // namespace peripherals

#endif  // PERIPHERALS_UTILITY_H_
