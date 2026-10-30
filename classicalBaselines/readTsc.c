#include <stdint.h>
#include <x86intrin.h>

uint64_t tsc_begin(void) {
    unsigned int low;
    unsigned int high;
    __asm__ volatile("cpuid\n\trdtsc" : "=a"(low), "=d"(high) : "a"(0) : "rbx", "rcx");
    return ((uint64_t)high << 32) | low;
}

uint64_t tsc_end(void) {
    unsigned int low;
    unsigned int high;
    unsigned int auxiliary;
    __asm__ volatile("rdtscp" : "=a"(low), "=d"(high), "=c"(auxiliary));
    __asm__ volatile("cpuid" : : "a"(0) : "rbx", "rcx", "rdx");
    return ((uint64_t)high << 32) | low;
}
