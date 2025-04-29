#ifndef INTERRUPTPWMINPUT_H
#define INTERRUPTPWMINPUT_H

#include <Arduino.h>

class RCInput {
public:
  RCInput(uint8_t pin);
  void begin();
  int getValue(); // Returns the current mapped value (-500 to 500)
  bool isSignalLost(unsigned long timeout = 100); // Optional failsafe

private:
  uint8_t _pin;
  volatile uint32_t _pulseStart;
  volatile int _value;
  volatile uint32_t _lastPulseTime;

  static void handleInterrupt0();
  static void handleInterrupt1();
  static void handleInterrupt2();

  static RCInput* instances[3];
  static void attachInterruptSafe(uint8_t pin, void (*isr)());
};

#endif
