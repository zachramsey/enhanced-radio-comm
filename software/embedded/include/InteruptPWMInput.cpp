#include "InteruptPWMInput.h"

RCInput* RCInput::instances[3] = {nullptr, nullptr, nullptr};

RCInput::RCInput(uint8_t pin) : _pin(pin), _pulseStart(0), _value(0), _lastPulseTime(0) {}

void RCInput::begin() {
  pinMode(_pin, INPUT);

  if (instances[0] == nullptr) {
    instances[0] = this;
    attachInterruptSafe(_pin, handleInterrupt0);
  } else if (instances[1] == nullptr) {
    instances[1] = this;
    attachInterruptSafe(_pin, handleInterrupt1);
  } else if (instances[2] == nullptr) {
    instances[2] = this;
    attachInterruptSafe(_pin, handleInterrupt2);
  } else {
    // Too many instances — extend array if needed
  }
}

int RCInput::getValue() {
  return _value;
}

bool RCInput::isSignalLost(unsigned long timeout) {
  return (millis() - _lastPulseTime) > timeout;
}

void RCInput::attachInterruptSafe(uint8_t pin, void (*isr)()) {
  attachInterrupt(digitalPinToInterrupt(pin), isr, CHANGE);
}

void RCInput::handleInterrupt0() {
  if (instances[0]) {
    RCInput* inst = instances[0];
    if (digitalRead(inst->_pin) == HIGH) {
      inst->_pulseStart = micros();
    } else {
      uint32_t duration = micros() - inst->_pulseStart;
      if (duration >= 1000 && duration <= 2000) {
        inst->_value = constrain(duration - 1500, -500, 500);
        inst->_lastPulseTime = millis();
      }
    }
  }
}

void RCInput::handleInterrupt1() {
  if (instances[1]) {
    RCInput* inst = instances[1];
    if (digitalRead(inst->_pin) == HIGH) {
      inst->_pulseStart = micros();
    } else {
      uint32_t duration = micros() - inst->_pulseStart;
      if (duration >= 1000 && duration <= 2000) {
        inst->_value = constrain(duration - 1500, -500, 500);
        inst->_lastPulseTime = millis();
      }
    }
  }
}

void RCInput::handleInterrupt2() {
  if (instances[2]) {
    RCInput* inst = instances[2];
    if (digitalRead(inst->_pin) == HIGH) {
      inst->_pulseStart = micros();
    } else {
      uint32_t duration = micros() - inst->_pulseStart;
      if (duration >= 1000 && duration <= 2000) {
        inst->_value = constrain(duration - 1500, -500, 500);
        inst->_lastPulseTime = millis();
      }
    }
  }
}
