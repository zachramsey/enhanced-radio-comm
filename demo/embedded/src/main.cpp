#include <Arduino.h>
#include <cmath>
#include "i2c_device.h"
//#include <Wire.h>
#include <VL53L0X.h>
#include "InterruptPWMInput.h"

//---------- Pin Allocations ----------
//Reciever inputs
const int Xin = 0;       //Ch1
const int Yin = 1;       //Ch2
const int RotateIn = 2;  //Ch4
const int WeaponIn = 3;  //Unused

//Motor PWM and Direction Pins
//A*****B
// *****
//  ***
//   C
/**
 *   C    B    A
 *      
 *  uC    Gyro
 */
//Side AB being the front.
const int Aout = 9;
const int Bout = 8;
const int Cout = 7;
const int AdirOut = 12; 
const int BdirOut = 11; 
const int CdirOut = 10; 

//ToF Sensor Control Pins
const int TofAAddr = 0x54;  //I2C Addresses for ToFA
const int TofAXShut = 13;
const int TofAGPIO1 = 14;

const int TofBAddr = 0x56;  //I2C Addresses for ToFB
const int TofBXShut = 15;
const int TofBGPIO1 = 16;

//VL53L0X Sensors
VL53L0X ToFA;
VL53L0X ToFB;

//Accelerometer
//GY-521
const int GyroAddr = 0x68;  //I2C address for Gyroscope

//----------PWM Inputs----------
RCInput rcX(0); //X Axis
RCInput rcY(1); //Y Axis
RCInput rcR(2); //Rotation Axis

//function declarations
void waitForSerial(unsigned long timeout=10000);
void debugRemoteInputs(int uX, int uY, int uR);
void debugWheelDirections(int wA, int wB, int wC);
void debugWheelSpeed(int wA, int wB, int wC);
void setupSensors();

//---------- Working Variables ----------
int16_t uX = 0, uY = 0, uR = 0;
int16_t wA = 0, wB = 0, wC = 0;

//---------- Debug Variables ----------
unsigned long lastDebugTime = 0;  // Tracks the last time the function was called
const unsigned long debugInterval = 100;  // 500ms interval

void setup() {
  //Initialize Serial for debugging
  Serial.begin(115200);
  waitForSerial();
  //Initialize I2C for VL53L0X
  Wire.begin();
  setupSensors();
  
  //Configure PWM and Direction Pins
  pinMode(Aout, OUTPUT);
  pinMode(Bout, OUTPUT);
  pinMode(Cout, OUTPUT);
  pinMode(AdirOut, OUTPUT);
  pinMode(BdirOut, OUTPUT);
  pinMode(CdirOut, OUTPUT);

  //begin the interupt timers for the PWM inputs
  rcX.begin();
  rcY.begin();
  rcR.begin();

  Serial.print("Setup Done"); 
}

void loop() {
  //Read inputs: Max value of -500 and 500
  //TODO Test manual interrupt code
  //TODO look into hardware fixes for the pulseIn function.
  uX = rcX.getValue(); //X Axis
  rcX.isSignalLost(30); //define signal loss timeout
  uY = rcY.getValue(); //Y Axis
  rcY.isSignalLost(30); //define signal loss timeout
  uR = rcR.getValue(); //Rotation Axis
  rcR.isSignalLost(30); //define signal loss timeout

  //Read distances from ToF sensors
  //uint16_t distanceA = ToFA.readRangeSingleMillimeters();
  //uint16_t distanceB = ToFB.readRangeSingleMillimeters();
  //Serial.print("ToF-A Reading: ");
  //Serial.print(distanceA);
  //Serial.print(" ToF-B Reading: ");
  //Serial.print(distanceB);
  //Serial.println();


  //Adjust rotation based on ToF sensors
  //if (distanceA < 2000 && distanceB > 2000) uR += 200;
  //if (distanceA > 2000 && distanceB < 2000) uR -= 200;
  //if (distanceA < 2000 && distanceB < 2000) uR += (distance1 - distance2) / 10;

  //Calculate wheel thrusts
  wA = round(((0.3333 * uX) + (0.5774 * uY) + (0.3333 * uR)) / 5);
  wB = round(((0.3333 * uX) - (0.5774 * uY) + (0.3333 * uR)) / 5);
  wC = round(((-0.6667 * uX) + (0.3333 * uR)) / 5);

  
  // Set motor directions
  digitalWrite(AdirOut, wA < 0);
  digitalWrite(BdirOut, wB < 0);
  digitalWrite(CdirOut, wC < 0);
  
  //Normalize and scale PWM values
  wA = abs(wA);
  wB = abs(wB);
  wC = abs(wC);
  int maxVal = max(max(wA, wB), wC);
  if (maxVal > 100) {
    wA = ((wA * 100) / maxVal);
    wB = ((wB * 100) / maxVal);
    wC = ((wC * 100) / maxVal);
  }

  if (millis() - lastDebugTime >= debugInterval) {
    lastDebugTime = millis();  // reset timer
    debugRemoteInputs(uX, uY, uR);  // your debug function
    debugWheelDirections(wA, wB, wC);
    //debugWheelSpeed(wA, wB, wC);
  }

  //Set wheel speeds
  wA = wA * 2.55;
  wB = wB * 2.55;
  wC = wC * 2.55; 

  analogWrite(Aout, wA);
  analogWrite(Bout, wB);
  analogWrite(Cout, wC);
}

void setupSensors(){
  //Configure ToF sensors
  pinMode(TofAXShut, OUTPUT);
  pinMode(TofBXShut, OUTPUT);

  //Reset and configure ToF A
  digitalWrite(TofAXShut, LOW);
  delay(10);
  digitalWrite(TofAXShut, HIGH);
  delay(10);
  ToFA.setAddress(TofAAddr);
  ToFA.init();
  ToFA.setMeasurementTimingBudget(500 * 1000);

  //Reset and configure ToF B
  digitalWrite(TofBXShut, LOW);
  delay(10);
  digitalWrite(TofBXShut, HIGH);
  delay(10);
  ToFB.setAddress(TofBAddr);
  ToFB.init();
  ToFB.setMeasurementTimingBudget(500 * 1000);
}

//Debug statement
void waitForSerial(unsigned long timeout) { //Waits for a serial connection before pursuing with the program (FOR DEBUG USE ONLY)
  unsigned long startTime = millis();
  // Wait for Serial to connect
  Serial.println("Waiting for serial connection...");
  while (!Serial) {
      if (millis() - startTime > timeout) {
          Serial.println("Serial connection timeout. Continuing without serial...");
          return;
      }
  }
  Serial.println("Serial connection established.");
}//END waitForSerial

//debug the remote inputs.
void debugRemoteInputs(int uX, int uY, int uR) {
  Serial.print("uX: ");
  Serial.print(uX);
  Serial.print(" | uY: ");
  Serial.print(uY);
  Serial.print(" | uR: ");
  Serial.println(uR);
}

//debug wheel directions
void debugWheelDirections(int wA, int wB, int wC){
  Serial.print("A-Dir: ");
  Serial.print(wA);
  Serial.print(" B-Dir: ");
  Serial.print(wB);
  Serial.print(" C-Dir: ");
  Serial.print(wC);
  Serial.println();
}
//debug wheel speeds
void debugWheelSpeed(int wA, int wB, int wC){
  Serial.print("wA: ");
  Serial.print(wA);
  Serial.print(" | wB: ");
  Serial.print(wB);
  Serial.print(" | wC: ");
  Serial.print(wC);
  Serial.println();
}