/*
  ESP32-S3 RGB LED Test using Adafruit NeoPixel Library
  Tests common RGB LED pins with Adafruit library

  IMPORTANT: Install "Adafruit NeoPixel" library first
  (Tools -> Manage Libraries -> Search "Adafruit NeoPixel")
*/

#include <Adafruit_NeoPixel.h>

// Common RGB LED pins on ESP32-S3 boards
int test_pins[] = {38, 48, 2, 8, 18};
int num_pins = 5;

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("\n===========================================");
  Serial.println("ESP32-S3 RGB LED Test (Adafruit NeoPixel)");
  Serial.println("===========================================");
  Serial.println("Testing pins: 38, 48, 2, 8, 18");
  Serial.println("Watch your board for LED activity!");
  Serial.println("===========================================\n");
}

void loop() {
  for (int i = 0; i < num_pins; i++) {
    int pin = test_pins[i];

    // Create NeoPixel object for this pin
    Adafruit_NeoPixel pixel(1, pin, NEO_GRB + NEO_KHZ800);
    pixel.begin();
    pixel.setBrightness(255);  // Full brightness

    Serial.printf("\n========== Testing GPIO %d ==========\n", pin);

    // Test RED
    Serial.printf("GPIO %d: RED (full brightness)\n", pin);
    pixel.setPixelColor(0, pixel.Color(255, 0, 0));
    pixel.show();
    delay(1500);

    // Test GREEN
    Serial.printf("GPIO %d: GREEN (full brightness)\n", pin);
    pixel.setPixelColor(0, pixel.Color(0, 255, 0));
    pixel.show();
    delay(1500);

    // Test BLUE
    Serial.printf("GPIO %d: BLUE (full brightness)\n", pin);
    pixel.setPixelColor(0, pixel.Color(0, 0, 255));
    pixel.show();
    delay(1500);

    // Test WHITE (all colors)
    Serial.printf("GPIO %d: WHITE (all colors)\n", pin);
    pixel.setPixelColor(0, pixel.Color(255, 255, 255));
    pixel.show();
    delay(1500);

    // OFF
    Serial.printf("GPIO %d: OFF\n", pin);
    pixel.clear();
    pixel.show();
    delay(1000);
  }

  Serial.println("\n===========================================");
  Serial.println("SCAN COMPLETE!");
  Serial.println("Did you see any LED light up?");
  Serial.println("If YES: Note which GPIO number worked");
  Serial.println("If NO: Your board might not have RGB LED");
  Serial.println("Restarting in 5 seconds...");
  Serial.println("===========================================\n");
  delay(5000);
}
