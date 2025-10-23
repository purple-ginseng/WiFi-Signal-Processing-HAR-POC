/*
  ESP32-S3 Complete Diagnostic Test

  This will test:
  1. Serial communication
  2. Board identification
  3. RGB LED with Adafruit library
  4. Simple LED with digitalWrite

  Upload this and check Serial Monitor at 115200 baud
*/

#include <Adafruit_NeoPixel.h>

void setup() {
  Serial.begin(115200);
  delay(3000);  // Wait for serial to stabilize

  Serial.println("\n\n===========================================");
  Serial.println("ESP32-S3 DIAGNOSTIC TEST");
  Serial.println("===========================================");

  // Board info
  Serial.printf("Chip Model: %s\n", ESP.getChipModel());
  Serial.printf("Chip Revision: %d\n", ESP.getChipRevision());
  Serial.printf("CPU Frequency: %d MHz\n", ESP.getCpuFreqMHz());
  Serial.printf("Flash Size: %d bytes\n", ESP.getFlashChipSize());
  Serial.println("===========================================\n");

  // Test 1: RGB LED on GPIO 38 (most common)
  Serial.println("TEST 1: RGB LED on GPIO 38 (Adafruit NeoPixel)");
  testRGB(38);

  // Test 2: RGB LED on GPIO 48 (alternative)
  Serial.println("\nTEST 2: RGB LED on GPIO 48 (Adafruit NeoPixel)");
  testRGB(48);

  // Test 3: Simple LED on common pins
  Serial.println("\nTEST 3: Simple LED on common pins");
  testSimpleLED();

  Serial.println("\n===========================================");
  Serial.println("DIAGNOSTIC COMPLETE");
  Serial.println("Did you see ANY LED activity?");
  Serial.println("===========================================\n");
}

void testRGB(int pin) {
  Serial.printf("  Creating NeoPixel on GPIO %d...\n", pin);

  Adafruit_NeoPixel pixel(1, pin, NEO_GRB + NEO_KHZ800);
  pixel.begin();
  pixel.setBrightness(255);

  Serial.println("  Testing RED...");
  pixel.setPixelColor(0, pixel.Color(255, 0, 0));
  pixel.show();
  delay(1000);

  Serial.println("  Testing GREEN...");
  pixel.setPixelColor(0, pixel.Color(0, 255, 0));
  pixel.show();
  delay(1000);

  Serial.println("  Testing BLUE...");
  pixel.setPixelColor(0, pixel.Color(0, 0, 255));
  pixel.show();
  delay(1000);

  Serial.println("  OFF");
  pixel.clear();
  pixel.show();
  delay(500);

  Serial.printf("  GPIO %d test complete.\n", pin);
}

void testSimpleLED() {
  int pins[] = {2, 13, 15};

  for (int i = 0; i < 3; i++) {
    int pin = pins[i];
    pinMode(pin, OUTPUT);

    Serial.printf("  Testing GPIO %d: ", pin);

    for (int j = 0; j < 3; j++) {
      digitalWrite(pin, HIGH);
      delay(200);
      digitalWrite(pin, LOW);
      delay(200);
    }

    Serial.println("done");
    pinMode(pin, INPUT);
  }
}

void loop() {
  // Continuous blink on GPIO 38 to verify if it's working
  static bool state = false;
  static unsigned long lastBlink = 0;

  if (millis() - lastBlink > 1000) {
    Adafruit_NeoPixel pixel(1, 38, NEO_GRB + NEO_KHZ800);
    pixel.begin();
    pixel.setBrightness(100);

    if (state) {
      pixel.setPixelColor(0, pixel.Color(0, 0, 255));  // Blue
    } else {
      pixel.clear();
    }
    pixel.show();

    state = !state;
    lastBlink = millis();

    Serial.printf("GPIO 38 loop blink: %s\n", state ? "ON" : "OFF");
  }
}
