/*
  Simple LED Test for ESP32-S3
  Test if the WS2812 RGB LED works on GPIO 38
*/

const int LED_PIN = 38;

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("===========================================");
  Serial.println("ESP32-S3 WS2812 LED Test");
  Serial.println("===========================================");
  Serial.printf("Testing LED on GPIO %d\n", LED_PIN);
  Serial.println("Cycling through Red, Green, Blue...");
  Serial.println("===========================================\n");
}

void loop() {
  // Test Red
  Serial.println("RED");
  neopixelWrite(LED_PIN, 255, 0, 0);  // Full brightness red
  delay(1000);

  // Test Green
  Serial.println("GREEN");
  neopixelWrite(LED_PIN, 0, 255, 0);  // Full brightness green
  delay(1000);

  // Test Blue
  Serial.println("BLUE");
  neopixelWrite(LED_PIN, 0, 0, 255);  // Full brightness blue
  delay(1000);

  // Off
  Serial.println("OFF");
  neopixelWrite(LED_PIN, 0, 0, 0);
  delay(1000);
}
