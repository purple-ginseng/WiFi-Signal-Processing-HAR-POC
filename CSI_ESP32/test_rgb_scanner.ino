/*
  VCC-GND Studio YD-ESP32-S3 RGB LED Test

  IMPORTANT: Your board has RGB LED on GPIO 48

  **HARDWARE REQUIREMENT**
  The RGB jumper on your board MUST be soldered/closed!
  Look for a jumper labeled "RGB" near the LED and solder it closed.
  Without this, the LED will NOT work!

  Reference: https://mischianti.org/vcc-gnd-studio-yd-esp32-s3-devkitc-1-clone-high-resolution-pinout-and-specs/
*/

// VCC-GND YD-ESP32-S3: RGB LED is on GPIO 48
const int RGB_LED_PIN = 48;

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("\n===========================================");
  Serial.println("VCC-GND YD-ESP32-S3 RGB LED Test");
  Serial.println("===========================================");
  Serial.println("Board: VCC-GND Studio YD-ESP32-S3");
  Serial.println("RGB LED: GPIO 48");
  Serial.println("");
  Serial.println("**IMPORTANT CHECK**");
  Serial.println("Did you solder the RGB jumper closed?");
  Serial.println("Look for 'RGB' jumper near the LED on PCB");
  Serial.println("===========================================\n");

  delay(2000);

  Serial.println("Starting RGB LED test...");
  Serial.println("Watch the RGB LED near the USB connector!\n");
}

void loop() {
  // Cycle through colors continuously

  // RED (Full brightness)
  Serial.println("RED (Full brightness)");
  neopixelWrite(RGB_LED_PIN, 255, 0, 0);
  delay(1000);

  // GREEN (Full brightness)
  Serial.println("GREEN (Full brightness)");
  neopixelWrite(RGB_LED_PIN, 0, 255, 0);
  delay(1000);

  // BLUE (Full brightness)
  Serial.println("BLUE (Full brightness)");
  neopixelWrite(RGB_LED_PIN, 0, 0, 255);
  delay(1000);

  // YELLOW (Red + Green)
  Serial.println("YELLOW (Red + Green)");
  neopixelWrite(RGB_LED_PIN, 255, 255, 0);
  delay(1000);

  // CYAN (Green + Blue)
  Serial.println("CYAN (Green + Blue)");
  neopixelWrite(RGB_LED_PIN, 0, 255, 255);
  delay(1000);

  // MAGENTA (Red + Blue)
  Serial.println("MAGENTA (Red + Blue)");
  neopixelWrite(RGB_LED_PIN, 255, 0, 255);
  delay(1000);

  // WHITE (All colors)
  Serial.println("WHITE (All colors)");
  neopixelWrite(RGB_LED_PIN, 255, 255, 255);
  delay(1000);

  // OFF
  Serial.println("OFF");
  neopixelWrite(RGB_LED_PIN, 0, 0, 0);
  delay(1000);

  Serial.println("\n--- If you see the LED: SUCCESS! ---");
  Serial.println("--- If NOT: Check RGB jumper is soldered ---\n");
  delay(1000);
}
