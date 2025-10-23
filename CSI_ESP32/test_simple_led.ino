/*
  ESP32-S3 Simple LED Test
  Tests for regular single-color LED (not RGB)

  Tests common LED pins using basic digitalWrite
*/

// Common LED pins on ESP32 boards
int test_pins[] = {2, 5, 13, 15, 18, 19, 21, 22, 23, 38, 48};
int num_pins = 11;

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("\n===========================================");
  Serial.println("ESP32-S3 Simple LED Scanner");
  Serial.println("===========================================");
  Serial.println("Testing common LED pins with digitalWrite");
  Serial.println("Watch for ANY LED to blink!");
  Serial.println("===========================================\n");
}

void loop() {
  for (int i = 0; i < num_pins; i++) {
    int pin = test_pins[i];

    // Configure pin as output
    pinMode(pin, OUTPUT);

    Serial.printf("\nTesting GPIO %d: ", pin);

    // Blink 3 times
    for (int j = 0; j < 3; j++) {
      digitalWrite(pin, HIGH);
      Serial.print("ON ");
      delay(300);

      digitalWrite(pin, LOW);
      Serial.print("OFF ");
      delay(300);
    }

    Serial.println("(done)");

    // Reset pin
    pinMode(pin, INPUT);
    delay(500);
  }

  Serial.println("\n===========================================");
  Serial.println("SCAN COMPLETE!");
  Serial.println("Did you see ANY LED blink?");
  Serial.println("If YES: Note the GPIO number");
  Serial.println("If NO: Board may not have onboard LED");
  Serial.println("Restarting in 5 seconds...");
  Serial.println("===========================================\n");
  delay(5000);
}
