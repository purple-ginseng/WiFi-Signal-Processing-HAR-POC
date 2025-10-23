/*
  Simple Serial Test - Verify ESP32 is working
*/

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n\n==================================");
  Serial.println("ESP32-S3 Serial Test");
  Serial.println("==================================");
  Serial.println("If you can see this, serial is working!");
  Serial.println("Now upload one of the CSI sketches.");
  Serial.println("==================================\n");
}

void loop() {
  static int count = 0;
  Serial.printf("Loop count: %d\n", count++);
  delay(1000);
}
