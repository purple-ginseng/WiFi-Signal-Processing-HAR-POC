/*
  ESP32 CSI/RSSI Transmitter (TX)
  --------------------------------
  Purpose: Actively transmit WiFi packets for CSI-based activity recognition

  This transmitter sends continuous ESP-NOW packets at a fixed rate,
  allowing a receiver ESP32 to capture CSI and RSSI data.

  Configuration:
  - Baud Rate: 921600 (high-speed serial output)
  - TX Rate: 50 Hz (configurable)
  - Channel: 6 (must match receiver)

  Upload this to the TRANSMITTER ESP32
*/

#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

// =================== CONFIGURATION PARAMETERS ===================
// TRANSMISSION SETTINGS
const int TX_RATE_HZ            = 50;      // Packets per second (20-100 Hz typical)
const int FIXED_CHANNEL         = 6;       // WiFi channel (1-13, must match RX)
const int PACKET_SIZE           = 200;     // Payload size in bytes

// SERIAL PERFORMANCE
const int BAUD_RATE             = 921600;  // High-speed serial for data output
const int SERIAL_TX_BUFFER      = 2048;    // Large TX buffer

// LED INDICATOR (VCC-GND YD-ESP32-S3 RGB WS2812 LED)
const int LED_PIN               = 48;      // VCC-GND YD-ESP32-S3 RGB LED on GPIO 48
const int LED_BLINK_DURATION    = 50;      // LED on duration in ms
const int LED_BRIGHTNESS        = 50;      // LED brightness (0-255)
// NOTE: RGB jumper must be soldered for LED to work!

// TARGET MAC (broadcast to all receivers)
static uint8_t receiver_mac[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
// ================================================================

// Packet tracking
static uint32_t packet_count = 0;
static uint32_t packets_sent_in_window = 0;
static uint32_t failed_sends = 0;
static uint32_t last_stats_time = 0;
static uint32_t led_off_time = 0;

// Payload buffer
static uint8_t tx_buffer[PACKET_SIZE];

// ESP-NOW peer info
esp_now_peer_info_t peer_info;

// =================== ESP-NOW CALLBACK ===================
void on_data_sent(const wifi_tx_info_t *tx_info, esp_now_send_status_t status) {
  if (status != ESP_NOW_SEND_SUCCESS) {
    failed_sends++;
  }
}

// =================== SETUP ===================
void setup() {
  // Initialize high-speed serial
  Serial.begin(BAUD_RATE);
  Serial.setTxBufferSize(SERIAL_TX_BUFFER);
  delay(500);

  Serial.println("===========================================");
  Serial.println("ESP32 CSI/RSSI Transmitter (TX)");
  Serial.println("===========================================");

  // Initialize WS2812 RGB LED (turn off initially)
  neopixelWrite(LED_PIN, 0, 0, 0);  // RGB: Off

  // LED startup sequence (3 quick blinks in blue)
  for (int i = 0; i < 3; i++) {
    neopixelWrite(LED_PIN, 0, 0, LED_BRIGHTNESS);  // Blue
    delay(100);
    neopixelWrite(LED_PIN, 0, 0, 0);  // Off
    delay(100);
  }

  // Disable WiFi sleep for consistent transmission
  WiFi.setSleep(false);

  // Set device as WiFi Station
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  // Set WiFi channel
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous(false);

  Serial.printf("MAC Address: %s\n", WiFi.macAddress().c_str());
  Serial.printf("Channel: %d\n", FIXED_CHANNEL);
  Serial.printf("Target TX Rate: %d Hz\n", TX_RATE_HZ);
  Serial.printf("Baud Rate: %d\n", BAUD_RATE);

  // Initialize ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("[ERROR] ESP-NOW initialization failed");
    while (true) delay(1000);
  }

  // Register send callback
  esp_now_register_send_cb(on_data_sent);

  // Register peer (broadcast address)
  memcpy(peer_info.peer_addr, receiver_mac, 6);
  peer_info.channel = FIXED_CHANNEL;
  peer_info.encrypt = false;
  peer_info.ifidx = WIFI_IF_STA;

  if (esp_now_add_peer(&peer_info) != ESP_OK) {
    Serial.println("[ERROR] Failed to add peer");
    while (true) delay(1000);
  }

  Serial.println("[OK] ESP-NOW initialized successfully");
  Serial.printf("Transmitting to: %02X:%02X:%02X:%02X:%02X:%02X\n",
                receiver_mac[0], receiver_mac[1], receiver_mac[2],
                receiver_mac[3], receiver_mac[4], receiver_mac[5]);
  Serial.printf("LED Indicator: GPIO %d (WS2812 RGB LED - Blue)\n", LED_PIN);
  Serial.println("===========================================");
  Serial.println("Starting transmission...\n");

  // Initialize packet payload with sequence pattern
  for (int i = 0; i < PACKET_SIZE; i++) {
    tx_buffer[i] = i % 256;
  }

  last_stats_time = millis();
}

// =================== MAIN LOOP ===================
void loop() {
  static uint32_t next_tx_time = 0;
  uint32_t now = millis();

  // Transmit packet at fixed interval
  if (now >= next_tx_time) {
    // Embed packet counter in payload (first 4 bytes)
    memcpy(tx_buffer, &packet_count, sizeof(packet_count));

    // Add timestamp (next 4 bytes)
    memcpy(tx_buffer + 4, &now, sizeof(now));

    // Send ESP-NOW packet
    esp_err_t result = esp_now_send(receiver_mac, tx_buffer, PACKET_SIZE);

    if (result == ESP_OK) {
      packet_count++;
      packets_sent_in_window++;

      // Flash blue LED on successful transmission
      neopixelWrite(LED_PIN, 0, 0, LED_BRIGHTNESS);  // Blue
      led_off_time = now + LED_BLINK_DURATION;
    } else {
      failed_sends++;
    }

    // Schedule next transmission (maintain fixed rate)
    next_tx_time = now + (1000 / TX_RATE_HZ);
  }

  // Turn off LED after blink duration
  if (led_off_time > 0 && now >= led_off_time) {
    neopixelWrite(LED_PIN, 0, 0, 0);  // Off
    led_off_time = 0;
  }

  // Print statistics every 2 seconds
  if (now - last_stats_time >= 2000) {
    float actual_rate = packets_sent_in_window / 2.0;
    float success_rate = (packets_sent_in_window + failed_sends > 0)
                         ? 100.0 * packets_sent_in_window / (packets_sent_in_window + failed_sends)
                         : 0.0;

    Serial.printf("TX | Packets: %6lu | Rate: %5.1f Hz | Success: %5.1f%% | Failed: %lu\n",
                  packet_count, actual_rate, success_rate, failed_sends);

    packets_sent_in_window = 0;
    failed_sends = 0;
    last_stats_time = now;
  }

  // Small delay to prevent watchdog timeout
  delayMicroseconds(100);
}
