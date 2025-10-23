/*
  ESP32 ESP-NOW Transmitter for CSI-based HAR
  -------------------------------------------
  Purpose: Generate continuous WiFi packets for CSI capture
  Target: Second ESP32 acts as packet sniffer/CSI receiver

  This transmitter sends ESP-NOW broadcast packets at a fixed rate
  (default: 50 Hz) to provide consistent CSI sampling for human
  activity recognition.
*/

#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

// ===================== USER CONFIG =====================
static const int TX_RATE_HZ = 50;              // Packets per second (20-100 Hz typical for HAR)
static const int FIXED_CHANNEL = 6;            // Must match receiver channel
static const int PACKET_SIZE = 200;            // Payload size in bytes (adjust as needed)

// Optional: Set a specific MAC address to transmit to
// Use broadcast FF:FF:FF:FF:FF:FF to let any receiver capture it
static uint8_t receiver_mac[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
// =======================================================

// Packet counter and timing
static uint32_t packet_count = 0;
static uint32_t last_stats_time = 0;
static uint32_t packets_sent_in_window = 0;
static uint32_t failed_sends = 0;

// Payload buffer
static uint8_t tx_buffer[PACKET_SIZE];

// ESP-NOW peer info
esp_now_peer_info_t peer_info;

// =======================================================
// ESP-NOW Send Callback
// =======================================================
void on_data_sent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  if (status != ESP_NOW_SEND_SUCCESS) {
    failed_sends++;
  }
}

// =======================================================
// SETUP
// =======================================================
void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("🚀 ESP32 ESP-NOW Transmitter for CSI HAR");

  // Set device as WiFi Station
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  // Set WiFi channel
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous(false);

  Serial.printf("MAC Address: %s\n", WiFi.macAddress().c_str());
  Serial.printf("Channel: %d\n", FIXED_CHANNEL);
  Serial.printf("Target Rate: %d Hz\n", TX_RATE_HZ);

  // Initialize ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("❌ ESP-NOW init failed");
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
    Serial.println("❌ Failed to add peer");
    while (true) delay(1000);
  }

  Serial.println("✅ ESP-NOW initialized");
  Serial.printf("Transmitting to: %02X:%02X:%02X:%02X:%02X:%02X\n",
                receiver_mac[0], receiver_mac[1], receiver_mac[2],
                receiver_mac[3], receiver_mac[4], receiver_mac[5]);
  Serial.println("Starting transmission...\n");

  // Initialize packet payload (can be any data pattern)
  for (int i = 0; i < PACKET_SIZE; i++) {
    tx_buffer[i] = i % 256;
  }

  last_stats_time = millis();
}

// =======================================================
// LOOP - Send packets at fixed rate
// =======================================================
void loop() {
  static uint32_t next_tx_time = 0;
  uint32_t now = millis();

  // Send packet at fixed interval
  if (now >= next_tx_time) {
    // Update packet counter in payload (optional)
    memcpy(tx_buffer, &packet_count, sizeof(packet_count));

    // Send ESP-NOW packet
    esp_err_t result = esp_now_send(receiver_mac, tx_buffer, PACKET_SIZE);

    if (result == ESP_OK) {
      packet_count++;
      packets_sent_in_window++;
    } else {
      failed_sends++;
    }

    // Schedule next transmission
    next_tx_time = now + (1000 / TX_RATE_HZ);
  }

  // Print statistics every 2 seconds
  if (now - last_stats_time >= 2000) {
    float actual_rate = packets_sent_in_window / 2.0;
    float success_rate = 100.0 * (packets_sent_in_window) / (packets_sent_in_window + failed_sends);

    Serial.printf("Packets: %6lu | Rate: %5.1f Hz | Success: %5.1f%% | Total: %lu\n",
                  packet_count, actual_rate, success_rate, packet_count);

    packets_sent_in_window = 0;
    failed_sends = 0;
    last_stats_time = now;
  }

  // Small delay to prevent watchdog issues
  delayMicroseconds(100);
}
