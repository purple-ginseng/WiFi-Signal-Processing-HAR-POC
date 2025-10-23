/*
  ESP32-S3 WiFi Scanner - Find Active MAC Addresses
  --------------------------------------------------
  This tool scans channel 6 and finds the most active
  MAC addresses, which you can then use as CSI filter targets
*/

#include <WiFi.h>
#include "esp_wifi.h"
#include <map>
#include <string>

#define SCAN_CHANNEL 6
#define SCAN_DURATION_SEC 10

std::map<std::string, uint32_t> mac_counts;

String macToString(const uint8_t* mac) {
  char buf[18];
  sprintf(buf, "%02X:%02X:%02X:%02X:%02X:%02X",
          mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
  return String(buf);
}

void promiscuous_rx_cb(void* buf, wifi_promiscuous_pkt_type_t type) {
  wifi_promiscuous_pkt_t* pkt = (wifi_promiscuous_pkt_t*)buf;

  // Extract source MAC address (bytes 10-15 in WiFi frame)
  if (pkt->rx_ctrl.sig_len > 24) {
    uint8_t* mac_src = &pkt->payload[10];
    String mac_str = macToString(mac_src);
    mac_counts[mac_str.c_str()]++;
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n📡 ESP32-S3 WiFi MAC Address Scanner");
  Serial.println("=====================================\n");

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(100);

  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(SCAN_CHANNEL, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous_rx_cb(&promiscuous_rx_cb);

  Serial.printf("Scanning channel %d for %d seconds...\n\n",
                SCAN_CHANNEL, SCAN_DURATION_SEC);

  uint32_t start = millis();
  while (millis() - start < (SCAN_DURATION_SEC * 1000)) {
    delay(100);
    if ((millis() - start) % 2000 == 0) {
      Serial.printf("Scanning... %ds remaining\n",
                    SCAN_DURATION_SEC - ((millis() - start) / 1000));
    }
  }

  esp_wifi_set_promiscuous(false);

  Serial.println("\n🎯 Most Active MAC Addresses:\n");
  Serial.println("Rank | MAC Address       | Packet Count");
  Serial.println("-----|-------------------|-------------");

  // Convert map to vector for sorting
  std::vector<std::pair<std::string, uint32_t>> vec(mac_counts.begin(), mac_counts.end());
  std::sort(vec.begin(), vec.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  int rank = 1;
  for (const auto& item : vec) {
    if (rank > 10) break;  // Show top 10
    Serial.printf(" %2d  | %s | %6lu\n", rank++, item.first.c_str(), item.second);
  }

  if (vec.size() > 0) {
    Serial.println("\n✅ COPY THIS MAC ADDRESS TO YOUR SETUP.INO:");
    Serial.println("   --------------------------------------------");
    Serial.printf("   static uint8_t filter_mac[6] = {");

    // Parse the top MAC address
    String top_mac = String(vec[0].first.c_str());
    for (int i = 0; i < 6; i++) {
      String byte_str = top_mac.substring(i*3, i*3+2);
      uint8_t byte_val = strtol(byte_str.c_str(), NULL, 16);
      Serial.printf("0x%02X", byte_val);
      if (i < 5) Serial.print(", ");
    }
    Serial.println("};");
    Serial.println("   --------------------------------------------\n");
  } else {
    Serial.println("\n⚠️  NO PACKETS DETECTED!");
    Serial.println("   - Channel 6 might be empty");
    Serial.println("   - Try scanning other channels (1, 6, 11)");
  }

  Serial.println("\nScan complete. Reset ESP32 to scan again.");
}

void loop() {
  delay(1000);
}
