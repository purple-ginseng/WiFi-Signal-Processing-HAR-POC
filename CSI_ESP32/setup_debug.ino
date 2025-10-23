/*
  ESP32-S3 CSI Sniffer - DIAGNOSTIC VERSION
  -----------------------------------------
  This version prints detailed debug info to help diagnose
  why CSI callback isn't firing frequently
*/

#include <WiFi.h>
#include "esp_wifi.h"
#include "esp_system.h"

// ===================== USER CONFIG =====================
static const int FIXED_CHANNEL = 6;
// =======================================================

// Counters
static volatile uint32_t csi_callback_count = 0;
static volatile uint32_t promiscuous_packet_count = 0;
static uint32_t last_stats_time = 0;

// =======================================================
// PROMISCUOUS MODE CALLBACK - counts ALL packets
// =======================================================
void wifi_promiscuous_cb(void* buf, wifi_promiscuous_pkt_type_t type) {
  promiscuous_packet_count++;
}

// =======================================================
// CSI CALLBACK - counts only CSI-enabled packets
// =======================================================
void wifi_csi_cb(void *ctx, wifi_csi_info_t *data) {
  if (!data || !data->buf || data->len == 0) return;
  csi_callback_count++;

  // Print first few packets for debugging
  if (csi_callback_count <= 5) {
    Serial.printf("CSI #%lu: len=%u rssi=%d\n",
                  csi_callback_count, data->len, data->rx_ctrl.rssi);
  }
}

// =======================================================
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n🔍 ESP32-S3 CSI Diagnostic Mode");
  Serial.println("================================\n");

  // Initialize Wi-Fi
  WiFi.mode(WIFI_MODE_NULL);
  esp_wifi_stop();
  esp_wifi_deinit();

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  esp_wifi_init(&cfg);
  esp_wifi_set_mode(WIFI_MODE_STA);
  esp_wifi_start();
  esp_wifi_set_ps(WIFI_PS_NONE);

  // Enable promiscuous mode
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous_rx_cb(&wifi_promiscuous_cb);

  // Configure CSI with ALL packet types enabled
  wifi_csi_config_t csi_config = {
    .lltf_en           = true,    // Legacy (802.11a/g)
    .htltf_en          = true,    // HT (802.11n)
    .stbc_htltf2_en    = true,    // STBC
    .ltf_merge_en      = true,    // Merge LTF
    .channel_filter_en = true,    // ⚠️ ENABLED for testing
    .manu_scale        = false,
    .shift             = false
  };

  esp_wifi_set_csi_config(&csi_config);
  esp_wifi_set_csi_rx_cb(&wifi_csi_cb, nullptr);
  esp_wifi_set_csi(true);

  Serial.printf("✅ Listening on channel %d\n", FIXED_CHANNEL);
  Serial.printf("✅ PSRAM: %s\n", psramFound() ? "Available" : "NOT FOUND");
  Serial.println("\nMonitoring both promiscuous packets and CSI callbacks...");
  Serial.println("Generate WiFi traffic (ping, download, etc.) and watch counts:\n");

  last_stats_time = millis();
}

// =======================================================
void loop() {
  uint32_t now = millis();

  // Print stats every 2 seconds
  if (now - last_stats_time >= 2000) {
    float elapsed = (now - last_stats_time) / 1000.0;
    float promiscuous_rate = promiscuous_packet_count / elapsed;
    float csi_rate = csi_callback_count / elapsed;
    float csi_percentage = (promiscuous_packet_count > 0) ?
                           (100.0 * csi_callback_count / promiscuous_packet_count) : 0;

    Serial.println("─────────────────────────────────────────────────────");
    Serial.printf("Total Promiscuous Packets: %6lu (%.1f/s)\n",
                  promiscuous_packet_count, promiscuous_rate);
    Serial.printf("CSI Callbacks Triggered:   %6lu (%.1f/s)\n",
                  csi_callback_count, csi_rate);
    Serial.printf("CSI Capture Rate:          %.1f%%\n\n", csi_percentage);

    if (promiscuous_packet_count == 0) {
      Serial.println("⚠️  NO PACKETS DETECTED!");
      Serial.println("   - Check if channel 6 has any WiFi traffic");
      Serial.println("   - Try generating traffic (ping, download, etc.)");
    } else if (csi_callback_count == 0) {
      Serial.println("⚠️  PACKETS DETECTED BUT NO CSI!");
      Serial.println("   - Packets visible but CSI callback not firing");
      Serial.println("   - This is the root cause of your low rate issue");
    } else if (csi_percentage < 10) {
      Serial.println("⚠️  LOW CSI CAPTURE RATE");
      Serial.printf("   - Only %.1f%% of packets generate CSI\n", csi_percentage);
    } else {
      Serial.println("✅ CSI capture working!");
    }

    // Reset counters
    promiscuous_packet_count = 0;
    csi_callback_count = 0;
    last_stats_time = now;
  }

  delay(100);
}
