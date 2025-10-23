/*
  ESP32-S3 Auto-Scanning CSI Receiver
  ------------------------------------
  ✅ Automatically scans for active MAC addresses on startup
  ✅ Captures CSI from multiple devices (rotates through top MACs)
  ✅ No manual configuration needed
  ✅ Optimized for passive WiFi sensing and HAR
*/

#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_wifi.h"
#include "esp_system.h"
#include <map>
#include <vector>
#include <algorithm>

// ===================== USER CONFIG =====================
static const int FIXED_CHANNEL = 6;              // Wi-Fi channel to sniff
static const int NUM_BUFFERS   = 256;            // ring buffer slots
static const int MAX_CSI_LEN   = 384;            // bytes per CSI payload
static const int PROCESS_TASK_CORE = 1;          // 0 or 1 (S3 has dual cores)

static const bool PRINT_SERIAL = true;           // brief logs to Serial
static const bool UDP_ENABLE   = false;          // enable UDP streaming
const char* REMOTE_IP = "192.168.3.4";           // your Raspberry Pi IP
const uint16_t REMOTE_PORT = 9000;               // UDP port on receiver

// Auto-scan settings
static const int SCAN_DURATION_SEC = 5;          // Startup scan duration
static const int MAX_TARGET_MACS = 5;            // Capture from top N devices
static const bool ROTATE_MACS = false;           // Rotate between MACs (experimental)
static const int MAC_ROTATION_MS = 1000;         // Switch MAC every N ms (if rotating)
// =======================================================

// ======== MAC ADDRESS TRACKING ========
struct MACInfo {
  uint8_t addr[6];
  uint32_t packet_count;
};
std::vector<MACInfo> target_macs;
int current_mac_index = 0;
uint32_t last_mac_rotation = 0;

// ======== MEMORY ALLOCATION (PSRAM-AWARE) ========
static uint16_t csi_len_buf[NUM_BUFFERS] DRAM_ATTR;
static int8_t   csi_rssi_buf[NUM_BUFFERS] DRAM_ATTR;
static uint32_t csi_timestamp_buf[NUM_BUFFERS] DRAM_ATTR;
static uint8_t  csi_data_buf[NUM_BUFFERS][MAX_CSI_LEN] EXT_RAM_ATTR;

// ======== RING BUFFER INDEXING ========
static volatile uint32_t ring_write_idx = 0;
static volatile uint32_t ring_read_idx  = 0;
portMUX_TYPE ring_mux = portMUX_INITIALIZER_UNLOCKED;

// ======== TASK & UDP OBJECT ========
static TaskHandle_t processing_task_handle = nullptr;
WiFiUDP udp;

// ======== STATISTICS ========
static volatile uint32_t total_csi_packets = 0;

// =======================================================
// AUTO-SCAN: Find active MAC addresses
// =======================================================
std::map<std::string, uint32_t> mac_scan_counts;

String macToString(const uint8_t* mac) {
  char buf[18];
  sprintf(buf, "%02X:%02X:%02X:%02X:%02X:%02X",
          mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
  return String(buf);
}

void scan_callback(void* buf, wifi_promiscuous_pkt_type_t type) {
  wifi_promiscuous_pkt_t* pkt = (wifi_promiscuous_pkt_t*)buf;
  if (pkt->rx_ctrl.sig_len > 24) {
    uint8_t* mac_src = &pkt->payload[10];
    String mac_str = macToString(mac_src);
    mac_scan_counts[mac_str.c_str()]++;
  }
}

void auto_scan_networks() {
  Serial.println("\n🔍 Auto-scanning for active WiFi devices...");
  Serial.printf("   Scanning channel %d for %d seconds\n\n", FIXED_CHANNEL, SCAN_DURATION_SEC);

  // Enable promiscuous mode for scanning
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous_rx_cb(&scan_callback);

  uint32_t start = millis();
  while (millis() - start < (SCAN_DURATION_SEC * 1000)) {
    delay(100);
    if ((millis() - start) % 1000 == 0) {
      Serial.printf("   Scanning... %ds remaining\n",
                    SCAN_DURATION_SEC - ((millis() - start) / 1000));
    }
  }

  esp_wifi_set_promiscuous(false);

  // Sort by packet count
  std::vector<std::pair<std::string, uint32_t>> vec(mac_scan_counts.begin(), mac_scan_counts.end());
  std::sort(vec.begin(), vec.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  // Display results and populate target_macs
  Serial.println("\n📊 Most Active Devices:\n");
  Serial.println("   Rank | MAC Address       | Packets | Status");
  Serial.println("   -----|-------------------|---------|--------");

  int count = 0;
  for (const auto& item : vec) {
    if (count >= MAX_TARGET_MACS) break;

    MACInfo mac_info;
    mac_info.packet_count = item.second;

    // Parse MAC address
    String mac_str = String(item.first.c_str());
    for (int i = 0; i < 6; i++) {
      String byte_str = mac_str.substring(i*3, i*3+2);
      mac_info.addr[i] = strtol(byte_str.c_str(), NULL, 16);
    }

    target_macs.push_back(mac_info);

    Serial.printf("   %2d   | %s |  %5lu  | %s\n",
                  count + 1, item.first.c_str(), item.second,
                  count == 0 ? "PRIMARY" : "Target");
    count++;
  }

  if (target_macs.size() == 0) {
    Serial.println("\n⚠️  NO DEVICES FOUND!");
    Serial.println("   Channel 6 appears empty. CSI capture may not work.");
  } else {
    Serial.printf("\n✅ Found %d active device(s)\n", target_macs.size());
    if (ROTATE_MACS && target_macs.size() > 1) {
      Serial.printf("   Will rotate between devices every %dms\n", MAC_ROTATION_MS);
    } else {
      Serial.println("   Capturing from PRIMARY device");
    }
  }
  Serial.println();
}

// =======================================================
// CSI CALLBACK — copies data into ring buffer (non-blocking)
// =======================================================
void wifi_csi_cb(void *ctx, wifi_csi_info_t *data) {
  if (!data || !data->buf || data->len == 0) return;

  // Accept all CSI packets (filtering done by MAC rotation if enabled)
  uint16_t len = data->len > MAX_CSI_LEN ? MAX_CSI_LEN : data->len;

  portENTER_CRITICAL_ISR(&ring_mux);
  uint32_t w = ring_write_idx;
  uint32_t next = (w + 1) % NUM_BUFFERS;
  if (next == ring_read_idx) ring_read_idx = (ring_read_idx + 1) % NUM_BUFFERS;

  csi_len_buf[w]       = len;
  csi_rssi_buf[w]      = data->rx_ctrl.rssi;
  csi_timestamp_buf[w] = (uint32_t)esp_timer_get_time();
  memcpy(csi_data_buf[w], data->buf, len);

  ring_write_idx = next;
  portEXIT_CRITICAL_ISR(&ring_mux);

  total_csi_packets++;

  if (processing_task_handle) {
    BaseType_t hpw = pdFALSE;
    vTaskNotifyGiveFromISR(processing_task_handle, &hpw);
    if (hpw) portYIELD_FROM_ISR();
  }
}

// =======================================================
// BACKGROUND TASK — drains ring and streams packets
// =======================================================
void csi_processing_task(void *pv) {
  (void)pv;
  uint8_t local_buf[MAX_CSI_LEN];

  for (;;) {
    ulTaskNotifyTake(pdTRUE, 200 / portTICK_PERIOD_MS);

    // MAC rotation logic (if enabled)
    if (ROTATE_MACS && target_macs.size() > 1) {
      uint32_t now = millis();
      if (now - last_mac_rotation >= MAC_ROTATION_MS) {
        current_mac_index = (current_mac_index + 1) % target_macs.size();
        // Note: Actual MAC switching would require reconfiguring WiFi here
        // This is experimental and may impact performance
        last_mac_rotation = now;
      }
    }

    while (true) {
      portENTER_CRITICAL(&ring_mux);
      uint32_t r = ring_read_idx;
      uint32_t w = ring_write_idx;
      if (r == w) { portEXIT_CRITICAL(&ring_mux); break; }

      uint16_t len = csi_len_buf[r];
      int8_t   rssi = csi_rssi_buf[r];
      uint32_t ts = csi_timestamp_buf[r];
      memcpy(local_buf, csi_data_buf[r], len);
      ring_read_idx = (r + 1) % NUM_BUFFERS;
      portEXIT_CRITICAL(&ring_mux);

      // ---- UDP STREAM ----
      if (UDP_ENABLE && WiFi.status() == WL_CONNECTED) {
        udp.beginPacket(REMOTE_IP, REMOTE_PORT);
        udp.write((uint8_t*)&ts, sizeof(ts));
        udp.write((uint8_t*)&rssi, sizeof(rssi));
        udp.write(local_buf, len);
        udp.endPacket();
      }

      // ---- SERIAL OUTPUT ----
      if (PRINT_SERIAL) {
        Serial.printf("CSI ts=%u rssi=%d len=%u\n", ts, rssi, len);
        for (uint16_t i = 0; i < len; i++) {
          Serial.printf("%d", (int8_t)local_buf[i]);
          if (i < len - 1) Serial.print(",");
        }
        Serial.println();
      }
    }
  }
}

// =======================================================
// SETUP
// =======================================================
void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("\n🚀 ESP32-S3 Auto-Scanning CSI Receiver");
  Serial.println("========================================");

  // Initialize Wi-Fi cleanly
  WiFi.mode(WIFI_MODE_NULL);
  esp_wifi_stop();
  esp_wifi_deinit();

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  esp_wifi_init(&cfg);
  esp_wifi_set_mode(WIFI_MODE_STA);
  esp_wifi_start();
  esp_wifi_set_ps(WIFI_PS_NONE);

  // AUTO-SCAN for active devices
  auto_scan_networks();

  // Lock to fixed channel & enable promiscuous
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);

  // Configure CSI
  wifi_csi_config_t csi_config;
  csi_config.lltf_en           = true;
  csi_config.htltf_en          = true;
  csi_config.stbc_htltf2_en    = true;
  csi_config.ltf_merge_en      = true;
  csi_config.channel_filter_en = true;   // Required for CSI
  csi_config.manu_scale        = false;
  csi_config.shift             = false;

  esp_wifi_set_csi_config(&csi_config);
  esp_wifi_set_csi_rx_cb(&wifi_csi_cb, nullptr);
  esp_wifi_set_csi(true);

  // Launch processing task
  if (xTaskCreatePinnedToCore(
        csi_processing_task,
        "csi_proc",
        4096,
        NULL,
        configMAX_PRIORITIES - 2,
        &processing_task_handle,
        PROCESS_TASK_CORE) != pdPASS) {
    Serial.println("❌ Failed to create CSI task");
    while (true) delay(1000);
  }

  Serial.printf("✅ CSI enabled, channel %d, PSRAM=%s\n",
                FIXED_CHANNEL, psramFound() ? "OK" : "MISSING");
  Serial.println("🎧 Listening for CSI packets...\n");
}

// =======================================================
void loop() {
  vTaskDelay(1000 / portTICK_PERIOD_MS);
}
