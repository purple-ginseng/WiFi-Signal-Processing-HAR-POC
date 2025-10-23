/*
  ESP32-S3 Continuous CSI Sniffer + UDP Streamer
  ----------------------------------------------
  ✅ Continuous non-blocking CSI capture
  ✅ Large ring buffers in PSRAM (no DRAM overflow)
  ✅ Streams each CSI packet to UDP receiver (e.g. Raspberry Pi)
  ✅ Optional Serial summary output
  ✅ Optimized for ESP-NOW transmitter packets (high-rate HAR)
*/

#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_wifi.h"
#include "esp_system.h"

// ===================== USER CONFIG =====================
static const int FIXED_CHANNEL = 6;              // Wi-Fi channel to sniff
static const int NUM_BUFFERS   = 256;            // ring buffer slots
static const int MAX_CSI_LEN   = 384;            // bytes per CSI payload
static const int PROCESS_TASK_CORE = 1;          // 0 or 1 (S3 has dual cores)

static const bool PRINT_SERIAL = true;           // brief logs to Serial
static const bool UDP_ENABLE   = false;          // enable UDP streaming
const char* REMOTE_IP = "192.168.3.4";           // your Raspberry Pi IP
const uint16_t REMOTE_PORT = 9000;               // UDP port on receiver

// MAC address filter - Set to most active device from scanner
// Scanner found: E6:FA:C4:1B:8C:9B (45 packets in 10s scan)
static uint8_t filter_mac[6] = {0xE6, 0xFA, 0xC4, 0x1B, 0x8C, 0x9B};
// =======================================================

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

// =======================================================
// CSI CALLBACK — copies data into ring buffer (non-blocking)
// =======================================================
void wifi_csi_cb(void *ctx, wifi_csi_info_t *data) {
  if (!data || !data->buf || data->len == 0) return;

  // Optional: Filter by MAC address if filter_mac is set
  // Check if we should filter by MAC
  bool filter_enabled = false;
  for (int i = 0; i < 6; i++) {
    if (filter_mac[i] != 0x00) {
      filter_enabled = true;
      break;
    }
  }

  // If filter is enabled, check if packet matches
  if (filter_enabled) {
    bool mac_match = true;
    for (int i = 0; i < 6; i++) {
      if (data->mac[i] != filter_mac[i]) {
        mac_match = false;
        break;
      }
    }
    if (!mac_match) return;  // Skip packets from other devices
  }

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
        // Print summary line
        Serial.printf("CSI ts=%u rssi=%d len=%u\n", ts, rssi, len);

        // Print raw CSI data as comma-separated I,Q pairs
        // CSI data is int8_t pairs (I, Q)
        for (uint16_t i = 0; i < len; i++) {
          Serial.printf("%d", (int8_t)local_buf[i]);
          if (i < len - 1) Serial.print(",");
        }
        Serial.println();  // End the CSI data line
      }
    }
  }
}

// =======================================================
// SETUP — Wi-Fi init + CSI enable + UDP connect
// =======================================================
void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("🚀 ESP32-S3 CSI Sniffer + UDP Streamer (PSRAM)");

  // Initialize Wi-Fi cleanly
  WiFi.mode(WIFI_MODE_NULL);
  esp_wifi_stop();
  esp_wifi_deinit();

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  esp_wifi_init(&cfg);
  esp_wifi_set_mode(WIFI_MODE_STA);
  esp_wifi_start();
  esp_wifi_set_ps(WIFI_PS_NONE);

  // Lock to fixed channel & enable promiscuous
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);

  // Configure CSI
  wifi_csi_config_t csi_config;
  csi_config.lltf_en           = true;   // Legacy packets
  csi_config.htltf_en          = true;   // HT (802.11n) packets
  csi_config.stbc_htltf2_en    = true;   // STBC packets
  csi_config.ltf_merge_en      = true;   // Merge LTF data
  csi_config.channel_filter_en = true;   // Enable MAC filtering (required!)
  csi_config.manu_scale        = false;
  csi_config.shift             = false;  // Get unshifted raw values

  esp_wifi_set_csi_config(&csi_config);
  esp_wifi_set_csi_rx_cb(&wifi_csi_cb, nullptr);
  esp_wifi_set_csi(true);

  // Display MAC filter configuration
  bool is_filter_enabled = false;
  for (int i = 0; i < 6; i++) {
    if (filter_mac[i] != 0x00) {
      is_filter_enabled = true;
      break;
    }
  }

  if (is_filter_enabled) {
    Serial.printf("🎯 Targeting MAC: %02X:%02X:%02X:%02X:%02X:%02X\n",
                  filter_mac[0], filter_mac[1], filter_mac[2],
                  filter_mac[3], filter_mac[4], filter_mac[5]);
    Serial.println("   CSI will capture packets from this device");
  } else {
    Serial.println("⚠️  WARNING: No MAC filter set!");
    Serial.println("   Run setup_scanner.ino to find active devices\n");
  }

  // UDP setup (requires STA mode connection if you want actual network)
  if (UDP_ENABLE) {
    Serial.printf("UDP target: %s:%u\n", REMOTE_IP, REMOTE_PORT);
    // Optional: connect to local Wi-Fi if you need network reachability
    // WiFi.begin("YourSSID","YourPassword");
    // while (WiFi.status() != WL_CONNECTED) { delay(200); Serial.print("."); }
  }

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
  Serial.println("Listening for Wi-Fi packets...");
}

// =======================================================
void loop() {
  vTaskDelay(1000 / portTICK_PERIOD_MS);
}
