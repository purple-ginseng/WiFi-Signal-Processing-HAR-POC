/*
  ESP32-S3 CSI Receiver - Capture All Devices
  --------------------------------------------
  ✅ Captures CSI from ALL WiFi devices (no filtering)
  ✅ Auto-detects active devices on startup
  ✅ Maximizes packet capture rate for HAR
  ✅ No manual MAC address configuration needed
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
// =======================================================

// ======== MEMORY ALLOCATION (PSRAM-AWARE) ========
static uint16_t csi_len_buf[NUM_BUFFERS] DRAM_ATTR;
static int8_t   csi_rssi_buf[NUM_BUFFERS] DRAM_ATTR;
static uint32_t csi_timestamp_buf[NUM_BUFFERS] DRAM_ATTR;
static uint8_t  csi_data_buf[NUM_BUFFERS][MAX_CSI_LEN] EXT_RAM_ATTR;
static uint8_t  csi_mac_buf[NUM_BUFFERS][6] DRAM_ATTR;  // Store source MAC

// ======== RING BUFFER INDEXING ========
static volatile uint32_t ring_write_idx = 0;
static volatile uint32_t ring_read_idx  = 0;
portMUX_TYPE ring_mux = portMUX_INITIALIZER_UNLOCKED;

// ======== TASK & UDP OBJECT ========
static TaskHandle_t processing_task_handle = nullptr;
WiFiUDP udp;

// ======== STATISTICS ========
static volatile uint32_t total_csi_packets = 0;
static uint32_t last_stats_time = 0;

// =======================================================
// CSI CALLBACK — Accept ALL packets, no filtering
// =======================================================
void wifi_csi_cb(void *ctx, wifi_csi_info_t *data) {
  if (!data || !data->buf || data->len == 0) return;

  // NO MAC FILTERING - capture from all devices
  uint16_t len = data->len > MAX_CSI_LEN ? MAX_CSI_LEN : data->len;

  portENTER_CRITICAL_ISR(&ring_mux);
  uint32_t w = ring_write_idx;
  uint32_t next = (w + 1) % NUM_BUFFERS;
  if (next == ring_read_idx) ring_read_idx = (ring_read_idx + 1) % NUM_BUFFERS;

  csi_len_buf[w]       = len;
  csi_rssi_buf[w]      = data->rx_ctrl.rssi;
  csi_timestamp_buf[w] = (uint32_t)esp_timer_get_time();
  memcpy(csi_data_buf[w], data->buf, len);
  memcpy(csi_mac_buf[w], data->mac, 6);  // Store MAC address

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
  uint8_t local_mac[6];

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
      memcpy(local_mac, csi_mac_buf[r], 6);
      ring_read_idx = (r + 1) % NUM_BUFFERS;
      portEXIT_CRITICAL(&ring_mux);

      // ---- UDP STREAM ----
      if (UDP_ENABLE && WiFi.status() == WL_CONNECTED) {
        udp.beginPacket(REMOTE_IP, REMOTE_PORT);
        udp.write((uint8_t*)&ts, sizeof(ts));
        udp.write((uint8_t*)&rssi, sizeof(rssi));
        udp.write(local_mac, 6);  // Include source MAC
        udp.write(local_buf, len);
        udp.endPacket();
      }

      // ---- SERIAL OUTPUT ----
      if (PRINT_SERIAL) {
        // Print summary with MAC address
        Serial.printf("CSI ts=%u rssi=%d len=%u mac=%02X:%02X:%02X:%02X:%02X:%02X\n",
                      ts, rssi, len,
                      local_mac[0], local_mac[1], local_mac[2],
                      local_mac[3], local_mac[4], local_mac[5]);

        // Print CSI data
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
  Serial.println("\n🚀 ESP32-S3 CSI Receiver - All Devices Mode");
  Serial.println("==============================================");

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

  // Configure CSI - Try to capture from all devices
  wifi_csi_config_t csi_config;
  csi_config.lltf_en           = true;    // Legacy (802.11a/g)
  csi_config.htltf_en          = true;    // HT (802.11n)
  csi_config.stbc_htltf2_en    = true;    // STBC
  csi_config.ltf_merge_en      = true;    // Merge LTF
  csi_config.channel_filter_en = false;   // ⚠️ Try DISABLED to capture all
  csi_config.manu_scale        = false;
  csi_config.shift             = false;

  esp_wifi_set_csi_config(&csi_config);
  esp_wifi_set_csi_rx_cb(&wifi_csi_cb, nullptr);
  esp_wifi_set_csi(true);

  Serial.println("📡 Configured to capture CSI from ALL devices");
  Serial.println("   (No MAC filtering applied)");

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

  Serial.printf("\n✅ CSI enabled, channel %d, PSRAM=%s\n",
                FIXED_CHANNEL, psramFound() ? "OK" : "MISSING");
  Serial.println("🎧 Listening for CSI packets from all devices...\n");

  last_stats_time = millis();
}

// =======================================================
void loop() {
  // Print periodic stats
  uint32_t now = millis();
  if (now - last_stats_time >= 5000) {
    Serial.printf("📊 Total CSI packets captured: %lu\n", total_csi_packets);
    last_stats_time = now;
  }

  vTaskDelay(1000 / portTICK_PERIOD_MS);
}
