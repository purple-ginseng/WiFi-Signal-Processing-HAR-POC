#include <WiFi.h>
#include "esp_wifi.h"

// =================== CONFIGURATION PARAMETERS ===================
const bool CSI_LLT_ENABLE       = true;    // Enable legacy long training field (LLTF)
const bool CSI_LTF_MERGE        = true;    // Merge HT-LTFs
const bool CSI_CHANNEL_FILTER   = false;   // Apply CSI channel filter
const bool CSI_MANUAL_SCALE     = false;   // Use manual scaling
const bool CSI_SHIFT_ENABLE     = false;   // Enable shift for dynamic range
const bool CSI_ENABLED          = true;    // Master toggle for CSI feature

// Channel scanning parameters
const int CHANNELS[] = {1, 6, 11};          // WiFi channels to scan
const int CHANNEL_COUNT = 3;
const int CHANNEL_DWELL_TIME = 1000;        // Time to stay on each channel (ms)
// ================================================================

unsigned long lastChannelSwitch = 0;
int currentChannelIndex = 0;
volatile int csiPacketCount = 0;

void IRAM_ATTR wifi_csi_cb(void *ctx, wifi_csi_info_t *data) {
  if (!data || !data->buf || data->len == 0) return;

  csiPacketCount++;

  // Optimized single buffer write - build string once
  char buffer[512];
  int pos = 0;

  pos += sprintf(buffer + pos, "CSI:CH=%d,RSSI=%d,RATE=%d,SIG_MODE=%d,MCS=%d,CWB=%d,SMOOTHING=%d,NOT_SOUNDING=%d,AGGREGATION=%d,STBC=%d,FEC_CODING=%d,SGI=%d,NOISE_FLOOR=%d,AMPDU_CNT=%d,CHANNEL=%d,SECONDARY_CHANNEL=%d,TIMESTAMP=%u,ANT=%d,SIG_LEN=%d,RX_STATE=%d,LEN=%d,FIRST_WORD_INVALID=%d,DATA=",
    CHANNELS[currentChannelIndex],
    data->rx_ctrl.rssi,
    data->rx_ctrl.rate,
    data->rx_ctrl.sig_mode,
    data->rx_ctrl.mcs,
    data->rx_ctrl.cwb,
    data->rx_ctrl.smoothing,
    data->rx_ctrl.not_sounding,
    data->rx_ctrl.aggregation,
    data->rx_ctrl.stbc,
    data->rx_ctrl.fec_coding,
    data->rx_ctrl.sgi,
    data->rx_ctrl.noise_floor,
    data->rx_ctrl.ampdu_cnt,
    data->rx_ctrl.channel,
    data->rx_ctrl.secondary_channel,
    data->rx_ctrl.timestamp,
    data->rx_ctrl.ant,
    data->rx_ctrl.sig_len,
    data->rx_ctrl.rx_state,
    data->len,
    data->first_word_invalid
  );

  Serial.print(buffer);

  // Output CSI data directly
  for (int i = 0; i < data->len; i++) {
    Serial.print(data->buf[i]);
    if (i < data->len - 1) Serial.print(',');
  }
  Serial.println();
}

void setup() {
  Serial.begin(921600);  // Increased baud rate for faster data transfer
  delay(500);  // Reduced delay

  Serial.println("=== ESP32 CSI Continuous Capture ===");

  WiFi.mode(WIFI_MODE_STA);
  WiFi.disconnect();
  esp_wifi_set_promiscuous(true);

  if (CSI_ENABLED) {
    wifi_csi_config_t csi_config = {
      .lltf_en         = CSI_LLT_ENABLE,
      .ltf_merge_en    = CSI_LTF_MERGE,
      .channel_filter_en = CSI_CHANNEL_FILTER,
      .manu_scale      = CSI_MANUAL_SCALE,
      .shift           = CSI_SHIFT_ENABLE
    };

    esp_wifi_set_csi_config(&csi_config);
    esp_wifi_set_csi_rx_cb(wifi_csi_cb, NULL);
    esp_wifi_set_csi(true);

    Serial.println("CSI Ready!");
  }

  esp_wifi_set_channel(CHANNELS[currentChannelIndex], WIFI_SECOND_CHAN_NONE);
  Serial.print("CH: "); Serial.println(CHANNELS[currentChannelIndex]);
  lastChannelSwitch = millis();
}

void loop() {
  unsigned long currentTime = millis();

  // Channel hopping
  if (currentTime - lastChannelSwitch >= CHANNEL_DWELL_TIME) {
    currentChannelIndex = (currentChannelIndex + 1) % CHANNEL_COUNT;
    esp_wifi_set_channel(CHANNELS[currentChannelIndex], WIFI_SECOND_CHAN_NONE);
    lastChannelSwitch = currentTime;

    Serial.print("CH_SWITCH:"); Serial.print(CHANNELS[currentChannelIndex]);
    Serial.print(",PKT="); Serial.println(csiPacketCount);
    csiPacketCount = 0;
  }

  // Removed status updates and delay for maximum speed
  // Feed watchdog without blocking
  yield();
}
