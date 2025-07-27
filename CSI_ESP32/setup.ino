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
const int CSI_OUTPUT_RATE = 100;            // Max CSI outputs per second
// ================================================================

unsigned long lastChannelSwitch = 0;
unsigned long lastCSIOutput = 0;
int currentChannelIndex = 0;
volatile int csiPacketCount = 0;

void wifi_csi_cb(void *ctx, wifi_csi_info_t *data) {
  if (!data || !data->buf || data->len == 0) return;
  
  // Rate limiting to prevent overwhelming the serial output
  unsigned long currentTime = millis();
  if (currentTime - lastCSIOutput < (1000 / CSI_OUTPUT_RATE)) {
    return;
  }
  lastCSIOutput = currentTime;
  
  csiPacketCount++;
  
  // Enhanced CSI data output with metadata
  Serial.print("CSI:");
  Serial.print("CH="); Serial.print(CHANNELS[currentChannelIndex]); Serial.print(",");
  Serial.print("RSSI="); Serial.print(data->rx_ctrl.rssi); Serial.print(",");
  Serial.print("RATE="); Serial.print(data->rx_ctrl.rate); Serial.print(",");
  Serial.print("SIG_MODE="); Serial.print(data->rx_ctrl.sig_mode); Serial.print(",");
  Serial.print("MCS="); Serial.print(data->rx_ctrl.mcs); Serial.print(",");
  Serial.print("CWB="); Serial.print(data->rx_ctrl.cwb); Serial.print(",");
  Serial.print("SMOOTHING="); Serial.print(data->rx_ctrl.smoothing); Serial.print(",");
  Serial.print("NOT_SOUNDING="); Serial.print(data->rx_ctrl.not_sounding); Serial.print(",");
  Serial.print("AGGREGATION="); Serial.print(data->rx_ctrl.aggregation); Serial.print(",");
  Serial.print("STBC="); Serial.print(data->rx_ctrl.stbc); Serial.print(",");
  Serial.print("FEC_CODING="); Serial.print(data->rx_ctrl.fec_coding); Serial.print(",");
  Serial.print("SGI="); Serial.print(data->rx_ctrl.sgi); Serial.print(",");
  Serial.print("NOISE_FLOOR="); Serial.print(data->rx_ctrl.noise_floor); Serial.print(",");
  Serial.print("AMPDU_CNT="); Serial.print(data->rx_ctrl.ampdu_cnt); Serial.print(",");
  Serial.print("CHANNEL="); Serial.print(data->rx_ctrl.channel); Serial.print(",");
  Serial.print("SECONDARY_CHANNEL="); Serial.print(data->rx_ctrl.secondary_channel); Serial.print(",");
  Serial.print("TIMESTAMP="); Serial.print(data->rx_ctrl.timestamp); Serial.print(",");
  Serial.print("ANT="); Serial.print(data->rx_ctrl.ant); Serial.print(",");
  Serial.print("SIG_LEN="); Serial.print(data->rx_ctrl.sig_len); Serial.print(",");
  Serial.print("RX_STATE="); Serial.print(data->rx_ctrl.rx_state); Serial.print(",");
  Serial.print("LEN="); Serial.print(data->len); Serial.print(",");
  Serial.print("FIRST_WORD_INVALID="); Serial.print(data->first_word_invalid); Serial.print(",");
  Serial.print("DATA=");
  
  // Output CSI data as comma-separated values
  for (int i = 0; i < data->len; i++) {
    Serial.print(data->buf[i]);
    if (i < data->len - 1) Serial.print(",");
  }
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  delay(2000);  // Give time for Serial to init
  
  Serial.println("=== ESP32 CSI Continuous Capture ===");
  Serial.println("Initializing WiFi in promiscuous mode...");

  WiFi.mode(WIFI_MODE_STA);
  WiFi.disconnect(); // Ensure not connected to AP
  esp_wifi_set_promiscuous(true);

  if (CSI_ENABLED) {
    Serial.println("Configuring CSI...");
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
    
    Serial.println("CSI capture enabled!");
  }
  
  // Set initial channel
  esp_wifi_set_channel(CHANNELS[currentChannelIndex], WIFI_SECOND_CHAN_NONE);
  Serial.print("Starting on channel: "); Serial.println(CHANNELS[currentChannelIndex]);
  lastChannelSwitch = millis();
}

void loop() {
  unsigned long currentTime = millis();
  
  // Channel hopping for continuous capture across multiple channels
  if (currentTime - lastChannelSwitch >= CHANNEL_DWELL_TIME) {
    currentChannelIndex = (currentChannelIndex + 1) % CHANNEL_COUNT;
    esp_wifi_set_channel(CHANNELS[currentChannelIndex], WIFI_SECOND_CHAN_NONE);
    lastChannelSwitch = currentTime;
    
    Serial.print("CHANNEL_SWITCH:"); Serial.print(CHANNELS[currentChannelIndex]);
    Serial.print(",PACKETS="); Serial.println(csiPacketCount);
    csiPacketCount = 0; // Reset counter for new channel
  }
  
  // Status update every 10 seconds
  static unsigned long lastStatusUpdate = 0;
  if (currentTime - lastStatusUpdate >= 10000) {
    Serial.print("STATUS:UPTIME="); Serial.print(currentTime);
    Serial.print(",CH="); Serial.print(CHANNELS[currentChannelIndex]);
    Serial.print(",PACKETS="); Serial.println(csiPacketCount);
    lastStatusUpdate = currentTime;
  }
  
  delay(10); // Small delay to prevent watchdog issues
}
