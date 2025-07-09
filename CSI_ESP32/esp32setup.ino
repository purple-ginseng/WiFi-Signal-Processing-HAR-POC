#include <WiFi.h>
#include "esp_wifi.h"

// =================== CONFIGURATION PARAMETERS ===================
const bool CSI_LLT_ENABLE       = true;    // Enable legacy long training field (LLTF)
const bool CSI_LTF_MERGE        = true;    // Merge HT-LTFs
const bool CSI_CHANNEL_FILTER   = false;   // Apply CSI channel filter
const bool CSI_MANUAL_SCALE     = false;   // Use manual scaling
const bool CSI_SHIFT_ENABLE     = false;   // Enable shift for dynamic range
const bool CSI_ENABLED          = true;    // Master toggle for CSI feature
// ================================================================

void wifi_csi_cb(void *ctx, wifi_csi_info_t *data) {
  if (!data || !data->buf || data->len == 0) return;

  Serial.print("CSI:");
  for (int i = 0; i < data->len; i++) {
    Serial.print(data->buf[i]);
    if (i < data->len - 1) Serial.print(",");
  }
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  delay(1000);  // Give time for Serial to init

  WiFi.mode(WIFI_MODE_STA);
  WiFi.disconnect(); // Ensure not connected to AP
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
  }
}

void loop() {
  delay(5); // CSI comes via callback
}
