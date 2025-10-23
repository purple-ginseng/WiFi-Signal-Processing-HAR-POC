#include <WiFi.h>
#include "esp_wifi.h"

// =================== CONFIGURATION PARAMETERS ===================
const bool CSI_LLT_ENABLE       = true;    // Enable legacy long training field (LLTF)
const bool CSI_LTF_MERGE        = true;    // Merge HT-LTFs
const bool CSI_CHANNEL_FILTER   = false;   // Apply CSI channel filter
const bool CSI_MANUAL_SCALE     = false;   // Use manual scaling
const bool CSI_SHIFT_ENABLE     = true;    // Enable shift for dynamic range
const bool CSI_ENABLED          = true;    // Master toggle for CSI feature

// PERFORMANCE OPTIMIZATION
const int BAUD_RATE             = 921600;  // Increased from 115200 for faster transmission
const int SERIAL_TX_BUFFER      = 2048;   // Larger TX buffer to prevent blocking
// ================================================================

// Optimized callback with minimal processing overhead
void wifi_csi_cb(void *ctx, wifi_csi_info_t *data) {
  if (!data || !data->buf || data->len == 0) return;

  // Pre-allocate buffer to avoid multiple allocations
  static char buffer[512];
  int pos = 0;

  // Write header
  buffer[pos++] = 'C';
  buffer[pos++] = 'S';
  buffer[pos++] = 'I';
  buffer[pos++] = ':';

  // Optimized integer-to-string conversion
  for (int i = 0; i < data->len; i++) {
    // Convert signed byte to string
    int8_t val = (int8_t)data->buf[i];

    if (val < 0) {
      buffer[pos++] = '-';
      val = -val;
    }

    // Fast integer conversion
    if (val >= 100) {
      buffer[pos++] = '0' + (val / 100);
      buffer[pos++] = '0' + ((val / 10) % 10);
      buffer[pos++] = '0' + (val % 10);
    } else if (val >= 10) {
      buffer[pos++] = '0' + (val / 10);
      buffer[pos++] = '0' + (val % 10);
    } else {
      buffer[pos++] = '0' + val;
    }

    if (i < data->len - 1) buffer[pos++] = ',';
  }

  // Add RSSI data to the same line
  buffer[pos++] = ',';
  buffer[pos++] = 'R';
  buffer[pos++] = 'S';
  buffer[pos++] = 'S';
  buffer[pos++] = 'I';
  buffer[pos++] = ':';

  // Extract RSSI value (signed integer)
  int8_t rssi = data->rx_ctrl.rssi;

  if (rssi < 0) {
    buffer[pos++] = '-';
    rssi = -rssi;
  }

  // Fast integer conversion for RSSI
  if (rssi >= 100) {
    buffer[pos++] = '0' + (rssi / 100);
    buffer[pos++] = '0' + ((rssi / 10) % 10);
    buffer[pos++] = '0' + (rssi % 10);
  } else if (rssi >= 10) {
    buffer[pos++] = '0' + (rssi / 10);
    buffer[pos++] = '0' + (rssi % 10);
  } else {
    buffer[pos++] = '0' + rssi;
  }

  buffer[pos++] = '\n';

  // Single write operation instead of multiple Serial.print() calls
  Serial.write(buffer, pos);
}

void setup() {
  // Increase serial baud rate for higher throughput
  Serial.begin(BAUD_RATE);
  Serial.setTxBufferSize(SERIAL_TX_BUFFER);
  delay(1000);  // Give time for Serial to init

  // Disable power saving for maximum performance
  WiFi.setSleep(false);

  WiFi.mode(WIFI_MODE_STA);
  WiFi.disconnect(); // Ensure not connected to AP

  // Set WiFi to promiscuous mode
  esp_wifi_set_promiscuous(true);

  // Optional: Set specific channel (1-13) for more consistent capture
  // esp_wifi_set_channel(6, WIFI_SECOND_CHAN_NONE);

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

  // Increase WiFi RX buffer size for better packet capture
  esp_wifi_set_promiscuous_rx_cb(NULL);  // Disable packet callback (we only want CSI)
}

void loop() {
  // No delay needed - CSI data streams continuously via callback
  // The wifi_csi_cb() interrupt handler processes packets as they arrive
}
