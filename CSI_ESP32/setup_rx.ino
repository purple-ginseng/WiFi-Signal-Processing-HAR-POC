/*
  ESP32 CSI/RSSI Receiver (RX)
  -----------------------------
  Purpose: Capture CSI and RSSI data from transmitter ESP32

  This receiver captures WiFi packets sent by the TX ESP32 and extracts:
  - Channel State Information (CSI) - 128 values per packet
  - Received Signal Strength Indicator (RSSI) - 1 value per packet

  Configuration:
  - Baud Rate: 921600 (high-speed serial for Python capture)
  - Channel: 6 (must match transmitter)
  - Promiscuous Mode: Enabled for CSI capture

  Upload this to the RECEIVER ESP32
*/

#include <WiFi.h>
#include "esp_wifi.h"

// =================== CONFIGURATION PARAMETERS ===================
// CSI CAPTURE SETTINGS
const bool CSI_LLT_ENABLE       = true;    // Enable legacy long training field
const bool CSI_LTF_MERGE        = true;    // Merge HT-LTFs for better CSI
const bool CSI_CHANNEL_FILTER   = false;   // Disable channel filter
const bool CSI_MANUAL_SCALE     = false;   // Automatic scaling
const bool CSI_SHIFT_ENABLE     = true;    // Enable shift for dynamic range
const bool CSI_ENABLED          = true;    // Master CSI toggle

// SERIAL PERFORMANCE (must match Python script)
const int BAUD_RATE             = 921600;  // High-speed serial
const int SERIAL_TX_BUFFER      = 2048;    // Large buffer to prevent blocking

// LED INDICATOR (VCC-GND YD-ESP32-S3 RGB WS2812 LED)
const int LED_PIN               = 48;      // VCC-GND YD-ESP32-S3 RGB LED on GPIO 48
const int LED_BLINK_DURATION    = 20;      // LED on duration in ms
const int LED_BRIGHTNESS        = 50;      // LED brightness (0-255)
// NOTE: RGB jumper must be soldered for LED to work!

// WIFI SETTINGS
const int FIXED_CHANNEL         = 6;       // Must match TX channel
// ================================================================

// Statistics tracking
static uint32_t csi_packets_received = 0;
static uint32_t last_stats_time = 0;
static uint32_t packets_in_window = 0;
static uint32_t led_off_time = 0;

// =================== CSI CALLBACK ===================
// Optimized callback with minimal processing overhead
void wifi_csi_cb(void *ctx, wifi_csi_info_t *data) {
  if (!data || !data->buf || data->len == 0) return;

  // Pre-allocated buffer to avoid memory fragmentation
  static char buffer[512];
  int pos = 0;

  // Write CSI header
  buffer[pos++] = 'C';
  buffer[pos++] = 'S';
  buffer[pos++] = 'I';
  buffer[pos++] = ':';

  // Fast integer-to-string conversion for CSI values
  for (int i = 0; i < data->len; i++) {
    int8_t val = (int8_t)data->buf[i];

    // Handle negative values
    if (val < 0) {
      buffer[pos++] = '-';
      val = -val;
    }

    // Convert to ASCII without sprintf (faster)
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

    // Add comma separator
    if (i < data->len - 1) buffer[pos++] = ',';
  }

  // Add RSSI data to same line (space-efficient format)
  buffer[pos++] = ',';
  buffer[pos++] = 'R';
  buffer[pos++] = 'S';
  buffer[pos++] = 'S';
  buffer[pos++] = 'I';
  buffer[pos++] = ':';

  // Extract RSSI value (typically -30 to -90 dBm)
  int8_t rssi = data->rx_ctrl.rssi;

  // Handle negative RSSI
  if (rssi < 0) {
    buffer[pos++] = '-';
    rssi = -rssi;
  }

  // Fast RSSI conversion
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

  // Terminate line
  buffer[pos++] = '\n';

  // Single optimized write operation
  Serial.write(buffer, pos);

  // Update statistics
  csi_packets_received++;
  packets_in_window++;

  // Flash green LED on packet reception
  neopixelWrite(LED_PIN, 0, LED_BRIGHTNESS, 0);  // Green
  led_off_time = millis() + LED_BLINK_DURATION;
}

// =================== SETUP ===================
void setup() {
  // Initialize high-speed serial
  Serial.begin(BAUD_RATE);
  Serial.setTxBufferSize(SERIAL_TX_BUFFER);
  delay(500);

  Serial.println("===========================================");
  Serial.println("ESP32 CSI/RSSI Receiver (RX)");
  Serial.println("===========================================");

  // Initialize WS2812 RGB LED (turn off initially)
  neopixelWrite(LED_PIN, 0, 0, 0);  // RGB: Off

  // LED startup sequence (2 slow blinks in green to differentiate from TX)
  for (int i = 0; i < 2; i++) {
    neopixelWrite(LED_PIN, 0, LED_BRIGHTNESS, 0);  // Green
    delay(200);
    neopixelWrite(LED_PIN, 0, 0, 0);  // Off
    delay(200);
  }

  // Disable WiFi sleep for consistent reception
  WiFi.setSleep(false);

  // Set WiFi to station mode
  WiFi.mode(WIFI_MODE_STA);
  WiFi.disconnect();

  Serial.printf("MAC Address: %s\n", WiFi.macAddress().c_str());
  Serial.printf("Channel: %d\n", FIXED_CHANNEL);
  Serial.printf("Baud Rate: %d\n", BAUD_RATE);

  // Enable promiscuous mode for CSI capture
  esp_wifi_set_promiscuous(true);

  // Set fixed channel (must match transmitter)
  esp_wifi_set_channel(FIXED_CHANNEL, WIFI_SECOND_CHAN_NONE);

  if (CSI_ENABLED) {
    // Configure CSI parameters
    wifi_csi_config_t csi_config = {
      .lltf_en           = CSI_LLT_ENABLE,
      .ltf_merge_en      = CSI_LTF_MERGE,
      .channel_filter_en = CSI_CHANNEL_FILTER,
      .manu_scale        = CSI_MANUAL_SCALE,
      .shift             = CSI_SHIFT_ENABLE
    };

    // Apply CSI configuration
    esp_wifi_set_csi_config(&csi_config);
    esp_wifi_set_csi_rx_cb(wifi_csi_cb, NULL);
    esp_wifi_set_csi(true);

    Serial.println("[OK] CSI capture enabled");
  }

  // Disable packet callback (CSI callback handles everything)
  esp_wifi_set_promiscuous_rx_cb(NULL);

  Serial.printf("LED Indicator: GPIO %d (WS2812 RGB LED - Green)\n", LED_PIN);
  Serial.println("===========================================");
  Serial.println("Waiting for CSI data from transmitter...\n");

  last_stats_time = millis();
}

// =================== MAIN LOOP ===================
void loop() {
  // CSI data streams continuously via interrupt callback
  uint32_t now = millis();

  // Turn off LED after blink duration
  if (led_off_time > 0 && now >= led_off_time) {
    neopixelWrite(LED_PIN, 0, 0, 0);  // Off
    led_off_time = 0;
  }

  // Print statistics every 2 seconds
  if (now - last_stats_time >= 2000) {
    float actual_rate = packets_in_window / 2.0;

    Serial.printf("RX | Packets: %6lu | Rate: %5.1f Hz | Total: %lu\n",
                  packets_in_window, actual_rate, csi_packets_received);

    packets_in_window = 0;
    last_stats_time = now;
  }

  // Small delay to prevent watchdog timeout
  delay(10);
}
