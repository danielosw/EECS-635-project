// ASL sign classifier on the Seeed XIAO ESP32S3 Sense.
// Captures a frame, resizes it to the model's input size, runs the int8 TFLite model,
// and prints the predicted sign, confidence, and timing over Serial.
//
// Setup:
//   1. Copy models/asl_int8_model.cc from your PC into this project as src/asl_int8_model.h
//      (just rename it to .h; it defines g_model[] and g_model_len).
//   2. pio run -t upload && pio device monitor
//
// UNTESTED starting point: the TFLite library API differs between versions. See the notes
// marked "VERSION NOTE" if it fails to compile.

#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>
#include "esp_camera.h"
#include "esp_heap_caps.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "asl_int8_model.h"  // g_model[], g_model_len

// ---- XIAO ESP32S3 Sense camera pins (verify against Seeed's wiki if the camera fails) ----
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  10
#define SIOD_GPIO_NUM  40
#define SIOC_GPIO_NUM  39
#define Y9_GPIO_NUM    48
#define Y8_GPIO_NUM    11
#define Y7_GPIO_NUM    12
#define Y6_GPIO_NUM    14
#define Y5_GPIO_NUM    16
#define Y4_GPIO_NUM    18
#define Y3_GPIO_NUM    17
#define Y2_GPIO_NUM    15
#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM  47
#define PCLK_GPIO_NUM  13

// Class order = sorted folder names from training. This assumes folders 0-9 then A-Z;
// check it against models/labels.txt.
static const char kLabels[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
static const int kNumClasses = 36;

// Tensor arena, allocated in PSRAM. Start generous, then check arena_used_bytes().
static const size_t kArenaSize = 1536 * 1024;

static tflite::MicroMutableOpResolver<8> resolver;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

static bool initCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565;   // raw pixels, no JPEG decoding needed
  config.frame_size = FRAMESIZE_240X240;    // square, then downscaled to the model size
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }
  return true;
}

static bool initModel() {
  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model schema %d != supported %d\n", model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  // Only the ops this model uses. If AllocateTensors() fails with a message like
  // "Didn't find op for builtin opcode 'X'", add the matching Add...() line here.
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddMean();            // GlobalAveragePooling2D
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();

  uint8_t* arena = (uint8_t*)heap_caps_aligned_alloc(16, kArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!arena) {
    Serial.println("Could not allocate the tensor arena in PSRAM (is PSRAM enabled?)");
    return false;
  }

  // VERSION NOTE: older library versions take an extra error-reporter argument here:
  //   new tflite::MicroInterpreter(model, resolver, arena, kArenaSize, &error_reporter);
  interpreter = new tflite::MicroInterpreter(model, resolver, arena, kArenaSize);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed (arena too small, or a missing op; see above)");
    return false;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.printf("Model ready. Input %dx%d type=%d scale=%.6f zp=%d | arena used: %u bytes\n",
                input->dims->data[1], input->dims->data[2], (int)input->type,
                input->params.scale, input->params.zero_point,
                (unsigned)interpreter->arena_used_bytes());
  return input->type == kTfLiteInt8;
}

// RGB565 frame -> int8 model input. Nearest-neighbor sampling, which matches the default
// resize used by Keras flow_from_directory during training.
static void fillInput(const camera_fb_t* fb) {
  const int H = input->dims->data[1];
  const int W = input->dims->data[2];
  const float scale = input->params.scale;
  const int zp = input->params.zero_point;
  int8_t* dst = input->data.int8;

  for (int y = 0; y < H; y++) {
    int sy = ((2 * y + 1) * fb->height) / (2 * H);
    for (int x = 0; x < W; x++) {
      int sx = ((2 * x + 1) * fb->width) / (2 * W);
      int idx = (sy * fb->width + sx) * 2;
      uint8_t hb = fb->buf[idx];       // RGB565 from the camera is high byte first
      uint8_t lb = fb->buf[idx + 1];
      uint8_t r = hb & 0xF8;
      uint8_t g = ((hb & 0x07) << 5) | ((lb & 0xE0) >> 3);
      uint8_t b = (lb & 0x1F) << 3;      // swap r and b if colors look wrong

      // Training used pixel/255, so quantize (pixel/255) with the model's scale and zero point.
      // With scale = 1/255 and zero point = -128 this reduces to (pixel - 128).
      int idx_out = (y * W + x) * 3;
      const uint8_t px[3] = {r, g, b};
      for (int c = 0; c < 3; c++) {
        int q = (int)roundf(px[c] / 255.0f / scale) + zp;
        dst[idx_out + c] = (int8_t)constrain(q, -128, 127);
      }
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(2000);  // give the USB serial monitor time to connect
  Serial.printf("PSRAM: %u bytes free\n", (unsigned)ESP.getFreePsram());

  if (!initCamera()) while (true) delay(1000);
  if (!initModel()) while (true) delay(1000);
}

void loop() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Frame capture failed");
    delay(500);
    return;
  }

  fillInput(fb);
  esp_camera_fb_return(fb);

  uint32_t t0 = micros();
  TfLiteStatus status = interpreter->Invoke();
  uint32_t t1 = micros();
  if (status != kTfLiteOk) {
    Serial.println("Invoke() failed");
    return;
  }

  int best = 0;
  for (int i = 1; i < kNumClasses; i++) {
    if (output->data.int8[i] > output->data.int8[best]) best = i;
  }
  float conf = (output->data.int8[best] - output->params.zero_point) * output->params.scale;

  Serial.printf("Predicted: %c  (confidence %.2f)  inference: %.1f ms\n",
                kLabels[best], conf, (t1 - t0) / 1000.0f);
  delay(200);
}