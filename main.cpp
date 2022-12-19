#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <ostream>
#include <span>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <CL/cl.h>

#include "libs/spng.h"

const char *getErrorString(cl_int error) {
  switch (error) {
  // run-time and JIT compiler errors
  case 0:
    return "CL_SUCCESS";
  case -1:
    return "CL_DEVICE_NOT_FOUND";
  case -2:
    return "CL_DEVICE_NOT_AVAILABLE";
  case -3:
    return "CL_COMPILER_NOT_AVAILABLE";
  case -4:
    return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5:
    return "CL_OUT_OF_RESOURCES";
  case -6:
    return "CL_OUT_OF_HOST_MEMORY";
  case -7:
    return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8:
    return "CL_MEM_COPY_OVERLAP";
  case -9:
    return "CL_IMAGE_FORMAT_MISMATCH";
  case -10:
    return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -11:
    return "CL_BUILD_PROGRAM_FAILURE";
  case -12:
    return "CL_MAP_FAILURE";
  case -13:
    return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14:
    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15:
    return "CL_COMPILE_PROGRAM_FAILURE";
  case -16:
    return "CL_LINKER_NOT_AVAILABLE";
  case -17:
    return "CL_LINK_PROGRAM_FAILURE";
  case -18:
    return "CL_DEVICE_PARTITION_FAILED";
  case -19:
    return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

  // compile-time errors
  case -30:
    return "CL_INVALID_VALUE";
  case -31:
    return "CL_INVALID_DEVICE_TYPE";
  case -32:
    return "CL_INVALID_PLATFORM";
  case -33:
    return "CL_INVALID_DEVICE";
  case -34:
    return "CL_INVALID_CONTEXT";
  case -35:
    return "CL_INVALID_QUEUE_PROPERTIES";
  case -36:
    return "CL_INVALID_COMMAND_QUEUE";
  case -37:
    return "CL_INVALID_HOST_PTR";
  case -38:
    return "CL_INVALID_MEM_OBJECT";
  case -39:
    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40:
    return "CL_INVALID_IMAGE_SIZE";
  case -41:
    return "CL_INVALID_SAMPLER";
  case -42:
    return "CL_INVALID_BINARY";
  case -43:
    return "CL_INVALID_BUILD_OPTIONS";
  case -44:
    return "CL_INVALID_PROGRAM";
  case -45:
    return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46:
    return "CL_INVALID_KERNEL_NAME";
  case -47:
    return "CL_INVALID_KERNEL_DEFINITION";
  case -48:
    return "CL_INVALID_KERNEL";
  case -49:
    return "CL_INVALID_ARG_INDEX";
  case -50:
    return "CL_INVALID_ARG_VALUE";
  case -51:
    return "CL_INVALID_ARG_SIZE";
  case -52:
    return "CL_INVALID_KERNEL_ARGS";
  case -53:
    return "CL_INVALID_WORK_DIMENSION";
  case -54:
    return "CL_INVALID_WORK_GROUP_SIZE";
  case -55:
    return "CL_INVALID_WORK_ITEM_SIZE";
  case -56:
    return "CL_INVALID_GLOBAL_OFFSET";
  case -57:
    return "CL_INVALID_EVENT_WAIT_LIST";
  case -58:
    return "CL_INVALID_EVENT";
  case -59:
    return "CL_INVALID_OPERATION";
  case -60:
    return "CL_INVALID_GL_OBJECT";
  case -61:
    return "CL_INVALID_BUFFER_SIZE";
  case -62:
    return "CL_INVALID_MIP_LEVEL";
  case -63:
    return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64:
    return "CL_INVALID_PROPERTY";
  case -65:
    return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66:
    return "CL_INVALID_COMPILER_OPTIONS";
  case -67:
    return "CL_INVALID_LINKER_OPTIONS";
  case -68:
    return "CL_INVALID_DEVICE_PARTITION_COUNT";

  // extension errors
  case -1000:
    return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case -1001:
    return "CL_PLATFORM_NOT_FOUND_KHR";
  case -1002:
    return "CL_INVALID_D3D10_DEVICE_KHR";
  case -1003:
    return "CL_INVALID_D3D10_RESOURCE_KHR";
  case -1004:
    return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1005:
    return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
  default:
    return "Unknown OpenCL error";
  }
}

std::byte safeSubtract(std::byte a, std::byte b) {
  int ia = (int)a;
  int ib = (int)b;
  int result = ia - ib;
  if (result < 0) {
    result = 0;
  }
  return std::byte(result);
}

std::byte safeAdd(std::byte a, std::byte b) {
  int ia = (int)a;
  int ib = (int)b;
  int result = ia + ib;
  if (result > 255) {
    result = 255;
  }
  return std::byte(result);
}

struct Pixel {
  std::byte r;
  std::byte g;
  std::byte b;
  std::byte a;

  Pixel operator-(const Pixel &other) const {
    // Do safe subtraction
    return {.r = safeSubtract(r, other.r),
            .g = safeSubtract(g, other.g),
            .b = safeSubtract(b, other.b),
            .a = std::max(a, other.a)};
  }

  Pixel operator+(const Pixel &other) const {
    return {.r = safeAdd(r, other.r),
            .g = safeAdd(g, other.g),
            .b = safeAdd(b, other.b),
            .a = std::max(a, other.a)};
  }

  Pixel operator*(float f) const {
    return {.r = std::byte((int)r * f),
            .g = std::byte((int)g * f),
            .b = std::byte((int)b * f),
            .a = a};
  }

  float magnitude() const {
    const float R = float(r) / 255.0f;
    const float G = float(g) / 255.0f;
    const float B = float(b) / 255.0f;
    return std::sqrt(R * R + G * G + B * B);
  }
};

struct Kernel {
  std::vector<float> data;
  int width;
  int height;
  float total;
};

struct Image {
  std::vector<Pixel> pixels;
  int width;
  int height;

  Image sub(const Image &other) const {
    assert(width == other.width);
    assert(height == other.height);

    Image result;
    result.width = width;
    result.height = height;

    result.pixels.resize(pixels.size());
    for (size_t i = 0; i < pixels.size(); i++) {
      result.pixels[i] = pixels[i] - other.pixels[i];
    }

    return result;
  }

  void bluify() {
    for (auto &pixel : pixels) {
      pixel.r = std::byte(0);
      pixel.g = std::byte(0);
    }
  }

  // rotate 90 degrees clockwise
  void rotate() {
    assert(width == height);

    for (int i = 0; i < width; i++) {
      for (int j = 0; j < i; j++) {
        std::swap(pixels[i * width + j], pixels[j * width + i]);
      }
    }
  }

  void multiply(float f) {
    for (auto &pixel : pixels) {
      pixel = pixel * f;
    }
  }

  Image add(const Image &other) const {
    assert(width == other.width);
    assert(height == other.height);

    Image result;
    result.width = width;
    result.height = height;

    result.pixels.resize(pixels.size());
    for (size_t i = 0; i < pixels.size(); i++) {
      result.pixels[i] = pixels[i] + other.pixels[i];
    }

    return result;
  }

  Image mult(const Image &other) const {
    assert(width == other.width);
    assert(height == other.height);

    Image result;
    result.width = width;
    result.height = height;

    result.pixels.resize(pixels.size());
    for (size_t i = 0; i < pixels.size(); i++) {
      result.pixels[i] = pixels[i] * other.pixels[i].magnitude();
    }

    return result;
  }

  void convolve(const Kernel &kernel) {
    assert(kernel.width % 2 == 1);
    assert(kernel.height % 2 == 1);

    const int halfWidth = kernel.width / 2;
    const int halfHeight = kernel.height / 2;

    std::vector<Pixel> result(pixels.size());

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        float r = 0.0f;
        float g = 0.0f;
        float b = 0.0f;
        float a = 0.0f;

        for (int ky = 0; ky < kernel.height; ky++) {
          for (int kx = 0; kx < kernel.width; kx++) {
            const int px = x + kx - halfWidth;
            const int py = y + ky - halfHeight;

            if (px < 0 || px >= width || py < 0 || py >= height) {
              continue;
            }

            const Pixel &pixel = pixels[py * width + px];
            const float weight =
                kernel.data[ky * kernel.width + kx] / kernel.total;

            r += float(pixel.r) * weight;
            g += float(pixel.g) * weight;
            b += float(pixel.b) * weight;
            a += float(pixel.a) * weight;
          }
        }

        result[y * width + x] = {.r = std::byte(r),
                                 .g = std::byte(g),
                                 .b = std::byte(b),
                                 .a = std::byte(255)};
      }
    }

    pixels = result;
  }
};

Image imageFromFloats(float *data, size_t width, size_t height) {
  Image image;
  image.width = width;
  image.height = height;

  image.pixels.resize(width * height);
  for (size_t i = 0; i < image.pixels.size(); i++) {
    image.pixels[i] = {.r = std::byte(data[i * 3] * 255.0f),
                       .g = std::byte(data[i * 3 + 1] * 255.0f),
                       .b = std::byte(data[i * 3 + 2] * 255.0f),
                       .a = std::byte(255)};
  }

  return image;
}

class ImageWriter {
private:
  spng_ctx *ctx;

public:
  void writePNG(const std::string &filename, const std::vector<Pixel> &pixels,
                unsigned width, unsigned height) {
    ctx = spng_ctx_new(SPNG_CTX_ENCODER);
    assert(pixels.size() == width * height);

    // open file
    FILE *fp = fopen(filename.c_str(), "wb");
    if (fp == nullptr) {
      throw std::runtime_error(std::string("fopen() failed: ") +
                               strerror(errno));
    }

    spng_set_png_file(ctx, fp);

    int ret = 0;

    struct spng_ihdr ihdr = {0};

    ihdr.width = width;
    ihdr.height = height;
    ihdr.bit_depth = 8;
    ihdr.color_type = SPNG_COLOR_TYPE_TRUECOLOR_ALPHA;
    ihdr.bit_depth = 8;

    spng_set_ihdr(ctx, &ihdr);

    ret = spng_encode_image(ctx, pixels.data(), pixels.size() * sizeof(Pixel),
                            SPNG_FMT_PNG, SPNG_ENCODE_FINALIZE);

    if (ret) {
      throw std::runtime_error(std::string("spng_encode_image() failed:") +
                               spng_strerror(ret));
    }

    fclose(fp);
    spng_ctx_free(ctx);
  }

  void writePNG(const std::string &filename, const Image &image) {
    writePNG(filename, image.pixels, image.width, image.height);
  }
};

Image createTestImage() {
  Image image;
  image.width = 256;
  image.height = 256;
  image.pixels.resize(image.width * image.height);
  for (int y = 0; y < image.height; y++) {
    for (int x = 0; x < image.width; x++) {
      Pixel &pixel = image.pixels[y * image.width + x];
      pixel.r = std::byte(x);
      pixel.g = std::byte(y);
      pixel.b = std::byte(0);
      pixel.a = std::byte(255);
    }
  }
  return image;
}

namespace std {
ostream &operator<<(ostream &os, const Pixel &pixel) {
  os << "Pixel(" << (int)pixel.r << ", " << (int)pixel.g << ", " << (int)pixel.b
     << ", " << (int)pixel.a << ")";
  return os;
}
} // namespace std

void testImageWriter() {
  Image image = createTestImage();
  ImageWriter writer;
  writer.writePNG("test.png", image);
  writer.writePNG("test2.png", image.pixels, image.width, image.height);
}

struct Vector;

struct Coordinate {
  int x;
  int y;
  Coordinate operator+(const Coordinate &other) const {
    return {.x = x + other.x, .y = y + other.y};
  }
  Vector operator-(const Vector &other) const;
};

struct Vector {
  float x;
  float y;
  Vector dot(const Vector &other) const {
    return {.x = x * other.x, .y = y * other.y};
  }
  Vector operator-(const Vector &other) const {
    return {.x = x - other.x, .y = y - other.y};
  }
  Vector operator-(const Coordinate &other) const {
    return {.x = x - other.x, .y = y - other.y};
  }
  Vector operator+(const Vector &other) const {
    return {.x = x + other.x, .y = y + other.y};
  }
  Vector operator+(const Coordinate &other) const {
    return {.x = x + other.x, .y = y + other.y};
  }
  float length() const { return sqrt(x * x + y * y); }
  Vector operator*(float scalar) const {
    return {.x = x * scalar, .y = y * scalar};
  }
  Vector operator/(float scalar) const {
    return {.x = x / scalar, .y = y / scalar};
  }
};

Vector Coordinate::operator-(const Vector &other) const {
  return {.x = x - other.x, .y = y - other.y};
}

Vector hash(const Coordinate &coord, int seed) {
  int hash = coord.x * 73856093 ^ coord.y * 19349663 ^ seed * 83492791;
  hash = ((hash >> 13) ^ hash >> 7);
  float angle = (float)hash;
  Vector result{.x = cosf(angle), .y = sinf(angle)};
  return result;
}

const std::array<Coordinate, 4> offsets = {
    Coordinate{.x = 0, .y = 0}, Coordinate{.x = 1, .y = 0},
    Coordinate{.x = 0, .y = 1}, Coordinate{.x = 1, .y = 1}};

float smoothstep(float edge0, float edge1, float x) {
  // Scale, bias and saturate x to 0..1 range
  x = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
  // Evaluate polynomial
  return x * x * (3 - 2 * x);
}

float lerp(float a, float b, float t) { return a + (b - a) * t; }

const float noiseAt(const Vector &where, float bias = 0.0f, int seed = 0) {
  const Coordinate coord = {.x = (int)where.x, .y = (int)where.y};

  std::array<Vector, 4> gradients;
  for (int i = 0; i < 4; i++) {
    gradients[i] = hash(coord + offsets[i], seed);
  }

  std::array<Vector, 4> diffs;
  for (int i = 0; i < 4; i++) {
    diffs[i] = where - (coord + offsets[i]);
  }

  std::array<Vector, 4> dotProducts;
  for (int i = 0; i < 4; i++) {
    dotProducts[i] = diffs[i].dot(gradients[i]);
  }

  // Do interpolation
  const float x = smoothstep(0.0f, 1.0f, where.x - coord.x);
  const float y = smoothstep(0.0f, 1.0f, where.y - coord.y);

  const float top = lerp(dotProducts[0].x, dotProducts[1].x, x);
  const float bottom = lerp(dotProducts[2].x, dotProducts[3].x, x);
  const float result = lerp(top, bottom, y);

  return std::clamp(result + bias, -1.0f, 1.0f);
}

struct OrderDensities {
  float totalDensity;
  std::vector<std::tuple<int, float, float>> orders;
  OrderDensities(std::vector<std::tuple<int, float, float>> &&orders) {
    this->orders = std::move(orders);
    for (auto &order : this->orders) {
      totalDensity += std::get<1>(order);
    }
  }
};

float multiOrderNoiseAt(const Vector &where, const OrderDensities &orders,
                        float bias = 0.0f, int seed = 0) {
  float result = 0.0f;
  for (auto &order : orders.orders) {
    result += std::get<1>(order) / orders.totalDensity *
              noiseAt(where / std::get<0>(order), std::get<2>(order), seed);
  }
  return std::clamp(result + bias, -1.0f, 1.0f);
}

// clang-format off
const auto orders2 = OrderDensities{std::move(std::vector<std::tuple<int, float, float>>{
  {2, 2.0f, 0.0f},
  // {5, 1.0f, 0.0f},
  {64, 1.0f, 0.3f},
  {70, 1.0f, 0.3f},
  {30, 1.0f, 0.3f},
  {110, 1.0f, 0.3f},
  {202, 1.0f, 0.3f},
  {500, 2.0f, 0.0f},
  {868, 2.0f, 0.0f}
})};

const auto orders = OrderDensities{std::move(std::vector<std::tuple<int, float, float>>{
  // {2, 2.0f, 0.0f},
  {5, 1.0f, 0.1f},
  // {135, 1.0f, 0.0f},
  {800, 1.0f, 0.0f},
  {915, 1.0f, 0.0f},
  {1300, 1.0f, 0.0f},
  {1630, 1.0f, 0.0f},
  {1768, 3.0f, 0.0f}
})};

const auto highOrder = OrderDensities{std::move(std::vector<std::tuple<int, float, float>>{
  {3, 2.0f, 0.0f},
  {2, 1.0f, 0.0f},
})};

const auto highOrder2 = OrderDensities{std::move(std::vector<std::tuple<int, float, float>>{
  {5, 1.0f, 0.0f},
  {7, 1.0f, 0.0f},
})};
// clang-format on

Image perlinNoise(int width, int height, const OrderDensities &orders,
                  int seed = 0) {
  Image image;
  image.width = width;
  image.height = height;
  image.pixels.resize(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float noise = multiOrderNoiseAt(Vector{.x = (float)x, .y = (float)y},
                                      orders, -0.5f, seed);
      Pixel &pixel = image.pixels[y * width + x];
      pixel.r = std::byte((noise + 1.0f) * 127.0f);
      pixel.g = std::byte((noise + 1.0f) * 127.0f);
      pixel.b = std::byte((noise + 1.0f) * 127.0f);
      pixel.a = std::byte(255);
    }
  }
  return image;
}

int oldMain(int argc, char **argv) {
  Image image = perlinNoise(4096, 4096, orders);
  Image image2 = perlinNoise(4096, 4096, orders2, 1);
  Image image3 = perlinNoise(4096, 4096, highOrder, 2);
  Image image4 = perlinNoise(4096, 4096, highOrder2, 3);

  // 5x5 gaussian kernel
  const Kernel kernel{{1.0f,  4.0f,  6.0f,  4.0f,  1.0f,  4.0f,  16.0f,
                       24.0f, 16.0f, 4.0f,  6.0f,  24.0f, 36.0f, 24.0f,
                       6.0f,  4.0f,  16.0f, 24.0f, 16.0f, 4.0f,  1.0f,
                       4.0f,  6.0f,  4.0f,  1.0f},
                      5,
                      5,
                      256.0f};

  image2.rotate();
  image2.convolve(kernel);
  image2.convolve(kernel);
  image2.multiply(1.2f);

  Image image5 = image2.sub(image);
  image5.multiply(3.0f);

  Image image6 = image3.mult(image5);
  Image image7 = image4.mult(image5);

  image6.multiply(0.5f);
  image7.multiply(0.5f);

  image7.bluify();
  // 3x3 gaussian kernel
  const Kernel kernel2{
      {1.0f, 2.0f, 1.0f, 2.0f, 4.0f, 2.0f, 1.0f, 2.0f, 1.0f}, 3, 3, 16.0f};

  image7.convolve(kernel2);

  Image image8 = image6.add(image7);
  Image image9 = image8.add(image7);

  image8.convolve(kernel2);
  image8.convolve(kernel2);
  image8.convolve(kernel2);
  image8.convolve(kernel2);
  image8.convolve(kernel2);
  image8.convolve(kernel2);
  image8.convolve(kernel2);
  image8.convolve(kernel2);

  image8.multiply(0.8f);
  Image image10 = image8.add(image9);

  ImageWriter writer;
  writer.writePNG("noise.png", image10);
  // writer.writePNG("noise2.png", image5);
  // writer.writePNG("noise3.png", image6);
  return 0;
}

class Chunk : public std::enable_shared_from_this<Chunk> {
  float *data;
  size_t dimension;
  size_t offset[2];
  std::mutex mutex;
  std::string swapFileName;

public:
  Chunk(size_t dimension, std::array<size_t, 2> offset) {
    assert(offset[0] % dimension == 0);
    assert(offset[1] % dimension == 0);
    this->dimension = dimension;
    this->offset[0] = offset[0];
    this->offset[1] = offset[1];
    data = new float[dimension * dimension * 3];
    std::memset(data, 0, dimension * dimension * 3 * sizeof(float));
    swapFileName = std::to_string(offset[0] / dimension) + "_" +
                   std::to_string(offset[1] / dimension);
  }
  ~Chunk() { delete[] data; }
  std::thread dumpPNG() {
    std::string filename = "chunk" + swapFileName + ".png";
    const auto self = shared_from_this();
    std::thread t([self, filename] {
      std::scoped_lock lock(self->mutex);
      ImageWriter writer;
      Image image =
          imageFromFloats(self->data, self->dimension, self->dimension);
      writer.writePNG(filename, image);
      std::cout << "Wrote " << filename << std::endl;
    });
    return t;
  }
  Chunk(const Chunk &) = delete;
  Chunk &operator=(const Chunk &) = delete;
  Chunk(Chunk &&) = delete;
  Chunk &operator=(Chunk &&) = delete;

  void swapOutToDisk() {
    if (mutex.try_lock()) {
      std::ofstream file(swapFileName, std::ios::binary);
      file.write((char *)data, dimension * dimension * 3 * sizeof(float));
      file.close();
      delete[] data;
    } else {
      std::cout << "Failed to lock mutex (aborting): " << swapFileName
                << std::endl;
    }
  }

  void swapInFromDisk() {
    data = new float[dimension * dimension * 3];
    std::ifstream file(swapFileName, std::ios::binary);
    file.read((char *)data, dimension * dimension * 3 * sizeof(float));
    file.close();
    mutex.unlock();
  }

  void generate(const cl_context &context, const cl_command_queue &queue, const cl_kernel &kernel, const cl_mem &outBuffer) {
    if (!mutex.try_lock()) {
      std::cout << "Failed to lock mutex (aborting): " << swapFileName << std::endl;
      return;
    }

    std::cout << "Generating " << swapFileName << std::endl;

    cl_int ret{CL_SUCCESS};

    size_t globalWorkSize[2] = {dimension, dimension};
    size_t localWorkSize[2] = {16, 16};

    ret = clEnqueueNDRangeKernel(queue, kernel, 2, this->offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
      std::cout << "Error enqueuing kernel" << std::endl;
      throw std::runtime_error("Error enqueuing kernel");
    }

    ret = clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, dimension * dimension * 3 * sizeof(float), data, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
      std::cout << "Error reading buffer: " << getErrorString(ret) << std::endl;
      // Get the error message
      throw std::runtime_error("Error reading buffer");
    }

    mutex.unlock();
  }
};

int main(int argc, char const *argv[]) {
  // Read in the shader
  std::ifstream t("gen.ocl");
  std::string source_str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

  const char *source = source_str.c_str();

  // Get platform and device information
  cl_platform_id platform_id = NULL;
  cl_device_id device_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  if (ret != CL_SUCCESS) {
    std::cout << "Error getting platform IDs" << std::endl;
    throw std::runtime_error("Error getting platform IDs");
  }

  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
  if (ret != CL_SUCCESS) {
    std::cout << "Error getting device IDs" << std::endl;
    throw std::runtime_error("Error getting device IDs");
  }

  cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

  // Create a command queue
  cl_command_queue command_queue =
      clCreateCommandQueue(context, device_id, 0, &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating command queue" << std::endl;
    throw std::runtime_error("Error creating command queue");
  }

  const size_t sourceSize = strlen(source);
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, &sourceSize, &ret);

  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  if (ret != CL_SUCCESS) {
    std::cout << "Error building program" << std::endl;
    size_t logSize = 10000;
    char *buildLog = new char[logSize + 1];
    buildLog[logSize] = '\0';
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, logSize, buildLog, NULL);
    std::cout << buildLog << std::endl;
    delete[] buildLog;
    throw std::runtime_error("Error building program");
  }

  cl_kernel kernel = clCreateKernel(program, "perlin", &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating kernel" << std::endl;
    throw std::runtime_error("Error creating kernel");
  }

  cl_mem outBuffer = clCreateBuffer(
      context, CL_MEM_WRITE_ONLY, 2048 * 2048 * 3 * sizeof(float), NULL, &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating buffer" << std::endl;
    throw std::runtime_error("Error creating buffer");
  }

  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &outBuffer);
  if (ret != CL_SUCCESS) {
    std::cout << "Error setting kernel arg" << std::endl;
    throw std::runtime_error("Error setting kernel arg");
  }

  std::vector <std::thread> threads;

  auto chunk = std::make_shared<Chunk>(2048, std::array<size_t, 2>({0UL, 0UL}));
  chunk->generate(context, command_queue, kernel, outBuffer);
  threads.push_back(chunk->dumpPNG());

  chunk = std::make_shared<Chunk>(2048, std::array<size_t, 2>({2048UL, 0UL}));
  chunk->generate(context, command_queue, kernel, outBuffer);
  threads.push_back(chunk->dumpPNG());

  clReleaseMemObject(outBuffer);

  for (auto &thread : threads) {
    thread.join();
  }

  return 0;
}
