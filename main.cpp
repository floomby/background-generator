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
  case 0: return "CL_SUCCESS";
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -11: return "CL_BUILD_PROGRAM_FAILURE";
  case -12: return "CL_MAP_FAILURE";
  case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15: return "CL_COMPILE_PROGRAM_FAILURE";
  case -16: return "CL_LINKER_NOT_AVAILABLE";
  case -17: return "CL_LINK_PROGRAM_FAILURE";
  case -18: return "CL_DEVICE_PARTITION_FAILED";
  case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

  // compile-time errors
  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALID_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALID_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64: return "CL_INVALID_PROPERTY";
  case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66: return "CL_INVALID_COMPILER_OPTIONS";
  case -67: return "CL_INVALID_LINKER_OPTIONS";
  case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

  // extension errors
  case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
  case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
  case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
  case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
  default: return "Unknown OpenCL error";
  }
}

#include "airyKernel.h"

struct Pixel {
  std::byte r;
  std::byte g;
  std::byte b;
  std::byte a;
};

struct Image {
  std::vector<Pixel> pixels;
  size_t width;
  size_t height;

  // Need for convolution kernels
  std::array<float, 4> integrate() const {
    std::array<float, 4> result = {0.0f, 0.0f, 0.0f, 1.0f};
    for (const auto &pixel : pixels) {
      result[0] += float(pixel.r);
      result[1] += float(pixel.g);
      result[2] += float(pixel.b);
    }
    return result;
  }

  void premultiplyAlpha() {
    for (auto &pixel : pixels) {
      float alpha = float(pixel.a) / 255.0f;
      pixel.r = std::byte(float(pixel.r) * alpha);
      pixel.g = std::byte(float(pixel.g) * alpha);
      pixel.b = std::byte(float(pixel.b) * alpha);
    }
  }
};

Image testConvKernel() {
  Image ret;
  ret.width = 3;
  ret.height = 3;
  ret.pixels.resize(9);
  std::fill(ret.pixels.begin(), ret.pixels.end(), Pixel{std::byte(0), std::byte(0), std::byte(0), std::byte(255)});
  ret.pixels[4] = Pixel{std::byte(255), std::byte(255), std::byte(255), std::byte(255)};
  return ret;
}

// Slow and stupid
Image imageFromFloats(float *data, size_t width, size_t height) {
  Image image;
  image.width = width;
  image.height = height;

  image.pixels.resize(width * height);
  for (size_t i = 0; i < image.pixels.size(); i++) {
    image.pixels[i] = {
      .r = std::byte(data[i * 4] * 255.0f),
      .g = std::byte(data[i * 4 + 1] * 255.0f),
      .b = std::byte(data[i * 4 + 2] * 255.0f),
      .a = std::byte(data[i * 4 + 3] * 255.0f)
    };
  }

  return image;
}

Image imageFromGimpExport(const GimpExport &img) {
  Image image;
  image.width = img.width;
  image.height = img.height;

  image.pixels.resize(img.width * img.height);
  std::memcpy(image.pixels.data(), img.pixel_data, img.width * img.height * sizeof(Pixel));

  return image;
}

struct ImageWriter {
  static void writePNG(const std::string &filename, const std::vector<Pixel> &pixels, unsigned width, unsigned height) {
    spng_ctx *ctx = spng_ctx_new(SPNG_CTX_ENCODER);
    assert(pixels.size() == width * height);

    // open file
    FILE *fp = fopen(filename.c_str(), "wb");
    if (fp == nullptr) {
      throw std::runtime_error(std::string("fopen() failed: ") + strerror(errno));
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

    ret = spng_encode_image(ctx, pixels.data(), pixels.size() * sizeof(Pixel), SPNG_FMT_PNG, SPNG_ENCODE_FINALIZE);

    if (ret) {
      throw std::runtime_error(std::string("spng_encode_image() failed:") + spng_strerror(ret));
    }

    fclose(fp);
    spng_ctx_free(ctx);
  }

  static void writePNG(const std::string &filename, const Image &image) {
    ImageWriter::writePNG(filename, image.pixels, image.width, image.height);
  }
};

Image createTestImage() {
  Image image;
  image.width = 256;
  image.height = 256;
  image.pixels.resize(image.width * image.height);
  for (size_t y = 0; y < image.height; y++) {
    for (size_t x = 0; x < image.width; x++) {
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
  ImageWriter::writePNG("test.png", image);
  ImageWriter::writePNG("test2.png", image.pixels, image.width, image.height);
}

void testKernelImage() {
  Image image = imageFromGimpExport(airyKernel);
  ImageWriter::writePNG("airyKernel.png", image);
}

class GridOfChunks;

class Chunk : public std::enable_shared_from_this<Chunk> {
  friend class GridOfChunks;
  float *data[2] = { nullptr, nullptr };
  size_t dimension;
  size_t offset[2];
  std::mutex mutex;

  std::shared_ptr<Chunk> topLeft;
  std::shared_ptr<Chunk> top;
  std::shared_ptr<Chunk> topRight;
  std::shared_ptr<Chunk> right;
  std::shared_ptr<Chunk> bottomRight;
  std::shared_ptr<Chunk> bottom;
  std::shared_ptr<Chunk> bottomLeft;
  std::shared_ptr<Chunk> left;
public:
  std::string swapFileName;
  Chunk(size_t dimension, std::array<size_t, 2> offset) {
    assert(offset[0] % dimension == 0);
    assert(offset[1] % dimension == 0);
    this->dimension = dimension;
    this->offset[0] = offset[0];
    this->offset[1] = offset[1];
    mutex.lock();
    swapFileName = std::to_string(offset[0] / dimension) + "_" + std::to_string(offset[1] / dimension);
  }
  ~Chunk() {
    if (data[0] != nullptr) {
      delete[] data[0];
    }
    if (data[1] != nullptr) {
      delete[] data[1];
    }
  }
  std::thread dumpPNG() {
    std::string filename = "chunk" + swapFileName + ".png";
    const auto self = shared_from_this();
    std::thread t([self, filename] {
      std::scoped_lock lock(self->mutex);
      Image image = imageFromFloats(self->data[0], self->dimension, self->dimension);
      ImageWriter::writePNG(filename, image);
      std::cout << "Wrote " << filename << std::endl;
    });
    return t;
  }
  Chunk(const Chunk &) = delete;
  Chunk &operator=(const Chunk &) = delete;
  Chunk(Chunk &&) = delete;
  Chunk &operator=(Chunk &&) = delete;

  bool canConvolve() {
    return topLeft != nullptr && top != nullptr && topRight != nullptr && right != nullptr &&
           bottomRight != nullptr && bottom != nullptr && bottomLeft != nullptr && left != nullptr;
  }

  std::shared_ptr<Chunk> getNeighbor(size_t i) {
    switch (i) {
    case 0: return topLeft;
    case 1: return top;
    case 2: return topRight;
    case 3: return right;
    case 4: return bottomRight;
    case 5: return bottom;
    case 6: return bottomLeft;
    case 7: return left;
    case 8: return shared_from_this();
    default: throw std::runtime_error("Invalid index");
    }
  }

  void init() {
    if (data[0] != nullptr) {
      return;
    }
    data[0] = new float[dimension * dimension * 4];
    data[1] = new float[dimension * dimension * 4];
    std::memset(data[0], 0, dimension * dimension * 4 * sizeof(float));
    std::memset(data[1], 0, dimension * dimension * 4 * sizeof(float));
    mutex.unlock();
  }

  void swapOutToDisk() {
    if (mutex.try_lock()) {
      std::ofstream file(swapFileName, std::ios::binary);
      file.write((char *)data[0], dimension * dimension * 4 * sizeof(float));
      file.write((char *)data[1], dimension * dimension * 4 * sizeof(float));
      file.close();
      delete[] data[0];
      delete[] data[1];
    } else {
      std::cout << "Failed to lock mutex (aborting): " << swapFileName << std::endl;
    }
  }

  void swapInFromDisk() {
    if (data[0] == nullptr) {
      throw std::runtime_error("Chunk never initialized: " + swapFileName);
    }
    data[0] = new float[dimension * dimension * 4];
    data[1] = new float[dimension * dimension * 4];
    std::ifstream file(swapFileName, std::ios::binary);
    file.read((char *)data[0], dimension * dimension * 4 * sizeof(float));
    file.read((char *)data[1], dimension * dimension * 4 * sizeof(float));
    file.close();
    mutex.unlock();
  }

  void clear(const cl_context &context, const cl_command_queue &queue, const cl_kernel &kernel, size_t layer = 0) {
    assert(layer < 2);

    if (data[0] == nullptr) {
      throw std::runtime_error("Chunk never initialized: " + swapFileName);
    }

    if (!mutex.try_lock()) {
      std::cout << "Failed to lock mutex (aborting): " << swapFileName << std::endl;
      return;
    }

    std::cout << "Clearing " << swapFileName << std::endl;

    cl_int ret{CL_SUCCESS};

    size_t globalWorkSize[2] = {dimension, dimension};
    size_t localWorkSize[2] = {16, 16};



    ret = clEnqueueNDRangeKernel(queue, kernel, 2, this->offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
      std::cout << "Error enqueuing kernel" << std::endl;
      throw std::runtime_error("Error enqueuing kernel");
    }
  }

  void generate(const cl_context &context, const cl_command_queue &queue, const cl_kernel &kernel, const cl_mem &outBuffer, size_t layer = 0) {
    assert(layer < 2);

    if (data[0] == nullptr) {
      throw std::runtime_error("Chunk never initialized: " + swapFileName);
    }

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

    ret = clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, dimension * dimension * 4 * sizeof(float), data[layer], 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
      std::cout << "Error reading buffer: " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error reading buffer");
    }

    mutex.unlock();
  }

  void clear(const cl_context &context, const cl_command_queue &queue, const cl_kernel &kernel, const cl_mem &outBuffer, size_t layer) {
    assert(layer < 2);

    if (data[0] == nullptr) {
      throw std::runtime_error("Chunk never initialized: " + swapFileName);
    }

    if (!mutex.try_lock()) {
      std::cout << "Failed to lock mutex (aborting): " << swapFileName << std::endl;
      return;
    }

    cl_int ret{CL_SUCCESS};

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &outBuffer);
    if (ret != CL_SUCCESS) {
      std::cout << "Error setting kernel arg: " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error setting kernel arg");
    }

    size_t globalWorkSize[2] = {dimension, dimension};
    size_t localWorkSize[2] = {16, 16};
    size_t offset[2] = {0, 0};

    ret = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
      std::cout << "Error enqueuing kernel" << std::endl;
      throw std::runtime_error("Error enqueuing kernel");
    }

    ret = clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, dimension * dimension * 4 * sizeof(float), data[0], 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
      std::cout << "Error reading buffer: " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error reading buffer");
    }

    mutex.unlock();
  }

  // Mixes layer 1 down into layer 0
  void mixDown(const cl_context &context, const cl_command_queue &queue, const cl_kernel &kernel, const cl_mem &outBuffer) {
    if (data[0] == nullptr) {
      throw std::runtime_error("Chunk never initialized: " + swapFileName);
    }

    if (!mutex.try_lock()) {
      std::cout << "Failed to lock mutex (aborting): " << swapFileName << std::endl;
      return;
    }

    std::cout << "Mixing down " << swapFileName << std::endl;

    cl_int ret{CL_SUCCESS};

    cl_mem layer0Buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, dimension * dimension * 4 * sizeof(float), NULL, &ret);
    if (ret != CL_SUCCESS) {
      std::cout << "Error creating buffer: " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error creating buffer");
    }

    cl_mem layer1Buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, dimension * dimension * 4 * sizeof(float), NULL, &ret);
    if (ret != CL_SUCCESS) {
      std::cout << "Error creating buffer: " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error creating buffer");
    }

    ret = clEnqueueWriteBuffer(queue, layer0Buffer, CL_TRUE, 0, dimension * dimension * 4 * sizeof(float), data[0], 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
      std::cout << "Error writing buffer: " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error writing buffer");
    }

    ret = clEnqueueWriteBuffer(queue, layer1Buffer, CL_TRUE, 0, dimension * dimension * 4 * sizeof(float), data[1], 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
      std::cout << "Error writing buffer: " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error writing buffer");
    }

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &layer0Buffer);
    if (ret != CL_SUCCESS) {
      std::cout << "Error setting kernel arg: " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error setting kernel arg");
    }

    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &layer1Buffer);
    if (ret != CL_SUCCESS) {
      std::cout << "Error setting kernel arg: " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error setting kernel arg");
    }

    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &outBuffer);
    if (ret != CL_SUCCESS) {
      std::cout << "Error setting kernel arg: " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error setting kernel arg");
    }

    size_t globalWorkSize[2] = {dimension, dimension};
    size_t localWorkSize[2] = {16, 16};

    size_t offset[2] = {0, 0};

    ret = clEnqueueNDRangeKernel(queue, kernel, 2, offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
      std::cout << "Error enqueuing kernel" << std::endl;
      throw std::runtime_error("Error enqueuing kernel");
    }

    ret = clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, dimension * dimension * 4 * sizeof(float), data[0], 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
      std::cout << "Error reading buffer: " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error reading buffer");
    }

    mutex.unlock();

    clReleaseMemObject(layer0Buffer);
    clReleaseMemObject(layer1Buffer);
  }

  void convolve(const cl_context &context, const cl_command_queue &queue, const cl_kernel &kernel, const cl_mem &outBuffer, size_t layer = 0) {
    assert(layer < 2);

    if (!mutex.try_lock()) {
      std::cout << "Failed to lock mutex (aborting): " << swapFileName << std::endl;
      return;
    }

    std::cout << "Convolving " << swapFileName << std::endl;

    cl_int ret{CL_SUCCESS};

    size_t globalWorkSize[2] = {dimension, dimension};
    size_t localWorkSize[2] = {16, 16};

    ret = clEnqueueNDRangeKernel(queue, kernel, 2, this->offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
      std::cout << "Error enqueuing kernel (convolution)" << std::endl;
      throw std::runtime_error("Error enqueuing kernel");
    }

    ret = clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, dimension * dimension * 4 * sizeof(float), data[layer], 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
      std::cout << "Error reading buffer (convolution): " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error reading buffer");
    }

    mutex.unlock();
  }
};

class GridOfChunks {
  std::shared_ptr<Chunk> dead;
  size_t dimension;

  std::shared_ptr<Chunk> getChunk(size_t x, size_t y) {
    if (x < 0 || y < 0 || x >= cellCount || y >= cellCount) {
      return dead;
    }
    return chunks[x + y * cellCount];
  }

public:
  std::vector<std::shared_ptr<Chunk>> chunks;
  size_t cellCount;
  GridOfChunks(size_t cellCount, size_t dimension) {
    this->cellCount = cellCount;
    this->dimension = dimension;

    dead = std::make_shared<Chunk>(dimension, std::array<size_t, 2>({0, 0}));
    dead->init();

    for (size_t x = 0; x < cellCount; x++) {
      for (size_t y = 0; y < cellCount; y++) {
        std::shared_ptr<Chunk> chunk = std::make_shared<Chunk>(dimension, std::array<size_t, 2>({x * dimension, y * dimension}));
        chunk->topLeft = dead;
        chunk->top = dead;
        chunk->topRight = dead;
        chunk->right = dead;
        chunk->bottomRight = dead;
        chunk->bottom = dead;
        chunk->bottomLeft = dead;
        chunk->left = dead;
        chunks.push_back(chunk);

        chunk->init();
      }
    }

    for (size_t x = 0; x < cellCount; x++) {
      for (size_t y = 0; y < cellCount; y++) {
        std::shared_ptr<Chunk> chunk = chunks[x + y * cellCount];
        chunk->topLeft = getChunk(x - 1, y - 1);
        chunk->top = getChunk(x, y - 1);
        chunk->topRight = getChunk(x + 1, y - 1);
        chunk->right = getChunk(x + 1, y);
        chunk->bottomRight = getChunk(x + 1, y + 1);
        chunk->bottom = getChunk(x, y + 1);
        chunk->bottomLeft = getChunk(x - 1, y + 1);
        chunk->left = getChunk(x - 1, y);
      }
    }
  }

  cl_mem kernBuffer = nullptr;

  void bindConvolutionKernel(
    const cl_context &context,
    const cl_command_queue &queue,
    const cl_kernel &kernel,
    const Image &kern,
    float correctionFactor,
    float clampLevel = 1.0f) {
    cl_int ret{CL_SUCCESS};

    assert(kern.height == kern.width);
    assert(kern.height % 2 == 1);
    kernBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, kern.width * kern.height * sizeof(float) * 4, NULL, &ret);
    if (ret != CL_SUCCESS) {
      std::cout << "Error creating buffer: " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error creating buffer");
    }

    float *tmpBuf = new float[kern.width * kern.height * 4];
    for (size_t i = 0; i < kern.width * kern.height; i++) {
      tmpBuf[i * 4 + 0] = static_cast<float>(kern.pixels[i].r);
      tmpBuf[i * 4 + 1] = static_cast<float>(kern.pixels[i].g);
      tmpBuf[i * 4 + 2] = static_cast<float>(kern.pixels[i].b);
      tmpBuf[i * 4 + 3] = static_cast<float>(kern.pixels[i].a);
    }

    ret = clEnqueueWriteBuffer(queue, kernBuffer, CL_TRUE, 0, kern.width * kern.height * sizeof(float) * 4, tmpBuf, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
      std::cout << "Error writing buffer: " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error writing buffer");
    }

    ret = clSetKernelArg(kernel, 10, sizeof(cl_mem), &kernBuffer);
    if (ret != CL_SUCCESS) {
      std::cout << "Error setting kernel argument: " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error setting kernel argument");
    }

    delete[] tmpBuf;

    int kernSize = kern.width >> 1;
    
    std::cout << "Kernel size: " << kernSize << " (" << kern.width << "x" << kern.height << ")" << std::endl;

    ret = clSetKernelArg(kernel, 11, sizeof(cl_uint), &kernSize);
    if (ret != CL_SUCCESS) {
      std::cout << "Error setting kernel argument: " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error setting kernel argument");
    }

    auto sums = kern.integrate();

    std::cout << "Kernel sums:" << std::endl;
    for (auto &s : sums) {
      std::cout << s << std::endl;
    }

    ret = clSetKernelArg(kernel, 12, sizeof(cl_float) * 4, sums.data());
    if (ret != CL_SUCCESS) {
      std::cout << "Error setting kernel argument: " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error setting kernel argument");
    }

    ret = clSetKernelArg(kernel, 13, sizeof(cl_float), &correctionFactor);
    if (ret != CL_SUCCESS) {
      std::cout << "Error setting kernel argument: " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error setting kernel argument");
    }

    ret = clSetKernelArg(kernel, 14, sizeof(cl_float), &clampLevel);
    if (ret != CL_SUCCESS) {
      std::cout << "Error setting kernel argument: " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error setting kernel argument");
    }
  }

  void convolutionCleanup() {
    clReleaseMemObject(kernBuffer);
  }

  // This is pretty dumb, but I need something to get started (I am pretty sure it still beats cpu convolution)
  // I should at least put it the check for dead chunk to avoid redundant copies
  auto bindForConvolution(
    const cl_context &context,
    const cl_command_queue &queue,
    const cl_kernel &kernel,
    size_t x,
    size_t y,
    const std::array<cl_mem, 9> &buffers,
    size_t layer = 0
  ) {
    assert(layer < 2);
    cl_uint ret{CL_SUCCESS};
    const auto chunk = chunks[x + y * cellCount];

    assert(chunk->canConvolve());

    for (size_t i = 0; i < 9; i++) {
      assert(chunk->getNeighbor(i)->data[0] != nullptr);
      ret = clEnqueueWriteBuffer(queue, buffers[i], CL_TRUE, 0, dimension * dimension * 4 * sizeof(float), chunk->getNeighbor(i)->data[layer], 0, NULL, NULL);
      if (ret != CL_SUCCESS) {
        std::cout << "Error writing buffer: " << getErrorString(ret) << std::endl;
        throw std::runtime_error("Error writing buffer");
      }

      ret = clSetKernelArg(kernel, i, sizeof(cl_mem), &buffers[i]);
      if (ret != CL_SUCCESS) {
        std::cout << "Error setting kernel arg: " << getErrorString(ret) << std::endl;
        throw std::runtime_error("Error setting kernel arg");
      }
    }

    return chunk;
  }

  std::array<cl_mem, 9> createBuffers(const cl_context &context) {
    std::array<cl_mem, 9> buffers;

    cl_int ret{CL_SUCCESS};

    for (size_t i = 0; i < 9; i++) {
      buffers[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, dimension * dimension * 4 * sizeof(float), NULL, &ret);
      if (ret != CL_SUCCESS) {
        std::cout << "Error creating buffer: " << getErrorString(ret) << std::endl;
        throw std::runtime_error("Error creating buffer");
      }
    }

    return buffers;
  }

  void doGenerationNoSwap(const cl_context &context, const cl_command_queue &queue, const cl_kernel &kernel, const cl_mem &outBuffer, size_t layer = 0) {
    cl_int ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &outBuffer);
    if (ret != CL_SUCCESS) {
      std::cout << "Error setting kernel arg" << std::endl;
      throw std::runtime_error("Error setting kernel arg");
    }

    assert(layer < 2);

    for (auto &chunk : chunks) {
      chunk->generate(context, queue, kernel, outBuffer, layer);
    }
  }

  void writeAllPNGs() {
    std::vector<std::thread> threads;

    for (auto &chunk : chunks) {
      threads.push_back(chunk->dumpPNG());
    }

    for (auto &thread : threads) {
      thread.join();
    }
  }
};


/*
Coordinate system is:

0 -> x
|
v
y

In memory it is y contiguous, i.e. (x_0, y_0), (x_0, y_1), (x_0, y_2)...
*/

int main(int argc, char const *argv[]) {
  // testKernelImage();
  // return 0;

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
  cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating command queue" << std::endl;
    throw std::runtime_error("Error creating command queue");
  }

  // Setting up the generation program and buffers should be somewhere else
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

  cl_kernel nebulaKernel = clCreateKernel(program, "nebula", &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating kernel" << std::endl;
    throw std::runtime_error("Error creating kernel");
  }

  cl_kernel starBackgroundKernel = clCreateKernel(program, "starBackground", &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating kernel" << std::endl;
    throw std::runtime_error("Error creating kernel");
  }

  cl_kernel starForegroundKernel = clCreateKernel(program, "starForeground", &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating kernel" << std::endl;
    throw std::runtime_error("Error creating kernel");
  }

  cl_kernel mixKernel = clCreateKernel(program, "mixLayers", &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating kernel" << std::endl;
    throw std::runtime_error("Error creating kernel");
  }

  cl_kernel clearKernel = clCreateKernel(program, "clear", &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating kernel" << std::endl;
    throw std::runtime_error("Error creating kernel");
  }

  cl_mem outBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 2048 * 2048 * 4 * sizeof(float), NULL, &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating buffer" << std::endl;
    throw std::runtime_error("Error creating buffer");
  }

  GridOfChunks grid(2, 2048);

  std::cout << "Clearing memory" << std::endl;
  for (auto &chunk : grid.chunks) {
    std::cout << "Clearing chunk" << chunk->swapFileName << std::endl;
    chunk->clear(context, command_queue, clearKernel, outBuffer, 0);
  }

  std::cout << "Generating background stars" << std::endl;
  grid.doGenerationNoSwap(context, command_queue, starBackgroundKernel, outBuffer, 1);

  std::cout << "Mixing layers" << std::endl;
  for (auto &chunk : grid.chunks) {
    chunk->mixDown(context, command_queue, mixKernel, outBuffer);
  }

  // load the convolution program
  std::ifstream t2("convolve.ocl");
  std::string source_str2((std::istreambuf_iterator<char>(t2)), std::istreambuf_iterator<char>());

  const char *source2 = source_str2.c_str();
  const size_t sourceSize2 = strlen(source2);

  cl_program program2 = clCreateProgramWithSource(context, 1, (const char **)&source2, &sourceSize2, &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating convolution program" << std::endl;
    throw std::runtime_error("Error creating program");
  }

  ret = clBuildProgram(program2, 1, &device_id, NULL, NULL, NULL);
  if (ret != CL_SUCCESS) {
    std::cout << "Error building convolution program" << std::endl;
    size_t logSize = 10000;
    char *buildLog = new char[logSize + 1];
    buildLog[logSize] = '\0';
    clGetProgramBuildInfo(program2, device_id, CL_PROGRAM_BUILD_LOG, logSize, buildLog, NULL);
    std::cout << buildLog << std::endl;
    delete[] buildLog;
    throw std::runtime_error("Error building program");
  }

  cl_kernel kernel2 = clCreateKernel(program2, "convolve", &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating convolution kernel" << std::endl;
    throw std::runtime_error("Error creating kernel");
  }

  const auto buffers = grid.createBuffers(context);

  Image airyDisk = imageFromGimpExport(airyKernel);
  airyDisk.premultiplyAlpha();
  // const Image airyDisk = testConvKernel();

  ret = clSetKernelArg(kernel2, 9, sizeof(cl_mem), &outBuffer);
  if (ret != CL_SUCCESS) {
    std::cout << "Error setting kernel arg" << std::endl;
    throw std::runtime_error("Error setting kernel arg");
  }

  grid.bindConvolutionKernel(context, command_queue, kernel2, airyDisk, 0.6f);
  for (size_t x = 0; x < grid.cellCount; x++) {
    for (size_t y = 0; y < grid.cellCount; y++) {
      auto chunk = grid.bindForConvolution(context, command_queue, kernel2, x, y, buffers);
      chunk->convolve(context, command_queue, kernel2, outBuffer);
    } 
  }

  grid.convolutionCleanup();

  std::cout << "Generating nebula" << std::endl;
  grid.doGenerationNoSwap(context, command_queue, nebulaKernel, outBuffer, 1);

  std::cout << "Mixing layers" << std::endl;
  for (auto &chunk : grid.chunks) {
    chunk->mixDown(context, command_queue, mixKernel, outBuffer);
  }

  std::cout << "Generating foreground stars" << std::endl;
  grid.doGenerationNoSwap(context, command_queue, starForegroundKernel, outBuffer, 1);

  ret = clSetKernelArg(kernel2, 9, sizeof(cl_mem), &outBuffer);
  if (ret != CL_SUCCESS) {
    std::cout << "Error setting kernel arg" << std::endl;
    throw std::runtime_error("Error setting kernel arg");
  }

  grid.bindConvolutionKernel(context, command_queue, kernel2, airyDisk, 0.4f, 0.8f);
  for (size_t x = 0; x < grid.cellCount; x++) {
    for (size_t y = 0; y < grid.cellCount; y++) {
      auto chunk = grid.bindForConvolution(context, command_queue, kernel2, x, y, buffers, 1);
      chunk->convolve(context, command_queue, kernel2, outBuffer, 1);
    } 
  }

  grid.convolutionCleanup();

  std::cout << "Mixing layers" << std::endl;
  for (auto &chunk : grid.chunks) {
    chunk->mixDown(context, command_queue, mixKernel, outBuffer);
  }

  clReleaseKernel(nebulaKernel);
  clReleaseKernel(starBackgroundKernel);
  clReleaseKernel(starForegroundKernel);
  clReleaseKernel(mixKernel);
  clReleaseKernel(clearKernel);
  
  clReleaseProgram(program);

  clReleaseMemObject(outBuffer);

  for (auto &buf : buffers) {
    clReleaseMemObject(buf);
  }

  // release the kernel
  clReleaseKernel(kernel2);

  // release the program
  clReleaseProgram(program2);

  // release the command queue
  clReleaseCommandQueue(command_queue);

  grid.writeAllPNGs();

  return 0;
}
