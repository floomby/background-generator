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

struct Pixel {
  std::byte r;
  std::byte g;
  std::byte b;
  std::byte a;
};

struct Image {
  std::vector<Pixel> pixels;
  int width;
  int height;
};

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

class ImageWriter {
private:
  spng_ctx *ctx;

public:
  void writePNG(const std::string &filename, const std::vector<Pixel> &pixels, unsigned width, unsigned height) {
    ctx = spng_ctx_new(SPNG_CTX_ENCODER);
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
    data = new float[dimension * dimension * 4];
    std::memset(data, 0, dimension * dimension * 4 * sizeof(float));
    swapFileName = std::to_string(offset[0] / dimension) + "_" + std::to_string(offset[1] / dimension);
  }
  ~Chunk() { delete[] data; }
  std::thread dumpPNG() {
    std::string filename = "chunk" + swapFileName + ".png";
    const auto self = shared_from_this();
    std::thread t([self, filename] {
      std::scoped_lock lock(self->mutex);
      ImageWriter writer;
      Image image = imageFromFloats(self->data, self->dimension, self->dimension);
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
      file.write((char *)data, dimension * dimension * 4 * sizeof(float));
      file.close();
      delete[] data;
    } else {
      std::cout << "Failed to lock mutex (aborting): " << swapFileName << std::endl;
    }
  }

  void swapInFromDisk() {
    data = new float[dimension * dimension * 4];
    std::ifstream file(swapFileName, std::ios::binary);
    file.read((char *)data, dimension * dimension * 4 * sizeof(float));
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

    ret = clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, dimension * dimension * 4 * sizeof(float), data, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
      std::cout << "Error reading buffer: " << getErrorString(ret) << std::endl;
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
  cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
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

  cl_mem outBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 2048 * 2048 * 4 * sizeof(float), NULL, &ret);
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
