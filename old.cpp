#include <stdio.h>
#include <stdlib.h>
#include <cstddef>
#include <string>
#include <span>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <array>
#include <algorithm>
#include <iostream>
#include <ostream>

#include <CL/cl.h>

#include "libs/spng.h"

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
    os << "Pixel(" << (int)pixel.r << ", " << (int)pixel.g << ", " << (int)pixel.b << ", " << (int)pixel.a << ")";
    return os;
  }
}

int testImageWriter() {
  Image image = createTestImage();
  ImageWriter writer;
  writer.writePNG("test.png", image);
  writer.writePNG("test2.png", image.pixels, image.width, image.height);
}

struct Coordinate;

typedef Coordinate Vector;

struct SimplexCoordinate {
  int x;
  int y;
  Vector hashToGradient() const;
  // SimplexCoordinate delta(const SimplexCoordinate &other) const {
  //   return SimplexCoordinate{x - other.x, y - other.y};
  // }
  
};

struct Coordinate {
  float x;
  float y;
  Coordinate mult(float s) const {
    return Coordinate{x * s, y * s};
  }
  const float F = (sqrt(3.0) - 1.0) / 2.0;
  const float G = (1.0 - 1.0 / sqrt(3.0)) / 2.0;
  Coordinate skew() const {
    const float factor = (x + y) * F;
    return (Coordinate{x + factor, y + factor});
  }
  Coordinate unskew() const {
    const float factor = (x + y) * G;
    return (Coordinate{x - factor, y - factor});
  }
  std::array<SimplexCoordinate, 3> subdivision() const {
    const int baseX = floor(x);
    const int baseY = floor(y);
    if (x > y) {
      return std::array<SimplexCoordinate, 3>{
        SimplexCoordinate{baseX + 0, baseY + 0},
        SimplexCoordinate{baseX + 1, baseY + 0},
        SimplexCoordinate{baseX + 1, baseY + 1}
      };
    } else {
      return std::array<SimplexCoordinate, 3>{
        SimplexCoordinate{baseX + 0, baseY + 0},
        SimplexCoordinate{baseX + 0, baseY + 1},
        SimplexCoordinate{baseX + 1, baseY + 1}
      };
    }
  }
  Coordinate dot(const Vector &v) const {
    return Coordinate{x * v.x, y * v.y};
  }
  Coordinate operator+(const Coordinate &other) const {
    return Coordinate{x + other.x, y + other.y};
  }
  Coordinate operator-(const Coordinate &other) const {
    return Coordinate{x - other.x, y - other.y};
  }
  Coordinate(float x, float y) : x(x), y(y) {}
  Coordinate(const SimplexCoordinate &other) : x(other.x), y(other.y) {}
  Coordinate(const Vector &other) : x(other.x), y(other.y) {}
  Coordinate operator=(const Vector &other) {
    x = other.x;
    y = other.y;
    return *this;
  }
  float length2() const {
    return x * x + y * y;
  }
  Coordinate frac() const {
    return Coordinate{x - floor(x), y - floor(y)};
  }
  Coordinate norm() const {
    const float len = sqrt(length2());
    return Coordinate{x / len, y / len};
  }
  Coordinate negate() const {
    return Coordinate{-x, -y};
  }
};

namespace std {
  ostream &operator<<(ostream &os, const Coordinate &coord) {
    os << "(" << coord.x << ", " << coord.y << ")";
    return os;
  }
  ostream &operator<<(ostream &os, const SimplexCoordinate &coord) {
    os << "SimplexCoordinate(" << coord.x << ", " << coord.y << ")";
    return os;
  }
}

Vector SimplexCoordinate::hashToGradient() const {
  // Create random gradients
  const int hash = x * 73856093 ^ y * 19349663;
  const float angle = 2.0f * 3.14159265358979323846f * ((hash >> 8) & 0xff) / 256.0f;
  return Vector{cos(angle), sin(angle)};
}

Vector hash6(const std::array<SimplexCoordinate, 3> &coords) {
  const int hash = coords[0].x * 73856093 ^ coords[0].y * 19349663 ^
                   coords[1].x * 83492791 ^ coords[1].y * 83492791 ^
                   coords[2].x * 83492791 ^ coords[2].y * 83492791;
  const float angle = 2.0f * 3.14159265358979323846f * ((hash >> 8) & 0xff) / 256.0f;
  return Vector{cos(angle), sin(angle)};
}

Vector kernelSum(const Coordinate &coord) {
  const Coordinate skewed = coord.skew();
  const std::array<SimplexCoordinate, 3> sub = skewed.subdivision();

  Vector acm{0.0f, 0.0f};
  const float r2 = 0.6f;

  // return hash6(sub);

  // std::cout << "coord: " << coord << "," << skewed << std::endl;
  // const Coordinate frac = coord.frac();

  for (int i = 0; i < 3; i++) {
    // std::cout << "sub: " << sub[i] << std::endl;
    // std::cout << "unskewed: " << unskewed << std::endl;
    // const Vector dis = skewed - sub[i];
    // const Vector displacement = dis.unskew();
    
    const Coordinate unskewed = Coordinate(sub[i]).unskew();
    const Vector displacement = coord - unskewed;

    // std::cout << "displacement: " << displacement << std::endl;

    const float scalar = powf(std::max(0.0f, r2 - displacement.length2()), 4);

    acm = acm + displacement.dot(sub[i].hashToGradient()).mult(scalar);
  }

  // std::cout << std::endl;

  return acm;

  // for (const auto &coord : sub) {
  //   const float d2 
  //   const float scalar = 
  //   Vector(coord).dot(coord.hashToGradient()).mult(scalar);
  // }
}


Image simplexNoise(int height, int width) {
  Image image;
  image.height = height;
  image.width = width;
  image.pixels.resize(height * width);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      const Coordinate coord = Coordinate{static_cast<float>(x) / 3.0f, static_cast<float>(y) / 3.0f};
      const Vector sum = kernelSum(coord);
      Pixel &pixel = image.pixels[y * width + x];
      pixel.r = std::byte((sum.x + 1.0f) * 127.5f * 30.0f);
      pixel.g = std::byte((sum.y + 1.0f) * 127.5f * 30.0f);
      pixel.b = std::byte(0);
      pixel.a = std::byte(255);
      // std::cout << sum << std::endl;
      // std::cout << std::endl;
    }
  }
  return image;
}

int main(int argc, char **argv) {
  Image image = simplexNoise(512, 512);
  ImageWriter writer;
  writer.writePNG("simplex_noise.png", image);
  return 0;
}

#define MAX_SOURCE_SIZE (0x100000)

int oldMain(int argc, char const *argv[]) {
  // Create the two input vectors
  int i;
  const int LIST_SIZE = 1024;
  int *A = (int *)malloc(sizeof(int) * LIST_SIZE);
  int *B = (int *)malloc(sizeof(int) * LIST_SIZE);
  for (i = 0; i < LIST_SIZE; i++) {
    A[i] = i;
    B[i] = LIST_SIZE - i;
  }

  // Load the kernel source code into the array source_str
  FILE *fp;
  char *source_str;
  size_t source_size;

  fp = fopen("vector_add_kernel.ocl", "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char *)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  // Get platform and device information
  cl_platform_id platform_id = NULL;
  cl_device_id device_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id,
                       &ret_num_devices);

  // Create an OpenCL context
  cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

  // Create a command queue
  cl_command_queue command_queue =
      clCreateCommandQueue(context, device_id, 0, &ret);

  // Create memory buffers on the device for each vector
  cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                    LIST_SIZE * sizeof(int), NULL, &ret);
  cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                    LIST_SIZE * sizeof(int), NULL, &ret);
  cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    LIST_SIZE * sizeof(int), NULL, &ret);

  // Copy the lists A and B to their respective memory buffers
  ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
                             LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
                             LIST_SIZE * sizeof(int), B, 0, NULL, NULL);

  // Create a program from the kernel source
  cl_program program =
      clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                (const size_t *)&source_size, &ret);

  // Build the program
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

  // Create the OpenCL kernel
  cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

  // Set the arguments of the kernel
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);

  // Execute the OpenCL kernel on the list
  size_t global_item_size = LIST_SIZE; // Process the entire lists
  size_t local_item_size = 64;         // Divide work items into groups of 64
  ret =
      clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size,
                             &local_item_size, 0, NULL, NULL);

  // Read the memory buffer C on the device to the local variable C
  int *C = (int *)malloc(sizeof(int) * LIST_SIZE);
  ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
                            LIST_SIZE * sizeof(int), C, 0, NULL, NULL);

  // Display the result to the screen
  for (i = 0; i < LIST_SIZE; i++)
    printf("%d + %d = %d\n", A[i], B[i], C[i]);

  // Clean up
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(a_mem_obj);
  ret = clReleaseMemObject(b_mem_obj);
  ret = clReleaseMemObject(c_mem_obj);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);
  free(A);
  free(B);
  free(C);
  return 0;
}
