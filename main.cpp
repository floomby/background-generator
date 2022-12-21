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
#include <filesystem>
#include <optional>

#include <CL/cl.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;
namespace pt = boost::property_tree;

#include "libs/spng.h"

#include "airyKernel.h"

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

unsigned roundUpToMultipleOf(unsigned value, unsigned multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

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

Image imageFromFloats(float *data, size_t width, size_t height) {
  Image image;
  image.width = width;
  image.height = height;

  image.pixels.resize(width * height);
  for (size_t i = 0; i < image.pixels.size(); i++) {
    // clamp to 0..1
    data[i * 4] = std::max(0.0f, std::min(1.0f, data[i * 4]));
    data[i * 4 + 1] = std::max(0.0f, std::min(1.0f, data[i * 4 + 1]));
    data[i * 4 + 2] = std::max(0.0f, std::min(1.0f, data[i * 4 + 2]));
    data[i * 4 + 3] = std::max(0.0f, std::min(1.0f, data[i * 4 + 3]));
    image.pixels[i] = {
      .r = std::byte(data[i * 4] * 255.0f),
      .g = std::byte(data[i * 4 + 1] * 255.0f),
      .b = std::byte(data[i * 4 + 2] * 255.0f),
      .a = std::byte(data[i * 4 + 3] * 255.0f)
    };
  }

  return image;
}

template <size_t N>
Image imageFromGimpExport(const GimpExport<N> &img) {
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

namespace std {
  ostream &operator<<(ostream &os, const Pixel &pixel) {
    os << "Pixel(" << (int)pixel.r << ", " << (int)pixel.g << ", " << (int)pixel.b
      << ", " << (int)pixel.a << ")";
    return os;
  }
} // namespace std


class Chunk : public std::enable_shared_from_this<Chunk> {
  float *data = nullptr;
  size_t dimension;
  size_t offset[2];
  std::string chunkName;
public:
  Chunk(size_t dimension, std::array<size_t, 2> offset) {
    assert(offset[0] % dimension == 0);
    assert(offset[1] % dimension == 0);
    this->dimension = dimension;
    this->offset[0] = offset[0];
    this->offset[1] = offset[1];
    chunkName = std::to_string(offset[0] / dimension) + "_" + std::to_string(offset[1] / dimension);
    data = new float[dimension * dimension * 4];
  }
  ~Chunk() {
    if (data != nullptr) {
      delete[] data;
    }
  }
  std::thread dumpPNG(const std::string &outDir) {
    std::string filename = outDir + "/chunk" + chunkName + ".png";
    const auto self = shared_from_this();
    std::thread t([self, filename] {
      Image image = imageFromFloats(self->data, self->dimension, self->dimension);
      ImageWriter::writePNG(filename, image);
      std::stringstream ss;
      ss << "Wrote " << filename << std::endl;
      std::cout << ss.str();
      delete[] self->data;
      self->data = nullptr;
    });
    return t;
  }
  Chunk(const Chunk &) = delete;
  Chunk &operator=(const Chunk &) = delete;
  Chunk(Chunk &&) = delete;
  Chunk &operator=(Chunk &&) = delete;

  void generate(
    const cl_context &context,
    const cl_command_queue &queue,
    const cl_kernel &starKern,
    const cl_kernel &finKern,
    const cl_mem &outBuffer,
    size_t idOffset
  ) {
#ifdef VERBOSE_OPS
    std::cout << "Generating " << chunkName << std::endl;
#endif

    cl_int ret{CL_SUCCESS};

    size_t globalWorkSize[2] = {dimension + idOffset * 2, dimension + idOffset * 2};
    size_t localWorkSize[2] = {32, 32};

    ret = clEnqueueNDRangeKernel(queue, starKern, 2, this->offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
      std::cout << "Error enqueuing kernel: " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error enqueuing kernel");
    }

    ret = clEnqueueNDRangeKernel(queue, finKern, 2, this->offset, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
      std::cout << "Error enqueuing kernel: " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error enqueuing kernel");
    }

    ret = clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, dimension * dimension * 4 * sizeof(float), data, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
      std::cout << "Error reading buffer: " << getErrorString(ret) << std::endl;
      throw std::runtime_error("Error reading buffer");
    }
  }
};

enum FeatureKind {
  GlobularCluster = 0,
  Nebula = 1,
};

struct Feature {
  float x;
  float y;
  float radius;
  float strength;
  FeatureKind kind;
  std::optional<std::array<float, 3>> color;
};

FeatureKind getFeatureKind(const std::string &kind) {
  if (kind == "GlobularCluster") {
    return GlobularCluster;
  } else if (kind == "Nebula") {
    return Nebula;
  } else {
    throw std::runtime_error("Unknown feature kind");
  }
}

std::string getFeatureKindString(FeatureKind kind) {
  switch (kind) {
    case GlobularCluster:
      return "GlobularCluster";
    case Nebula:
      return "Nebula";
    default:
      throw std::runtime_error("Unknown feature kind");
  }
}

namespace std {
ostream &operator<<(ostream &os, const Feature &f) {
  os << "Feature { x: " << f.x << ", y: " << f.y << ", radius: " << f.radius << ", strength: " << f.strength << ", kind: " << getFeatureKindString(f.kind);
  if (f.color) {
    os << ", color: " << f.color.value()[0] << ", " << f.color.value()[1] << ", " << f.color.value()[2];
  }
  os << " }";
  return os;
}
}

// read in the features from the json feature file
std::vector<Feature> readFeatures(const std::string &fileName) {
  pt::ptree tree;
  pt::read_json(fileName, tree);
  // root of the json file is an array of features
  std::vector<Feature> features;

  for (auto &feature : tree) {
    Feature f;
    f.x = feature.second.get<float>("x");
    f.y = feature.second.get<float>("y");
    f.radius = feature.second.get<float>("radius");
    f.strength = feature.second.get<float>("strength");
    f.kind = getFeatureKind(feature.second.get<std::string>("kind"));
    if (f.kind == Nebula) {
      f.color = std::array<float, 3>{
        feature.second.get<float>("color.r"),
        feature.second.get<float>("color.g"),
        feature.second.get<float>("color.b")
      };
    } else {
      f.color = std::nullopt;
      // Warn if color is specified for a globular cluster
      if (feature.second.count("color")) {
        std::cout << "Warning: color specified for globular cluster" << std::endl;
      }
    }
    features.push_back(f);
  }

  return features;
}

void ensureDirectoryExists(const std::string &dirName) {
  // Make the directory if it doesn't exist
  std::filesystem::path dir(dirName);
  if (!std::filesystem::exists(dir)) {
    std::filesystem::create_directory(dir);
  }
}

cl_mem setupGlobularClusters(const cl_context &context, const cl_command_queue &queue, const cl_kernel &kernel, const std::vector<Feature> &features) {
  cl_int ret{CL_SUCCESS};

  // Filter out globular clusters
  std::vector<Feature> globularClusters;
  for (auto &feature : features) {
    if (feature.kind == GlobularCluster) {
      globularClusters.push_back(feature);
    }
  }

  // Create a buffer for the globular clusters
  cl_mem globularClusterBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, globularClusters.size() * sizeof(Feature), NULL, &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating globular cluster buffer: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error creating globular cluster buffer");
  }

  // Copy the globular clusters to the buffer
  float *buf = new float[globularClusters.size() * sizeof(float) * 4];
  for (size_t i = 0; i < globularClusters.size(); i++) {
    buf[i * 4] = globularClusters[i].x;
    buf[i * 4 + 1] = globularClusters[i].y;
    buf[i * 4 + 2] = globularClusters[i].radius;
    buf[i * 4 + 3] = globularClusters[i].strength;
  }

  ret = clEnqueueWriteBuffer(queue, globularClusterBuffer, CL_TRUE, 0, globularClusters.size() * sizeof(Feature), buf, 0, NULL, NULL);
  if (ret != CL_SUCCESS) {
    std::cout << "Error writing globular cluster buffer: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error writing globular cluster buffer");
  }

  delete[] buf;

  // Set the globular cluster buffer as an argument to the kernel
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &globularClusterBuffer);
  if (ret != CL_SUCCESS) {
    std::cout << "Error setting globular cluster buffer as kernel argument: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error setting globular cluster buffer as kernel argument");
  }

  int globularClusterCount = globularClusters.size();

  ret = clSetKernelArg(kernel, 1, sizeof(int), &globularClusterCount);
  if (ret != CL_SUCCESS) {
    std::cout << "Error setting globular cluster count as kernel argument: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error setting globular cluster count as kernel argument");
  }

  return globularClusterBuffer;
}

std::array<cl_mem, 2> setupNebulas(const cl_context &context, const cl_command_queue &queue, const cl_kernel &kernel, const std::vector<Feature> &features) {
  cl_int ret{CL_SUCCESS};

  // Filter out nebulae
  std::vector<Feature> nebulae;
  for (auto &feature : features) {
    if (feature.kind == Nebula) {
      nebulae.push_back(feature);
    }
  }

  // Create a buffer for the nebulae
  cl_mem nebulaBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, nebulae.size() * sizeof(Feature), NULL, &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating nebula buffer: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error creating nebula buffer");
  }

  // Copy the nebulae to the buffer
  float *buf = new float[nebulae.size() * sizeof(float) * 4];
  for (size_t i = 0; i < nebulae.size(); i++) {
    buf[i * 4] = nebulae[i].x;
    buf[i * 4 + 1] = nebulae[i].y;
    buf[i * 4 + 2] = nebulae[i].radius;
    buf[i * 4 + 3] = nebulae[i].strength;
  }

  ret = clEnqueueWriteBuffer(queue, nebulaBuffer, CL_TRUE, 0, nebulae.size() * sizeof(Feature), buf, 0, NULL, NULL);
  if (ret != CL_SUCCESS) {
    std::cout << "Error writing nebula buffer: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error writing nebula buffer");
  }

  delete[] buf;

  // Set the nebula buffer as an argument to the kernel
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &nebulaBuffer);
  if (ret != CL_SUCCESS) {
    std::cout << "Error setting nebula buffer as kernel argument: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error setting nebula buffer as kernel argument");
  }

  // Create a buffer for the nebulae colors
  cl_mem nebulaColorBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, nebulae.size() * sizeof(cl_float4), NULL, &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating nebula color buffer: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error creating nebula color buffer");
  }

  // Copy the nebulae colors to the buffer
  float *colorBuf = new float[nebulae.size() * sizeof(float) * 4];
  for (size_t i = 0; i < nebulae.size(); i++) {
    colorBuf[i * 4] = nebulae[i].color->at(0);
    colorBuf[i * 4 + 1] = nebulae[i].color->at(1);
    colorBuf[i * 4 + 2] = nebulae[i].color->at(2);
    colorBuf[i * 4 + 3] = 0.0f;
  }

  ret = clEnqueueWriteBuffer(queue, nebulaColorBuffer, CL_TRUE, 0, nebulae.size() * sizeof(cl_float4), colorBuf, 0, NULL, NULL);
  if (ret != CL_SUCCESS) {
    std::cout << "Error writing nebula color buffer: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error writing nebula color buffer");
  }

  delete[] colorBuf;

  // Set the nebula color buffer as an argument to the kernel
  ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &nebulaColorBuffer);
  if (ret != CL_SUCCESS) {
    std::cout << "Error setting nebula color buffer as kernel argument: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error setting nebula color buffer as kernel argument");
  }

  int nebulaCount = nebulae.size();

  ret = clSetKernelArg(kernel, 4, sizeof(int), &nebulaCount);
  if (ret != CL_SUCCESS) {
    std::cout << "Error setting nebula count as kernel argument: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error setting nebula count as kernel argument");
  }

  return {nebulaBuffer, nebulaColorBuffer};
}

cl_mem bindConvolutionKernel(const cl_context &context, const cl_command_queue &queue, const cl_kernel &kernel, const Image &kern, unsigned int argIndex) {
  cl_int ret{CL_SUCCESS};

  assert(kern.height == kern.width);
  assert(kern.height % 2 == 1);
  cl_mem kernBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, kern.width * kern.height * sizeof(float) * 4, NULL, &ret);
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

  ret = clSetKernelArg(kernel, argIndex, sizeof(cl_mem), &kernBuffer);
  if (ret != CL_SUCCESS) {
    std::cout << "Error setting kernel argument: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error setting kernel argument");
  }

  delete[] tmpBuf;

  int kernSize = kern.width >> 1;
  
#ifdef VERBOSE_OPS
  std::cout << "Convolution kernel size: " << kernSize << " (" << kern.width << "x" << kern.height << ")" << std::endl;
#endif

  ret = clSetKernelArg(kernel, argIndex + 1, sizeof(cl_uint), &kernSize);
  if (ret != CL_SUCCESS) {
    std::cout << "Error setting kernel argument: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error setting kernel argument");
  }

  auto sums = kern.integrate();

  ret = clSetKernelArg(kernel, argIndex + 2, sizeof(cl_float) * 4, sums.data());
  if (ret != CL_SUCCESS) {
    std::cout << "Error setting kernel argument: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error setting kernel argument");
  }

  return kernBuffer;
}

int main(int argc, char const *argv[]) {
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "Print help message")
    ("chunkCount", po::value<unsigned>()->default_value(2), "Number of chunks per side to generate (total chunks generated is chunkCount^2)")
    ("chunkDimension", po::value<unsigned>()->default_value(2048), "Size of chunks to generate")
    ("outputDirectory", po::value<std::string>()->default_value("output"), "Directory to output the pngs")
    ("featureFile", po::value<std::string>()->default_value("features.json"), "File containing the features to generate")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  const auto outDir = vm["outputDirectory"].as<std::string>();

  ensureDirectoryExists(outDir);

  unsigned chunkCount = vm["chunkCount"].as<unsigned>();
  unsigned chunkDimension = vm["chunkDimension"].as<unsigned>();

  auto features = readFeatures(vm["featureFile"].as<std::string>());

  // Print the features
  std::cout << "Using the following features:" << std::endl;
  for (auto &feature : features) {
    std::cout << feature << std::endl;
  }

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
  cl_command_queue commandQueue = clCreateCommandQueue(context, device_id, 0, &ret);
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

  cl_kernel finalizeKernel = clCreateKernel(program, "finalize", &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating kernel" << std::endl;
    throw std::runtime_error("Error creating kernel");
  }

  cl_kernel stargenKernel = clCreateKernel(program, "stargen", &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating kernel" << std::endl;
    throw std::runtime_error("Error creating kernel");
  }

  cl_mem outBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, chunkDimension * chunkDimension * 4 * sizeof(float), NULL, &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating buffer" << std::endl;
    throw std::runtime_error("Error creating buffer");
  }

  ret = clSetKernelArg(finalizeKernel, 0, sizeof(cl_mem), &outBuffer);
  if (ret != CL_SUCCESS) {
    std::cout << "Error setting kernel argument: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error setting kernel arg");
  }

  auto starBuf = setupGlobularClusters(context, commandQueue, stargenKernel, features);
  auto nebulaBuf = setupNebulas(context, commandQueue, stargenKernel, features);

  Image airyDisk = imageFromGimpExport(airyKernel);
  airyDisk.premultiplyAlpha();

  Image airyDisk2 = imageFromGimpExport(airyKernel2);
  airyDisk2.premultiplyAlpha();

  cl_mem backgroundBuf = bindConvolutionKernel(context, commandQueue, finalizeKernel, airyDisk, 1);
  cl_mem foregroundBuf = bindConvolutionKernel(context, commandQueue, finalizeKernel, airyDisk2, 4);
  cl_mem nebulaConBuf = bindConvolutionKernel(context, commandQueue, finalizeKernel, airyDisk, 7);

  unsigned maxKernelOffset = roundUpToMultipleOf(std::max(airyDisk.width, airyDisk2.height) >> 1, 16);
  size_t scratchBufferSize = (chunkDimension + maxKernelOffset * 2) * (chunkDimension + maxKernelOffset * 2) * 4 * sizeof(float);

  cl_mem scratchBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, scratchBufferSize, NULL, &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating buffer" << std::endl;
    throw std::runtime_error("Error creating buffer");
  }

  cl_mem scratchBuffer2 = clCreateBuffer(context, CL_MEM_READ_WRITE, scratchBufferSize, NULL, &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating buffer" << std::endl;
    throw std::runtime_error("Error creating buffer");
  }

  cl_mem scratchBuffer3 = clCreateBuffer(context, CL_MEM_READ_WRITE, scratchBufferSize, NULL, &ret);
  if (ret != CL_SUCCESS) {
    std::cout << "Error creating buffer" << std::endl;
    throw std::runtime_error("Error creating buffer");
  }

  ret = clSetKernelArg(finalizeKernel, 10, sizeof(cl_mem), &scratchBuffer);
  if (ret != CL_SUCCESS) {
    std::cout << "Error setting kernel argument: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error setting kernel arg");
  }

  ret = clSetKernelArg(finalizeKernel, 11, sizeof(unsigned), &maxKernelOffset);
  if (ret != CL_SUCCESS) {
    std::cout << "Error setting kernel argument: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error setting kernel arg");
  }

  ret = clSetKernelArg(finalizeKernel, 12, sizeof(cl_mem), &scratchBuffer2);
  if (ret != CL_SUCCESS) {
    std::cout << "Error setting kernel argument: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error setting kernel arg");
  }

  ret = clSetKernelArg(finalizeKernel, 13, sizeof(cl_mem), &scratchBuffer3);
  if (ret != CL_SUCCESS) {
    std::cout << "Error setting kernel argument: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error setting kernel arg");
  }

  ret = clSetKernelArg(stargenKernel, 5, sizeof(cl_mem), &scratchBuffer);
  if (ret != CL_SUCCESS) {
    std::cout << "Error setting kernel argument: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error setting kernel arg");
  }

  ret = clSetKernelArg(stargenKernel, 6, sizeof(unsigned), &maxKernelOffset);
  if (ret != CL_SUCCESS) {
    std::cout << "Error setting kernel argument: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error setting kernel arg");
  }

  ret = clSetKernelArg(stargenKernel, 7, sizeof(cl_mem), &scratchBuffer2);
  if (ret != CL_SUCCESS) {
    std::cout << "Error setting kernel argument: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error setting kernel arg");
  }

  ret = clSetKernelArg(stargenKernel, 8, sizeof(cl_mem), &scratchBuffer3);
  if (ret != CL_SUCCESS) {
    std::cout << "Error setting kernel argument: " << getErrorString(ret) << std::endl;
    throw std::runtime_error("Error setting kernel arg");
  }

  std::vector<std::thread> pngThreads;

  for (size_t x = 0; x < chunkCount; x++) {
    for (size_t y = 0; y < chunkCount; y++) {
      auto chunk = std::make_shared<Chunk>(chunkDimension, std::array<size_t, 2>({x * chunkDimension, y * chunkDimension}));
      chunk->generate(context, commandQueue, stargenKernel, finalizeKernel, outBuffer, maxKernelOffset);
      pngThreads.push_back(chunk->dumpPNG(outDir));
    }
  }

  clReleaseMemObject(backgroundBuf);
  clReleaseMemObject(foregroundBuf);
  clReleaseMemObject(nebulaConBuf);

  clReleaseMemObject(starBuf);
  clReleaseMemObject(nebulaBuf[0]);
  clReleaseMemObject(nebulaBuf[1]);

  clReleaseMemObject(scratchBuffer);
  clReleaseMemObject(scratchBuffer2);
  clReleaseMemObject(scratchBuffer3);

  clReleaseKernel(stargenKernel);
  clReleaseKernel(finalizeKernel);
  clReleaseProgram(program);
  clReleaseMemObject(outBuffer);
  clReleaseCommandQueue(commandQueue);
  clReleaseContext(context);

  for (auto &thread : pngThreads) {
    thread.join();
  }

  return 0;
}
