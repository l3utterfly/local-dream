#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "stb_image.h"
#include "stb_image_resize2.h"

namespace reference_image {
inline constexpr uint64_t kMaxDecodePixels = 4096ull * 4096;
inline constexpr uint64_t kMaxPixels = 1024ull * 1024;
}  // namespace reference_image

inline void decode_reference_image(const std::vector<uint8_t> &image_binary,
                                   std::vector<uint8_t> &output_pixels,
                                   int &output_width, int &output_height) {
  output_pixels.clear();
  output_width = output_height = 0;
  if (image_binary.empty() ||
      image_binary.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("Invalid reference image size");

  const int size = static_cast<int>(image_binary.size());
  int width = 0, height = 0, channels = 0;
  if (!stbi_info_from_memory(image_binary.data(), size, &width, &height,
                             &channels) || width <= 0 || height <= 0)
    throw std::runtime_error("Reference image header is invalid");
  // Inspect the header before stb allocates the full-resolution decoded image.
  if (static_cast<uint64_t>(width) * height > reference_image::kMaxDecodePixels)
    throw std::runtime_error(
        "Reference image exceeds the 16777216-pixel decode limit; resize it first");

  std::unique_ptr<uint8_t, decltype(&stbi_image_free)> decoded(
      stbi_load_from_memory(image_binary.data(), size, &width, &height,
                            &channels, 3),
      stbi_image_free);
  if (!decoded)
    throw std::runtime_error("Reference image decode failed");

  const uint64_t pixels = static_cast<uint64_t>(width) * height;
  int resized_width = width, resized_height = height;
  if (pixels > reference_image::kMaxPixels) {
    const double scale = std::min(
        std::sqrt(static_cast<double>(reference_image::kMaxPixels) / pixels),
        static_cast<double>(reference_image::kMaxPixels) / std::max(width, height));
    resized_width = std::max(1, static_cast<int>(std::floor(width * scale)));
    resized_height = std::max(1, static_cast<int>(std::floor(height * scale)));
    output_pixels.resize(static_cast<size_t>(resized_width) * resized_height * 3);
    if (!stbir_resize_uint8_linear(decoded.get(), width, height, 0,
                                  output_pixels.data(), resized_width,
                                  resized_height, 0, STBIR_RGB)) {
      output_pixels.clear();
      throw std::runtime_error("Failed to resize reference image");
    }
  } else {
    output_pixels.assign(decoded.get(), decoded.get() + pixels * 3);
  }
  output_width = resized_width;
  output_height = resized_height;
}
