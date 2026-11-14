// HTP memory budget for a DiT context (stable-diffusion.cpp's max_vram).
//
// A budget makes the graph cut split the model into segments and stream the
// ones that do not fit, re-reading and repacking their weights every step, so
// RAM headroom and DSP mapping capacity are independent constraints.
#pragma once

#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace dit_budget {

constexpr uint64_t kMiB = 1024ull * 1024ull;
constexpr uint64_t kGiB = 1024ull * kMiB;
// Leave room for Android, the UI and allocations outside HTP accounting.
constexpr uint64_t kReservedHeadroom = 2200ull * 1000ull * 1000ull;

struct Inputs {
  int htp_arch = 0;               // e.g. 79; 0 when unknown
  std::optional<uint64_t> mem_available; // MemAvailable at context creation
  bool klein = false;
};

inline uint64_t fallback_budget_bytes(const Inputs &in) {
  return in.klein ? 9 * kGiB / 4 : 5 * kGiB;
}

inline uint64_t architecture_cap_bytes(const Inputs &in) {
  if (in.htp_arch > 0 && in.htp_arch < 79) {
    // One FastRPC session maps only ~3.1 GB below v79 (less on v73), and the
    // budget covers compute buffers as well as weights.
    // Both Klein sizes need headroom for reference tokens and large contiguous
    // single-stream activations. Checkpoint size does not bound this live set.
    uint64_t cap = (in.klein ? 9 : 11) * kGiB / 4;
    if (in.htp_arch < 75) cap -= 3 * kGiB / 4;
    return cap;
  }
  // Preserve a bounded fallback if architecture detection is unavailable.
  return in.htp_arch <= 0 ? fallback_budget_bytes(in) : std::numeric_limits<uint64_t>::max();
}

inline uint64_t htp_budget_bytes(const Inputs &in) {
  if (!in.mem_available) {
    return std::min(fallback_budget_bytes(in), architecture_cap_bytes(in));
  }
  if (*in.mem_available <= kReservedHeadroom ||
      *in.mem_available - kReservedHeadroom < kMiB)
    throw std::runtime_error("Insufficient available RAM after reserving HTP headroom");
  // Round down so GiB serialization never spends the reserved headroom.
  return std::min(*in.mem_available - kReservedHeadroom, architecture_cap_bytes(in)) / kMiB * kMiB;
}

inline std::string htp_budget_gib(const Inputs &in) {
  std::ostringstream text;
  text.imbue(std::locale::classic());
  text << std::fixed << std::setprecision(10)
       << static_cast<double>(htp_budget_bytes(in)) / kGiB;
  return text.str();
}

} // namespace dit_budget
