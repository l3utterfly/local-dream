// HTP memory budget for a DiT context (stable-diffusion.cpp's max_vram).
//
// A budget makes the graph cut split the model into segments and stream the
// ones that do not fit, re-reading and repacking their weights every step, so
// it is only worth paying when keeping the model whole could get the app
// reclaimed. The thresholds follow nightmare-mobile's measured policy
// (BackendProcess.kt: kreaBudget, importBudget, lowArchDitBudget).
#pragma once

#include <cstdint>
#include <string>

namespace dit_budget {

constexpr uint64_t kGB = 1000ull * 1000ull * 1000ull;

// Free memory needed on top of the largest HTP component to keep it whole:
// its compute buffers plus a floor for Android and the UI.
constexpr uint64_t kWholeHeadroom = 2200ull * 1000ull * 1000ull;
// Below this much free memory a 5 GiB budget left too little (S25, Krea 2).
constexpr uint64_t kFiveGiBFree = 6100ull * 1000ull * 1000ull;
// Above this the "Klein" DiT is the 9B one, whose blocks need a lower cap.
constexpr uint64_t kLargeKleinDit = 5 * kGB;

struct Inputs {
  int htp_arch = 0;               // e.g. 79; 0 when unknown
  uint64_t mem_available = 0;     // MemAvailable at context creation; 0 when unknown
  uint64_t largest_component = 0; // largest of the DiT and text-encoder files; 0 when unknown
  uint64_t dit_bytes = 0;         // the DiT file alone; 0 when unknown
  bool klein = false;
};

// GiB budget for HTP0 as a max_vram value, or "" to keep every component whole.
inline std::string htp_budget_gib(const Inputs &in) {
  if (in.htp_arch > 0 && in.htp_arch < 79) {
    // One FastRPC session maps only ~3.1 GB below v79 (less on v73), and the
    // budget covers compute buffers as well as weights.
    double gib = in.klein && in.dit_bytes > kLargeKleinDit ? 2.25 : 2.75;
    if (in.htp_arch < 75) gib -= 0.75;
    std::string text = std::to_string(gib);
    text.erase(text.find_last_not_of('0') + 1);
    if (!text.empty() && text.back() == '.') text.pop_back();
    return text;
  }
  if (in.mem_available == 0 || in.largest_component == 0) {
    return "5";
  }
  if (in.mem_available >= in.largest_component + kWholeHeadroom) {
    return "";
  }
  return in.mem_available >= kFiveGiBFree ? "5" : "4";
}

} // namespace dit_budget
