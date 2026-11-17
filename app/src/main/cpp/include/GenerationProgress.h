#ifndef LOCALDREAM_GENERATION_PROGRESS_H
#define LOCALDREAM_GENERATION_PROGRESS_H
// One progress contract for every image generation backend.
//
// stable-diffusion.cpp drives a single global progress callback from several
// unrelated loops: each sampling pass, every VAE tile, tensor loading and the
// hires upscaler all count from zero with their own totals. Forwarding those
// counters as they come makes a progress bar restart for every phase.
//
// GenerationProgressTracker folds them into one bar that never moves
// backwards. The bar is a slot layout fixed at the start of a generation:
//
//   one slot per VAE encode before sampling (init image, references, control)
//   + every step of every sampling pass
//   + one slot for the hires upscale, when the request has a hires pass
//   + one slot for the final VAE decode
//
// `phase` names the work that is running now, so a caller can label the bar.
// `phase_step`/`phase_total` count inside the phase (sampling steps, or VAE
// tiles when the engine reports them) and are 0/0 when the phase has no
// counter of its own.
//
// Used by PipelineDit (Android DiT) and the sdCpp runtime (Windows, and
// sdcpp.json models on Android) so both emit the same events.

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace localdream {

// Phase names as they reach JavaScript.
inline constexpr const char *kPhasePreparing = "preparing";
inline constexpr const char *kPhaseSampling = "sampling";
inline constexpr const char *kPhaseUpscaling = "upscaling";
inline constexpr const char *kPhaseRefining = "refining";
inline constexpr const char *kPhaseDecoding = "decoding";

struct GenerationProgress {
  int step = 0;
  int total = 1;
  const char *phase = kPhasePreparing;
  int phase_step = 0;
  int phase_total = 0;
};

struct GenerationPlan {
  // VAE encodes that run before the first sampling pass.
  int pre_sample_slots = 0;
  // Expected effective steps of each sampling pass, in order. The tracker
  // replaces an estimate with the engine's real count when the pass starts.
  std::vector<int> pass_steps;
  // Index of the first hires pass in pass_steps, or -1 without hires. Passes
  // from this index on are reported as `refining`, and one upscale slot sits
  // in front of them.
  int refine_from_pass = -1;
};

class GenerationProgressTracker {
 public:
  explicit GenerationProgressTracker(GenerationPlan plan) : plan_(std::move(plan)) {
    plan_.pre_sample_slots = std::max(0, plan_.pre_sample_slots);
    if (plan_.pass_steps.empty()) plan_.pass_steps.push_back(1);
    for (int &steps : plan_.pass_steps) steps = std::max(1, steps);
    if (plan_.refine_from_pass >= static_cast<int>(plan_.pass_steps.size()))
      plan_.refine_from_pass = -1;
  }

  // A sampling pass starts with its effective step count. Starting a pass
  // the plan did not expect appends it (batch or adetail runs).
  GenerationProgress begin_pass(int steps) {
    if (in_pass_) end_pass();
    current_pass_ = next_pass_++;
    if (current_pass_ >= static_cast<int>(plan_.pass_steps.size()))
      plan_.pass_steps.push_back(1);
    if (steps > 0) plan_.pass_steps[current_pass_] = steps;
    in_pass_ = true;
    sampled_ = 0;
    aux_step_ = aux_total_ = 0;
    return current();
  }

  // Completed steps of the current pass. Reaching the pass's last step ends
  // it, because the DiT ABI reports no separate end-of-sampling event.
  GenerationProgress sampled(int step) {
    if (!in_pass_) return current();
    sampled_ = std::max(sampled_, std::clamp(step, 0, pass_steps(current_pass_)));
    if (sampled_ >= pass_steps(current_pass_)) end_pass();
    return current();
  }

  GenerationProgress end_pass() {
    if (in_pass_) {
      in_pass_ = false;
      completed_passes_ = std::max(completed_passes_, current_pass_ + 1);
      aux_step_ = aux_total_ = 0;
    }
    return current();
  }

  // A counter outside sampling (VAE tiles, weight loading, upscaler tiles).
  // It labels the current phase but never moves the bar: the engines do not
  // say which encode/decode it belongs to.
  GenerationProgress auxiliary(int step, int total) {
    if (!in_pass_ && total > 0) {
      aux_step_ = std::clamp(step, 0, total);
      aux_total_ = total;
    }
    return current();
  }

  GenerationProgress current() {
    GenerationProgress progress;
    progress.total = total_slots();
    progress.step = bar_position();
    progress.phase = phase_name();
    if (in_pass_) {
      progress.phase_step = sampled_;
      progress.phase_total = pass_steps(current_pass_);
    } else if (aux_total_ > 0) {
      progress.phase_step = aux_step_;
      progress.phase_total = aux_total_;
    }
    // A pass that turned out longer or shorter than estimated rescales the
    // bar. Hold the previous fraction rather than step it backwards.
    if (last_total_ > 0 &&
        static_cast<int64_t>(progress.step) * last_total_ <
            static_cast<int64_t>(last_step_) * progress.total) {
      progress.step = last_step_;
      progress.total = last_total_;
    }
    last_step_ = progress.step;
    last_total_ = progress.total;
    return progress;
  }

 private:
  int passes() const { return static_cast<int>(plan_.pass_steps.size()); }
  int pass_steps(int pass) const { return plan_.pass_steps[pass]; }
  bool has_hires() const { return plan_.refine_from_pass >= 0; }

  int total_slots() const {
    int total = plan_.pre_sample_slots + 1;
    for (int steps : plan_.pass_steps) total += steps;
    return total + (has_hires() ? 1 : 0);
  }

  // Slots done before the given pass starts sampling.
  int slots_before_pass(int pass) const {
    int slots = plan_.pre_sample_slots;
    for (int i = 0; i < pass && i < passes(); ++i) slots += pass_steps(i);
    if (has_hires() && pass >= plan_.refine_from_pass) ++slots;
    return slots;
  }

  int bar_position() const {
    const int total = total_slots();
    if (in_pass_) return slots_before_pass(current_pass_) + sampled_;
    if (completed_passes_ == 0) return 0;
    // Between passes or after the last one: everything up to the next pass
    // is done except the upscale slot, which is the work now running.
    if (completed_passes_ >= passes()) return total - 1;
    int slots = slots_before_pass(completed_passes_);
    if (has_hires() && completed_passes_ == plan_.refine_from_pass) --slots;
    return slots;
  }

  const char *phase_name() const {
    if (in_pass_) {
      return has_hires() && current_pass_ >= plan_.refine_from_pass ? kPhaseRefining
                                                                    : kPhaseSampling;
    }
    if (completed_passes_ >= passes()) return kPhaseDecoding;
    if (completed_passes_ == 0) return kPhasePreparing;
    if (has_hires() && completed_passes_ == plan_.refine_from_pass) return kPhaseUpscaling;
    return has_hires() && completed_passes_ > plan_.refine_from_pass ? kPhaseRefining
                                                                     : kPhaseSampling;
  }

  GenerationPlan plan_;
  bool in_pass_ = false;
  int current_pass_ = -1;
  int next_pass_ = 0;
  int completed_passes_ = 0;
  int sampled_ = 0;
  int aux_step_ = 0;
  int aux_total_ = 0;
  int last_step_ = 0;
  int last_total_ = 0;
};

}  // namespace localdream

#endif  // LOCALDREAM_GENERATION_PROGRESS_H
