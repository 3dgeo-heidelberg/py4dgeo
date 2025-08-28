#pragma once

#include <tbb/task_arena.h>
#include <tbb/task_scheduler_observer.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#define ENABLE_TBB_PINNING
#endif

/// A TBB task scheduler observer that optionally pins threads to specific CPU
/// cores
class PinningObserver : public tbb::task_scheduler_observer
{
public:
  PinningObserver(tbb::task_arena& arena, const std::vector<int>& cpu_ids)
    : tbb::task_scheduler_observer(arena)
  {
#ifdef ENABLE_TBB_PINNING
    cpu_ids_ = cpu_ids;
    observe(true);
#endif
  }

  ~PinningObserver()
  {
#ifdef ENABLE_TBB_PINNING
    observe(false);
#endif
  }

  void on_scheduler_entry(bool /*worker*/) override
  {
#ifdef ENABLE_TBB_PINNING
    const int thread_index = tbb::this_task_arena::current_thread_index();
    if (thread_index < 0 || thread_index >= static_cast<int>(cpu_ids_.size()))
      return;

    const int requested_cpu = cpu_ids_[thread_index];

    // Basic guard for classic affinity masks (single group only).
    if (requested_cpu < 0 ||
        requested_cpu >= static_cast<int>(sizeof(DWORD_PTR) * 8)) {
      std::printf(
        "[TBB] Thread %2d: requested CPU %d out of mask range (%zu bits)\n",
        thread_index,
        requested_cpu,
        sizeof(DWORD_PTR) * 8);
      std::fflush(stdout);
      return;
    }

    HANDLE thread = GetCurrentThread();

    const DWORD_PTR mask =
      (static_cast<DWORD_PTR>(1) << static_cast<unsigned>(requested_cpu));
    const DWORD_PTR prev = SetThreadAffinityMask(thread, mask);
    if (prev == 0) {
      const DWORD err = GetLastError();
      std::printf(
        "[TBB] Thread %2d: SetThreadAffinityMask failed (CPU %d), error=%lu\n",
        thread_index,
        requested_cpu,
        static_cast<unsigned long>(err));
      std::fflush(stdout);
      return;
    }

    const DWORD actual_cpu =
      GetCurrentProcessorNumber(); // index within current group
    std::printf("[TBB] Thread %2d pinned to CPU %2d, running on CPU %2lu\n",
                thread_index,
                requested_cpu,
                static_cast<unsigned long>(actual_cpu));
    std::fflush(stdout);
#endif
  }

private:
#ifdef ENABLE_TBB_PINNING
  std::vector<int> cpu_ids_;
#endif
};
