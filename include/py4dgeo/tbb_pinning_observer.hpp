#pragma once

#include <tbb/task_arena.h>
#include <tbb/task_scheduler_observer.h>
#include <vector>

#if defined(_WIN32)
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

  void on_scheduler_entry(bool /*worker*/) override
  {
#ifdef ENABLE_TBB_PINNING
    int thread_index = tbb::this_task_arena::current_thread_index();
    if (thread_index < 0 || thread_index >= static_cast<int>(cpu_ids_.size()))
      return;

    int cpu = cpu_ids_[thread_index];
    SetThreadAffinityMask(GetCurrentThread(), 1ULL << cpu);
#endif
  }

private:
#ifdef ENABLE_TBB_PINNING
  std::vector<int> cpu_ids_;
#endif
};
