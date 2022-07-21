#include "testsetup.hpp"

#include <py4dgeo/segmentation.hpp>

#include <benchmark/benchmark.h>

using namespace py4dgeo;

static void
changepoint_detection_benchmark(benchmark::State& state)
{
  auto n = state.range(0);
  EigenTimeSeries ts(n);
  for (std::size_t i = n / 2; i < n; ++i)
    ts[i] += 1.0;

  ChangePointDetectionData data{ ts, 24, 12, 1, 1.0 };

  for (auto _ : state) {
    auto cp = change_point_detection(data);
  }
}

BENCHMARK(changepoint_detection_benchmark)
  ->Unit(benchmark::kMicrosecond)
  ->RangeMultiplier(10)
  ->Range(10, 100000);
BENCHMARK_MAIN();
