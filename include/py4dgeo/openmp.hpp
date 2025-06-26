#pragma once

#ifdef PY4DGEO_WITH_OPENMP
#include <omp.h>
#endif

#include <exception>
#include <utility>

/** @brief A container to handle exceptions in parallel regions
 *
 *  OpenMP does have limited support for C++ exceptions in parallel
 *  regions: Exceptions need to be catched on the same thread they
 *  have been thrown on. This class allows to store the first thrown
 *  exception to then rethrow it after we left the parallel region.
 *  This is a necessary construct to propagate exceptions from Python
 *  callbacks through the multithreaded C++ layer back to the calling
 *  Python scope. Inspiration is taken from:
 *  https://stackoverflow.com/questions/11828539/elegant-exceptionhandling-in-openmp
 */
class CallbackExceptionVault
{
  std::exception_ptr ptr = nullptr;

public:
  template<typename Function, typename... Parameters>
  void run(Function&& f, Parameters&&... parameters)
  {
    try {
      std::forward<Function>(f)(std::forward<Parameters>(parameters)...);
    } catch (...) {
#ifdef PY4DGEO_WITH_OPENMP
#pragma omp critical
#endif
      {
        if (!this->ptr)
          this->ptr = std::current_exception();
      }
    }
  }

  void rethrow() const
  {
    if (this->ptr)
      std::rethrow_exception(this->ptr);
  }
};
