#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

namespace py = pybind11;

namespace py4dgeo {

/** Copy-less integration of std::vector<T> and numpy.array
 *
 * Loads of inspiration taken from here:
 * https://github.com/pybind/pybind11/issues/1042
 * https://stackoverflow.com/a/44682603
 *
 */
template<typename T, typename A = std::allocator<T>>
class NumpyVector
{
public:
  using Vector = std::vector<T, A>;

  NumpyVector()
    : vec(new Vector())
    , ownership(true)
  {}

  ~NumpyVector()
  {
    if (ownership)
      delete vec;
  }

  Vector& as_std() { return *vec; }

  const Vector& as_std() const { return *vec; }

  // Let's forbid all the operations that could cause problems
  NumpyVector(const NumpyVector&) = delete;
  NumpyVector(NumpyVector&&) = delete;
  NumpyVector& operator=(const NumpyVector&) = delete;
  NumpyVector& operator=(NumpyVector&&) = delete;

  py::array_t<T> as_numpy()
  {
    // This function can be called exactly once to avoid a double free
    if (!ownership)
      throw std::runtime_error(
        "Trying to transfer ownership of same C++ object to Python twice!");
    ownership = false;

    // The capsule object is managing the deferred deletion of the object
    py::capsule free(
      vec, [](void* v) { delete reinterpret_cast<std::vector<T, A>*>(v); });

    // Constructing a numpy array including the correct descriptors
    return py::array_t<T>({ vec->size() }, { sizeof(T) }, vec->data(), free);
  }

private:
  bool ownership;
  Vector* vec;
};

}