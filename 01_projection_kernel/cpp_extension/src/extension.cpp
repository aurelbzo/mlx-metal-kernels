#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/fast.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

mx::array nearest_surface_projection(
    const mx::array& points,
    const mx::array& sdf_values,
    const mx::array& sdf_grads,
    mx::StreamOrDevice stream = {}) {
  if (points.ndim() != 2 || points.shape(1) != 3 || points.shape(0) == 0) {
    throw std::invalid_argument("points must have nonempty shape (N, 3)");
  }
  if (sdf_values.ndim() != 2 || sdf_values.shape(0) != points.shape(0) ||
      sdf_values.shape(1) == 0) {
    throw std::invalid_argument("sdf_values must have shape (N, M), with M > 0");
  }
  if (sdf_grads.ndim() != 3 || sdf_grads.shape(0) != points.shape(0) ||
      sdf_grads.shape(1) != sdf_values.shape(1) || sdf_grads.shape(2) != 3) {
    throw std::invalid_argument("sdf_grads must have shape (N, M, 3)");
  }
  if (points.dtype() != mx::float32 || sdf_values.dtype() != mx::float32 ||
      sdf_grads.dtype() != mx::float32) {
    throw std::invalid_argument("all inputs must have dtype float32");
  }

  const int rows = points.shape(0);
  static const auto kernel = mx::fast::metal_kernel(
      "nearest_surface_projection",
      {"points", "sdf_values", "sdf_grads"},
      {"projected"},
      R"metal(
        uint i = thread_position_in_grid.x;
        uint N = points_shape[0];
        if (i >= N) {
          return;
        }

        uint M = sdf_values_shape[1];
        float best_abs = INFINITY;
        uint best_idx = 0;
        for (uint m = 0; m < M; ++m) {
          float value = sdf_values[i * M + m];
          float abs_value = fabs(value);
          if (abs_value < best_abs) {
            best_abs = abs_value;
            best_idx = m;
          }
        }

        float distance = sdf_values[i * M + best_idx];
        for (uint axis = 0; axis < 3; ++axis) {
          float point = points[i * 3 + axis];
          float gradient = sdf_grads[(i * M + best_idx) * 3 + axis];
          projected[i * 3 + axis] = point - distance * gradient;
        }
      )metal");

  const int threads = std::min(rows, 256);
  return kernel(
             {points, sdf_values, sdf_grads},
             {points.shape()},
             {mx::float32},
             {rows, 1, 1},
             {threads, 1, 1},
             {},
             std::nullopt,
             false,
             stream)
      .front();
}

std::vector<mx::array> nearest_surface_projection_vjp(
    const mx::array& points,
    const mx::array& sdf_values,
    const mx::array& sdf_grads,
    const mx::array& cotangent,
    mx::StreamOrDevice stream = {}) {
  if (points.ndim() != 2 || points.shape(1) != 3 || points.shape(0) == 0) {
    throw std::invalid_argument("points must have nonempty shape (N, 3)");
  }
  if (sdf_values.ndim() != 2 || sdf_values.shape(0) != points.shape(0) ||
      sdf_values.shape(1) == 0) {
    throw std::invalid_argument("sdf_values must have shape (N, M), with M > 0");
  }
  if (sdf_grads.ndim() != 3 || sdf_grads.shape(0) != points.shape(0) ||
      sdf_grads.shape(1) != sdf_values.shape(1) || sdf_grads.shape(2) != 3) {
    throw std::invalid_argument("sdf_grads must have shape (N, M, 3)");
  }
  if (cotangent.ndim() != 2 || cotangent.shape(0) != points.shape(0) ||
      cotangent.shape(1) != 3) {
    throw std::invalid_argument("cotangent must have shape (N, 3)");
  }
  if (points.dtype() != mx::float32 || sdf_values.dtype() != mx::float32 ||
      sdf_grads.dtype() != mx::float32 || cotangent.dtype() != mx::float32) {
    throw std::invalid_argument("all inputs must have dtype float32");
  }

  const int rows = points.shape(0);
  static const auto kernel = mx::fast::metal_kernel(
      "nearest_surface_projection_vjp",
      {"points", "sdf_values", "sdf_grads", "cotangent"},
      {"points_grad", "sdf_values_grad", "sdf_grads_grad"},
      R"metal(
        uint i = thread_position_in_grid.x;
        uint N = points_shape[0];
        if (i >= N) {
          return;
        }

        uint M = sdf_values_shape[1];
        float best_abs = INFINITY;
        uint best_idx = 0;
        for (uint m = 0; m < M; ++m) {
          float abs_value = fabs(sdf_values[i * M + m]);
          if (abs_value < best_abs) {
            best_abs = abs_value;
            best_idx = m;
          }
        }

        float distance = sdf_values[i * M + best_idx];
        float distance_grad = 0.0f;
        for (uint axis = 0; axis < 3; ++axis) {
          float upstream = cotangent[i * 3 + axis];
          float normal = sdf_grads[(i * M + best_idx) * 3 + axis];
          points_grad[i * 3 + axis] = upstream;
          distance_grad -= upstream * normal;
        }

        for (uint m = 0; m < M; ++m) {
          bool selected = (m == best_idx);
          sdf_values_grad[i * M + m] = selected ? distance_grad : 0.0f;
          for (uint axis = 0; axis < 3; ++axis) {
            float upstream = cotangent[i * 3 + axis];
            sdf_grads_grad[(i * M + m) * 3 + axis] =
                selected ? -distance * upstream : 0.0f;
          }
        }
      )metal");

  const int threads = std::min(rows, 256);
  return kernel(
             {points, sdf_values, sdf_grads, cotangent},
             {points.shape(), sdf_values.shape(), sdf_grads.shape()},
             {mx::float32, mx::float32, mx::float32},
             {rows, 1, 1},
             {threads, 1, 1},
             {},
             std::nullopt,
             false,
             stream);
}

NB_MODULE(_ext, m) {
  m.def(
      "_nearest_surface_projection",
      &nearest_surface_projection,
      "points"_a,
      "sdf_values"_a,
      "sdf_grads"_a,
      nb::kw_only(),
      "stream"_a = nb::none());
  m.def(
      "_nearest_surface_projection_vjp",
      &nearest_surface_projection_vjp,
      "points"_a,
      "sdf_values"_a,
      "sdf_grads"_a,
      "cotangent"_a,
      nb::kw_only(),
      "stream"_a = nb::none());
}
