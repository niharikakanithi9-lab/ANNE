#pragma once
#include <cstddef>
#include <functional>
#include <string>

namespace ann {

// -------------------------------------------------------
//  Distance metric enum
// -------------------------------------------------------
enum class Metric { L2, Cosine, DotProduct };

Metric metric_from_string(const std::string& s);
std::string metric_to_string(Metric m);

// -------------------------------------------------------
//  Scalar implementations (always available, used in tests)
// -------------------------------------------------------
float l2_sq_scalar   (const float* a, const float* b, size_t dim);
float cosine_scalar  (const float* a, const float* b, size_t dim);
float dot_scalar     (const float* a, const float* b, size_t dim);

// -------------------------------------------------------
//  SIMD implementations (guarded by HAVE_AVX2)
// -------------------------------------------------------
#ifdef HAVE_AVX2
float l2_sq_avx2   (const float* a, const float* b, size_t dim);
float cosine_avx2  (const float* a, const float* b, size_t dim);
float dot_avx2     (const float* a, const float* b, size_t dim);
#endif

// -------------------------------------------------------
//  Dispatch: picks AVX2 if available, else scalar
// -------------------------------------------------------
float l2_sq   (const float* a, const float* b, size_t dim);
float cosine  (const float* a, const float* b, size_t dim);
float dot     (const float* a, const float* b, size_t dim);

// Returns the right dispatch function for a metric
using DistFn = std::function<float(const float*, const float*, size_t)>;
DistFn get_dist_fn(Metric m);

} // namespace ann