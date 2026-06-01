#include "distance.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifdef HAVE_AVX2
#include <immintrin.h>
#endif

namespace ann {

// -------------------------------------------------------
//  Metric helpers
// -------------------------------------------------------
Metric metric_from_string(const std::string& s) {
    if (s == "l2"  || s == "L2")         return Metric::L2;
    if (s == "cosine")                    return Metric::Cosine;
    if (s == "dot" || s == "dotproduct") return Metric::DotProduct;
    throw std::invalid_argument("Unknown metric: " + s);
}

std::string metric_to_string(Metric m) {
    switch (m) {
        case Metric::L2:          return "l2";
        case Metric::Cosine:      return "cosine";
        case Metric::DotProduct:  return "dot";
    }
    return "unknown";
}

// -------------------------------------------------------
//  Scalar implementations
// -------------------------------------------------------
float l2_sq_scalar(const float* a, const float* b, size_t dim) {
    float sum = 0.f;
    for (size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

float dot_scalar(const float* a, const float* b, size_t dim) {
    float sum = 0.f;
    for (size_t i = 0; i < dim; ++i) sum += a[i] * b[i];
    return sum;
}

float cosine_scalar(const float* a, const float* b, size_t dim) {
    float dot = 0.f, na = 0.f, nb = 0.f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    if (denom < 1e-10f) return 1.f;  // treat zero vectors as maximally distant
    return 1.f - dot / denom;        // return DISTANCE (lower = more similar)
}

// -------------------------------------------------------
//  AVX2 implementations
// -------------------------------------------------------
#ifdef HAVE_AVX2

// Horizontal sum of 8-lane __m256 register
inline float hsum_avx(__m256 v) {
    __m128 lo  = _mm256_castps256_ps128(v);
    __m128 hi  = _mm256_extractf128_ps(v, 1);
    __m128 s4  = _mm_add_ps(lo, hi);
    __m128 s2  = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1  = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0x1));
    return _mm_cvtss_f32(s1);
}

float l2_sq_avx2(const float* a, const float* b, size_t dim) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va   = _mm256_loadu_ps(a + i);
        __m256 vb   = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }
    float result = hsum_avx(acc);
    // tail
    for (; i < dim; ++i) { float d = a[i] - b[i]; result += d * d; }
    return result;
}

float dot_avx2(const float* a, const float* b, size_t dim) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float result = hsum_avx(acc);
    for (; i < dim; ++i) result += a[i] * b[i];
    return result;
}

float cosine_avx2(const float* a, const float* b, size_t dim) {
    __m256 acc_dot = _mm256_setzero_ps();
    __m256 acc_na  = _mm256_setzero_ps();
    __m256 acc_nb  = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc_dot = _mm256_fmadd_ps(va, vb, acc_dot);
        acc_na  = _mm256_fmadd_ps(va, va, acc_na);
        acc_nb  = _mm256_fmadd_ps(vb, vb, acc_nb);
    }
    float dot_v = hsum_avx(acc_dot);
    float na    = hsum_avx(acc_na);
    float nb    = hsum_avx(acc_nb);
    // tail
    for (; i < dim; ++i) {
        dot_v += a[i] * b[i];
        na    += a[i] * a[i];
        nb    += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    if (denom < 1e-10f) return 1.f;
    return 1.f - dot_v / denom;
}

#endif // HAVE_AVX2

// -------------------------------------------------------
//  Dispatch functions — pick AVX2 or scalar at runtime
// -------------------------------------------------------
float l2_sq(const float* a, const float* b, size_t dim) {
#ifdef HAVE_AVX2
    return l2_sq_avx2(a, b, dim);
#else
    return l2_sq_scalar(a, b, dim);
#endif
}

float cosine(const float* a, const float* b, size_t dim) {
#ifdef HAVE_AVX2
    return cosine_avx2(a, b, dim);
#else
    return cosine_scalar(a, b, dim);
#endif
}

float dot(const float* a, const float* b, size_t dim) {
#ifdef HAVE_AVX2
    return dot_avx2(a, b, dim);
#else
    return dot_scalar(a, b, dim);
#endif
}

DistFn get_dist_fn(Metric m) {
    switch (m) {
        case Metric::L2:         return l2_sq;
        case Metric::Cosine:     return cosine;
        case Metric::DotProduct: return [](const float* a, const float* b, size_t d){
                                     return -dot(a, b, d);  // negate: closer = smaller
                                 };
    }
    return l2_sq;
}

} // namespace ann