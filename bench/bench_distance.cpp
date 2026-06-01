#include <benchmark/benchmark.h>
#include "distance.h"
#include <vector>
#include <random>
#include <numeric>

using namespace ann;

// -------------------------------------------------------
//  Fixtures
// -------------------------------------------------------
static std::vector<float> make_vecs(int n, int dim, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = d(rng);
    return v;
}

// -------------------------------------------------------
//  L2 — scalar
// -------------------------------------------------------
static void BM_L2_Scalar_Dim128(benchmark::State& state) {
    auto data = make_vecs(2, 128);
    const float* a = data.data();
    const float* b = data.data() + 128;
    for (auto _ : state)
        benchmark::DoNotOptimize(l2_sq_scalar(a, b, 128));
}
BENCHMARK(BM_L2_Scalar_Dim128);

static void BM_L2_Scalar_Dim960(benchmark::State& state) {
    auto data = make_vecs(2, 960);
    const float* a = data.data();
    const float* b = data.data() + 960;
    for (auto _ : state)
        benchmark::DoNotOptimize(l2_sq_scalar(a, b, 960));
}
BENCHMARK(BM_L2_Scalar_Dim960);

// -------------------------------------------------------
//  L2 — AVX2
// -------------------------------------------------------
#ifdef HAVE_AVX2
static void BM_L2_AVX2_Dim128(benchmark::State& state) {
    auto data = make_vecs(2, 128);
    const float* a = data.data();
    const float* b = data.data() + 128;
    for (auto _ : state)
        benchmark::DoNotOptimize(l2_sq_avx2(a, b, 128));
}
BENCHMARK(BM_L2_AVX2_Dim128);

static void BM_L2_AVX2_Dim960(benchmark::State& state) {
    auto data = make_vecs(2, 960);
    const float* a = data.data();
    const float* b = data.data() + 960;
    for (auto _ : state)
        benchmark::DoNotOptimize(l2_sq_avx2(a, b, 960));
}
BENCHMARK(BM_L2_AVX2_Dim960);

static void BM_Cosine_AVX2_Dim128(benchmark::State& state) {
    auto data = make_vecs(2, 128);
    const float* a = data.data();
    const float* b = data.data() + 128;
    for (auto _ : state)
        benchmark::DoNotOptimize(cosine_avx2(a, b, 128));
}
BENCHMARK(BM_Cosine_AVX2_Dim128);

static void BM_Dot_AVX2_Dim128(benchmark::State& state) {
    auto data = make_vecs(2, 128);
    const float* a = data.data();
    const float* b = data.data() + 128;
    for (auto _ : state)
        benchmark::DoNotOptimize(dot_avx2(a, b, 128));
}
BENCHMARK(BM_Dot_AVX2_Dim128);
#endif

// -------------------------------------------------------
//  L2 dispatch
// -------------------------------------------------------
static void BM_L2_Dispatch_Dim128(benchmark::State& state) {
    auto data = make_vecs(2, 128);
    const float* a = data.data();
    const float* b = data.data() + 128;
    for (auto _ : state)
        benchmark::DoNotOptimize(l2_sq(a, b, 128));
}
BENCHMARK(BM_L2_Dispatch_Dim128);

// -------------------------------------------------------
//  Batch throughput: 1 query vs 10K database vectors
// -------------------------------------------------------
static void BM_L2_BatchQuery_10K(benchmark::State& state) {
    int dim = 128, n = 10000;
    auto db    = make_vecs(n, dim, 1);
    auto query = make_vecs(1, dim, 2);
    for (auto _ : state) {
        float worst = 1e30f;
        for (int i = 0; i < n; ++i) {
            float d = l2_sq(query.data(), db.data() + (size_t)i * dim, dim);
            benchmark::DoNotOptimize(d);
            worst = std::min(worst, d);
        }
        benchmark::DoNotOptimize(worst);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_L2_BatchQuery_10K);

BENCHMARK_MAIN();