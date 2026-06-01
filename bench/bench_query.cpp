#include <benchmark/benchmark.h>
#include "hnsw.h"
#include <random>
#include <vector>

using namespace ann;

// -------------------------------------------------------
//  Shared fixture: build once, reuse across benchmarks
// -------------------------------------------------------
struct IndexFixture {
    static constexpr int N   = 100000;  // 100K vectors
    static constexpr int DIM = 128;
    static constexpr int K   = 10;

    std::vector<float> db;
    std::vector<float> queries;
    HNSW index;

    IndexFixture()
        : db(make_vecs(N, DIM, 1))
        , queries(make_vecs(1000, DIM, 2))
        , index(DIM, make_cfg())
    {
        index.build(db.data(), N);
    }

    static std::vector<float> make_vecs(int n, int dim, unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        std::vector<float> v((size_t)n * dim);
        for (auto& x : v) x = d(rng);
        return v;
    }

    static HNSWConfig make_cfg() {
        HNSWConfig c;
        c.M = 16;
        c.ef_construction = 200;
        c.ef_search = 50;
        return c;
    }
};

// Singleton — build index once
static IndexFixture& get_fixture() {
    static IndexFixture fx;
    return fx;
}

// -------------------------------------------------------
//  Query benchmarks at different ef_search values
// -------------------------------------------------------
static void BM_Query_ef10(benchmark::State& state) {
    auto& fx = get_fixture();
    fx.index.set_ef_search(10);
    int q = 0;
    for (auto _ : state) {
        const float* qv = fx.queries.data() + (size_t)(q++ % 1000) * IndexFixture::DIM;
        auto res = fx.index.search(qv, IndexFixture::K);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_Query_ef10)->Unit(benchmark::kMicrosecond);

static void BM_Query_ef50(benchmark::State& state) {
    auto& fx = get_fixture();
    fx.index.set_ef_search(50);
    int q = 0;
    for (auto _ : state) {
        const float* qv = fx.queries.data() + (size_t)(q++ % 1000) * IndexFixture::DIM;
        auto res = fx.index.search(qv, IndexFixture::K);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_Query_ef50)->Unit(benchmark::kMicrosecond);

static void BM_Query_ef200(benchmark::State& state) {
    auto& fx = get_fixture();
    fx.index.set_ef_search(200);
    int q = 0;
    for (auto _ : state) {
        const float* qv = fx.queries.data() + (size_t)(q++ % 1000) * IndexFixture::DIM;
        auto res = fx.index.search(qv, IndexFixture::K);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_Query_ef200)->Unit(benchmark::kMicrosecond);

static void BM_Query_ef500(benchmark::State& state) {
    auto& fx = get_fixture();
    fx.index.set_ef_search(500);
    int q = 0;
    for (auto _ : state) {
        const float* qv = fx.queries.data() + (size_t)(q++ % 1000) * IndexFixture::DIM;
        auto res = fx.index.search(qv, IndexFixture::K);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_Query_ef500)->Unit(benchmark::kMicrosecond);

// -------------------------------------------------------
//  Brute force baseline
// -------------------------------------------------------
static void BM_BruteForce(benchmark::State& state) {
    auto& fx = get_fixture();
    int q = 0;
    for (auto _ : state) {
        const float* qv = fx.queries.data() + (size_t)(q++ % 1000) * IndexFixture::DIM;
        auto res = brute_force_search(fx.db.data(), IndexFixture::N,
                                       qv, IndexFixture::K, IndexFixture::DIM);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_BruteForce)->Unit(benchmark::kMillisecond);

// -------------------------------------------------------
//  Build time benchmark (small n to keep CI fast)
// -------------------------------------------------------
static void BM_Build_10K(benchmark::State& state) {
    int n = 10000, dim = 128;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<float> data((size_t)n * dim);
    for (auto& x : data) x = d(rng);

    HNSWConfig cfg; cfg.M = 16; cfg.ef_construction = 200;
    for (auto _ : state) {
        HNSW idx(dim, cfg);
        idx.build(data.data(), n);
        benchmark::DoNotOptimize(idx.size());
    }
}
BENCHMARK(BM_Build_10K)->Unit(benchmark::kMillisecond)->Iterations(3);

BENCHMARK_MAIN();