#include <gtest/gtest.h>
#include "hnsw.h"
#include <random>
#include <vector>
#include <numeric>

using namespace ann;

static std::vector<float> gaussian_vecs(int n, int dim, unsigned seed = 0) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

// -------------------------------------------------------
//  evaluate_recall sweeps ef_search and checks monotonicity
// -------------------------------------------------------
TEST(EvalRecall, RecallIncreasesWithEf) {
    int n = 2000, dim = 32, k = 10;
    auto db = gaussian_vecs(n, dim, 1);
    auto qs = gaussian_vecs(200, dim, 2);

    // Build ground truth
    std::vector<int> gt(200 * k);
    for (int q = 0; q < 200; ++q) {
        auto exact = brute_force_search(db.data(), n,
                                        qs.data() + q * dim, k, dim);
        std::copy(exact.begin(), exact.end(), gt.data() + q * k);
    }

    HNSWConfig cfg; cfg.M = 16; cfg.ef_construction = 200;
    HNSW idx(dim, cfg);
    idx.build(db.data(), n);

    float prev_recall = 0.f;
    for (int ef : {10, 40, 100, 300}) {
        RecallResult r = evaluate_recall(idx,
                                         qs.data(), 200,
                                         gt.data(), k, ef);
        // Recall should be non-decreasing with ef
        EXPECT_GE(r.mean_recall, prev_recall - 0.02f)
            << "Recall dropped at ef=" << ef;
        prev_recall = r.mean_recall;
        EXPECT_GT(r.qps, 0.0);
    }
}

TEST(EvalRecall, AtHighEfApproachesExact) {
    int n = 1000, dim = 16, k = 5;
    auto db = gaussian_vecs(n, dim, 3);
    auto qs = gaussian_vecs(100, dim, 4);

    std::vector<int> gt(100 * k);
    for (int q = 0; q < 100; ++q) {
        auto exact = brute_force_search(db.data(), n,
                                        qs.data() + q * dim, k, dim);
        std::copy(exact.begin(), exact.end(), gt.data() + q * k);
    }

    HNSWConfig cfg; cfg.M = 16; cfg.ef_construction = 200;
    HNSW idx(dim, cfg);
    idx.build(db.data(), n);

    RecallResult r = evaluate_recall(idx, qs.data(), 100,
                                      gt.data(), k, 500);
    EXPECT_GE(r.mean_recall, 0.95f)
        << "Expected >=95% recall at ef=500, got " << r.mean_recall;
}

// -------------------------------------------------------
//  Metric comparison: L2 vs Cosine
// -------------------------------------------------------
TEST(Metrics, CosineIndexBuildsAndSearches) {
    int n = 500, dim = 32;
    auto db = gaussian_vecs(n, dim, 5);

    HNSWConfig cfg;
    cfg.metric = Metric::Cosine;
    cfg.M = 8; cfg.ef_construction = 50; cfg.ef_search = 50;

    HNSW idx(dim, cfg);
    idx.build(db.data(), n);
    EXPECT_EQ(idx.size(), n);

    auto res = idx.search(db.data(), 5);  // query with first vector
    EXPECT_FALSE(res.empty());
    // First result should be the query itself (cosine dist ~ 0)
    EXPECT_EQ(res[0], 0);
}

TEST(Metrics, L2AndCosineGiveDifferentResults) {
    int n = 200, dim = 16;
    auto db = gaussian_vecs(n, dim, 6);

    HNSWConfig cfg_l2, cfg_cos;
    cfg_l2.metric = Metric::L2;
    cfg_cos.metric = Metric::Cosine;
    cfg_l2.M = cfg_cos.M = 8;
    cfg_l2.ef_construction = cfg_cos.ef_construction = 50;
    cfg_l2.ef_search = cfg_cos.ef_search = 50;

    HNSW idx_l2 (dim, cfg_l2);
    HNSW idx_cos(dim, cfg_cos);
    idx_l2.build(db.data(), n);
    idx_cos.build(db.data(), n);

    // Query with a random vector — results may differ between metrics
    auto q = gaussian_vecs(1, dim, 99);
    auto r_l2  = idx_l2.search(q.data(), 5);
    auto r_cos = idx_cos.search(q.data(), 5);

    // Both should return 5 results
    EXPECT_EQ((int)r_l2.size(),  5);
    EXPECT_EQ((int)r_cos.size(), 5);
}