#include <gtest/gtest.h>
#include "hnsw.h"
#include <fstream>
#include <random>
#include <vector>
#include <unordered_set>
#include <cmath>

using namespace ann;

// -------------------------------------------------------
//  Helpers
// -------------------------------------------------------
static std::vector<float> random_vecs(int n, int dim, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = d(rng);
    return v;
}

// -------------------------------------------------------
//  Basic construction
// -------------------------------------------------------
TEST(HNSWBuild, EmptyIndex) {
    HNSW idx(128);
    EXPECT_EQ(idx.size(), 0);
}

TEST(HNSWBuild, SingleInsert) {
    HNSW idx(4);
    float v[] = {1.f, 2.f, 3.f, 4.f};
    idx.insert(v);
    EXPECT_EQ(idx.size(), 1);
}

TEST(HNSWBuild, TenInserts) {
    HNSW idx(8);
    auto vecs = random_vecs(10, 8);
    for (int i = 0; i < 10; ++i)
        idx.insert(vecs.data() + i * 8);
    EXPECT_EQ(idx.size(), 10);
}

TEST(HNSWBuild, BuildApi) {
    auto vecs = random_vecs(200, 32);
    HNSWConfig cfg; cfg.M = 8; cfg.ef_construction = 50;
    HNSW idx(32, cfg);
    idx.build(vecs.data(), 200);
    EXPECT_EQ(idx.size(), 200);
    EXPECT_GE(idx.num_layers(), 1);
}

// -------------------------------------------------------
//  Correctness: search finds nearest neighbors
// -------------------------------------------------------
TEST(HNSWSearch, FindsItselfWhenQueried) {
    // Insert 100 vectors; query each one back — should return itself as #1
    int n = 100, dim = 16;
    auto vecs = random_vecs(n, dim, 7);
    HNSWConfig cfg; cfg.M = 8; cfg.ef_construction = 100; cfg.ef_search = 100;
    HNSW idx(dim, cfg);
    idx.build(vecs.data(), n);

    int found = 0;
    for (int i = 0; i < n; ++i) {
        auto res = idx.search(vecs.data() + i * dim, 1);
        ASSERT_FALSE(res.empty());
        if (res[0] == i) ++found;
    }
    // Expect at least 95% self-retrieval — HNSW is approximate
    EXPECT_GE(found, (int)(n * 0.95));
}

TEST(HNSWSearch, ReturnsKResults) {
    auto vecs = random_vecs(500, 16);
    HNSW idx(16);
    idx.build(vecs.data(), 500);

    float q[16] = {};
    auto res = idx.search(q, 10);
    EXPECT_EQ((int)res.size(), 10);
}

TEST(HNSWSearch, EmptyIndexReturnsEmpty) {
    HNSW idx(8);
    float q[8] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    auto res = idx.search(q, 5);
    EXPECT_TRUE(res.empty());
}

TEST(HNSWSearch, KLargerThanNReturnsAll) {
    auto vecs = random_vecs(5, 8);
    HNSW idx(8);
    idx.build(vecs.data(), 5);
    auto res = idx.search(vecs.data(), 100);
    EXPECT_EQ((int)res.size(), 5);  // at most n
}

TEST(HNSWSearch, NoDuplicateResults) {
    auto vecs = random_vecs(200, 32);
    HNSW idx(32);
    idx.build(vecs.data(), 200);
    auto res = idx.search(vecs.data(), 20);
    std::unordered_set<int> seen(res.begin(), res.end());
    EXPECT_EQ(seen.size(), res.size());
}

TEST(HNSWSearch, SearchWithDistances) {
    auto vecs = random_vecs(100, 16);
    HNSW idx(16);
    idx.build(vecs.data(), 100);
    auto res = idx.search_with_distances(vecs.data(), 5);
    ASSERT_EQ((int)res.size(), 5);
    // Distances should be non-decreasing
    for (int i = 1; i < (int)res.size(); ++i)
        EXPECT_LE(res[i-1].first, res[i].first);
}

// -------------------------------------------------------
//  Recall against brute force
// -------------------------------------------------------
TEST(HNSWRecall, HighRecallAtHighEf) {
    int n = 1000, dim = 32, k = 10;
    auto db = random_vecs(n, dim, 1);
    auto qs = random_vecs(100, dim, 2);

    HNSWConfig cfg; cfg.M = 16; cfg.ef_construction = 200; cfg.ef_search = 500;
    HNSW idx(dim, cfg);
    idx.build(db.data(), n);

    float total_recall = 0.f;
    for (int q = 0; q < 100; ++q) {
        auto approx = idx.search(qs.data() + q * dim, k);
        auto exact  = brute_force_search(db.data(), n, qs.data() + q * dim,
                                          k, dim);
        total_recall += compute_recall(approx, exact.data(), k);
    }
    float mean = total_recall / 100;
    // Should be > 0.90 at ef=500 on random vectors
    EXPECT_GE(mean, 0.90f) << "Mean recall@10 = " << mean;
}

// -------------------------------------------------------
//  Save / load round-trip
// -------------------------------------------------------
TEST(HNSWSaveLoad, RoundTrip) {
    int n = 300, dim = 16;
    auto vecs = random_vecs(n, dim);
    HNSWConfig cfg; cfg.M = 8; cfg.ef_construction = 50; cfg.ef_search = 100;
    HNSW idx(dim, cfg);
    idx.build(vecs.data(), n);

    std::string tmp = "/tmp/test_hnsw_saveload.bin";
    idx.save(tmp);

    HNSW loaded(1);
    loaded.load(tmp);

    EXPECT_EQ(loaded.size(),  idx.size());
    EXPECT_EQ(loaded.dim(),   idx.dim());

    // Queries on loaded index should match original
    float q[16] = {};
    auto r1 = idx.search(q, 5);
    auto r2 = loaded.search(q, 5);
    EXPECT_EQ(r1, r2);
}

TEST(HNSWSaveLoad, WrongMagicThrows) {
    std::string tmp = "/tmp/test_bad_magic.bin";
    {
        std::ofstream f(tmp, std::ios::binary);
        uint32_t bad = 0xDEADBEEF;
        f.write((char*)&bad, 4);
    }
    HNSW idx(8);
    EXPECT_THROW(idx.load(tmp), std::runtime_error);
}

// -------------------------------------------------------
//  brute_force_search
// -------------------------------------------------------
TEST(BruteForce, ReturnsExactNearest) {
    float db[] = {0.f,0.f,  1.f,0.f,  2.f,0.f,  10.f,0.f};
    float q[]  = {1.5f, 0.f};
    auto res = brute_force_search(db, 4, q, 2, 2);
    // closest to 1.5 are 2.0 (id=2) and 1.0 (id=1)
    std::unordered_set<int> s(res.begin(), res.end());
    EXPECT_TRUE(s.count(1) || s.count(2));
    EXPECT_EQ((int)res.size(), 2);
}

// -------------------------------------------------------
//  compute_recall
// -------------------------------------------------------
TEST(Recall, PerfectRecall) {
    std::vector<int> approx = {0,1,2,3,4};
    int gt[] = {0,1,2,3,4};
    EXPECT_FLOAT_EQ(compute_recall(approx, gt, 5), 1.f);
}

TEST(Recall, ZeroRecall) {
    std::vector<int> approx = {5,6,7,8,9};
    int gt[] = {0,1,2,3,4};
    EXPECT_FLOAT_EQ(compute_recall(approx, gt, 5), 0.f);
}

TEST(Recall, HalfRecall) {
    std::vector<int> approx = {0,1,5,6,7};
    int gt[] = {0,1,2,3,4};
    EXPECT_NEAR(compute_recall(approx, gt, 5), 0.4f, 1e-5f);
}
