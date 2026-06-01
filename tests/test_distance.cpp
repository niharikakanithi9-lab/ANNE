#include <gtest/gtest.h>
#include "distance.h"
#include <cmath>
#include <random>
#include <vector>

using namespace ann;

// -------------------------------------------------------
//  Scalar correctness
// -------------------------------------------------------
TEST(L2Scalar, ZeroVector) {
    std::vector<float> a(128, 0.f), b(128, 0.f);
    EXPECT_FLOAT_EQ(l2_sq_scalar(a.data(), b.data(), 128), 0.f);
}

TEST(L2Scalar, IdenticalVectors) {
    std::vector<float> a(128, 3.14f);
    EXPECT_FLOAT_EQ(l2_sq_scalar(a.data(), a.data(), 128), 0.f);
}

TEST(L2Scalar, KnownValue) {
    // [1,0,0] vs [0,1,0] => L2_sq = 2
    float a[] = {1.f, 0.f, 0.f};
    float b[] = {0.f, 1.f, 0.f};
    EXPECT_FLOAT_EQ(l2_sq_scalar(a, b, 3), 2.f);
}

TEST(L2Scalar, Dim1) {
    float a[] = {5.f};
    float b[] = {2.f};
    EXPECT_FLOAT_EQ(l2_sq_scalar(a, b, 1), 9.f);
}

TEST(DotScalar, Orthogonal) {
    float a[] = {1.f, 0.f};
    float b[] = {0.f, 1.f};
    EXPECT_FLOAT_EQ(dot_scalar(a, b, 2), 0.f);
}

TEST(DotScalar, Parallel) {
    float a[] = {1.f, 0.f};
    EXPECT_FLOAT_EQ(dot_scalar(a, a, 2), 1.f);
}

TEST(CosineScalar, IdenticalVectors) {
    std::vector<float> a(16, 1.f);
    float dist = cosine_scalar(a.data(), a.data(), 16);
    EXPECT_NEAR(dist, 0.f, 1e-5f);  // cosine distance = 0 when identical
}

TEST(CosineScalar, Orthogonal) {
    float a[] = {1.f, 0.f};
    float b[] = {0.f, 1.f};
    float dist = cosine_scalar(a, b, 2);
    EXPECT_NEAR(dist, 1.f, 1e-5f);  // cosine distance = 1 when orthogonal
}

// -------------------------------------------------------
//  AVX2 vs scalar agreement
// -------------------------------------------------------
#ifdef HAVE_AVX2
class DistanceAvx2Test : public ::testing::Test {
protected:
    std::vector<float> a, b;
    void SetUp() override {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-5.f, 5.f);
        a.resize(128); b.resize(128);
        for (auto& x : a) x = dist(rng);
        for (auto& x : b) x = dist(rng);
    }
};

TEST_F(DistanceAvx2Test, L2AgreesDim128) {
    float s = l2_sq_scalar(a.data(), b.data(), 128);
    float v = l2_sq_avx2  (a.data(), b.data(), 128);
    EXPECT_NEAR(s, v, std::abs(s) * 1e-4f + 1e-6f);
}

TEST_F(DistanceAvx2Test, L2AgreesDim127_TailLoop) {
    float s = l2_sq_scalar(a.data(), b.data(), 127);
    float v = l2_sq_avx2  (a.data(), b.data(), 127);
    EXPECT_NEAR(s, v, std::abs(s) * 1e-4f + 1e-6f);
}

TEST_F(DistanceAvx2Test, L2AgreesDim1) {
    float s = l2_sq_scalar(a.data(), b.data(), 1);
    float v = l2_sq_avx2  (a.data(), b.data(), 1);
    EXPECT_NEAR(s, v, 1e-5f);
}

TEST_F(DistanceAvx2Test, DotAgresDim128) {
    float s = dot_scalar(a.data(), b.data(), 128);
    float v = dot_avx2  (a.data(), b.data(), 128);
    EXPECT_NEAR(s, v, std::abs(s) * 1e-4f + 1e-6f);
}

TEST_F(DistanceAvx2Test, CosineAgresDim128) {
    float s = cosine_scalar(a.data(), b.data(), 128);
    float v = cosine_avx2  (a.data(), b.data(), 128);
    EXPECT_NEAR(s, v, 1e-4f);
}

TEST(AvxDispatch, DispatchCallsAvx2) {
    std::vector<float> a(128, 1.f), b(128, 2.f);
    // dispatch should match AVX2 exactly
    float dispatched = l2_sq(a.data(), b.data(), 128);
    float avx2       = l2_sq_avx2(a.data(), b.data(), 128);
    EXPECT_FLOAT_EQ(dispatched, avx2);
}
#endif

// -------------------------------------------------------
//  get_dist_fn
// -------------------------------------------------------
TEST(DistFn, L2FnWorks) {
    auto fn = get_dist_fn(Metric::L2);
    float a[] = {0.f, 0.f};
    float b[] = {3.f, 4.f};
    EXPECT_NEAR(fn(a, b, 2), 25.f, 1e-4f);
}

TEST(DistFn, DotFnReturnsNegative) {
    // DotProduct metric negates — so result should be negative for positive dot
    auto fn = get_dist_fn(Metric::DotProduct);
    float a[] = {1.f, 0.f};
    float b[] = {1.f, 0.f};
    EXPECT_LT(fn(a, b, 2), 0.f);
}

TEST(MetricString, RoundTrip) {
    EXPECT_EQ(metric_to_string(metric_from_string("l2")),     "l2");
    EXPECT_EQ(metric_to_string(metric_from_string("cosine")), "cosine");
    EXPECT_EQ(metric_to_string(metric_from_string("dot")),    "dot");
}

TEST(MetricString, UnknownThrows) {
    EXPECT_THROW(metric_from_string("euclidean"), std::invalid_argument);
}