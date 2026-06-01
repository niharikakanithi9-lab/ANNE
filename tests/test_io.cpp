#include <gtest/gtest.h>
#include "io.h"
#include <fstream>
#include <cstdio>

using namespace ann;

static std::string write_fvecs(const std::string& path,
                                const std::vector<std::vector<float>>& vecs) {
    std::ofstream f(path, std::ios::binary);
    for (auto& v : vecs) {
        int32_t dim = (int32_t)v.size();
        f.write(reinterpret_cast<char*>(&dim), 4);
        f.write(reinterpret_cast<const char*>(v.data()), dim * 4);
    }
    return path;
}

static std::string write_ivecs(const std::string& path,
                                const std::vector<std::vector<int>>& vecs) {
    std::ofstream f(path, std::ios::binary);
    for (auto& v : vecs) {
        int32_t dim = (int32_t)v.size();
        f.write(reinterpret_cast<char*>(&dim), 4);
        f.write(reinterpret_cast<const char*>(v.data()), dim * 4);
    }
    return path;
}

// -------------------------------------------------------
//  fvecs
// -------------------------------------------------------
TEST(IOFvecs, LoadCorrectly) {
    std::vector<std::vector<float>> vecs = {{1.f,2.f,3.f},{4.f,5.f,6.f}};
    write_fvecs("/tmp/test_io.fvecs", vecs);

    Dataset ds = load_fvecs("/tmp/test_io.fvecs");
    EXPECT_EQ(ds.n,   2);
    EXPECT_EQ(ds.dim, 3);
    EXPECT_FLOAT_EQ(ds.row(0)[0], 1.f);
    EXPECT_FLOAT_EQ(ds.row(0)[2], 3.f);
    EXPECT_FLOAT_EQ(ds.row(1)[0], 4.f);
}

TEST(IOFvecs, MaxNLimit) {
    std::vector<std::vector<float>> vecs = {{1.f},{2.f},{3.f},{4.f},{5.f}};
    write_fvecs("/tmp/test_max_n.fvecs", vecs);
    Dataset ds = load_fvecs("/tmp/test_max_n.fvecs", 3);
    EXPECT_EQ(ds.n, 3);
}

TEST(IOFvecs, NonExistentFileThrows) {
    EXPECT_THROW(load_fvecs("/tmp/nonexistent_xyz.fvecs"), std::runtime_error);
}

TEST(IOFvecs, SaveLoadRoundTrip) {
    std::vector<std::vector<float>> vecs = {{1.f,2.f},{3.f,4.f},{5.f,6.f}};
    write_fvecs("/tmp/test_rt.fvecs", vecs);
    Dataset ds1 = load_fvecs("/tmp/test_rt.fvecs");
    save_fvecs("/tmp/test_rt_out.fvecs", ds1);
    Dataset ds2 = load_fvecs("/tmp/test_rt_out.fvecs");
    EXPECT_EQ(ds1.n,   ds2.n);
    EXPECT_EQ(ds1.dim, ds2.dim);
    EXPECT_EQ(ds1.data, ds2.data);
}

// -------------------------------------------------------
//  ivecs
// -------------------------------------------------------
TEST(IOIvecs, LoadCorrectly) {
    std::vector<std::vector<int>> vecs = {{10,20,30},{40,50,60}};
    write_ivecs("/tmp/test_io.ivecs", vecs);
    IDataset ds = load_ivecs("/tmp/test_io.ivecs");
    EXPECT_EQ(ds.n,   2);
    EXPECT_EQ(ds.dim, 3);
    EXPECT_EQ(ds.row(0)[0], 10);
    EXPECT_EQ(ds.row(1)[2], 60);
}

// -------------------------------------------------------
//  Dataset::row
// -------------------------------------------------------
TEST(DatasetRow, CorrectPointerOffset) {
    Dataset ds;
    ds.n = 3; ds.dim = 4;
    ds.data = {0,1,2,3, 4,5,6,7, 8,9,10,11};
    EXPECT_EQ(ds.row(0)[0], 0);
    EXPECT_EQ(ds.row(1)[0], 4);
    EXPECT_EQ(ds.row(2)[3], 11);
}