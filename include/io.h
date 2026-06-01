#pragma once
#include <vector>
#include <string>
#include <cstdint>

namespace ann {

// -------------------------------------------------------
//  Standard ANN benchmark file formats
//  .fvecs: [dim:int32][v0..v_{dim-1}:float32] x N
//  .bvecs: [dim:int32][v0..v_{dim-1}:uint8  ] x N
//  .ivecs: [dim:int32][v0..v_{dim-1}:int32  ] x N
// -------------------------------------------------------

struct Dataset {
    std::vector<float> data;  // flat: data[i*dim + d]
    int n   = 0;
    int dim = 0;

    const float* row(int i) const { return data.data() + (size_t)i * dim; }
};

struct IDataset {
    std::vector<int> data;
    int n   = 0;
    int dim = 0;

    const int* row(int i) const { return data.data() + (size_t)i * dim; }
};

// Load float vectors (.fvecs)
Dataset  load_fvecs(const std::string& path, int max_n = -1);

// Load byte vectors (.bvecs) — converts to float
Dataset  load_bvecs(const std::string& path, int max_n = -1);

// Load integer vectors (.ivecs) — used for ground-truth
IDataset load_ivecs(const std::string& path, int max_n = -1);

// Save float vectors back to .fvecs
void save_fvecs(const std::string& path, const Dataset& ds);

// Pretty-print dataset info
void print_dataset_info(const std::string& name, const Dataset& ds);

} // namespace ann