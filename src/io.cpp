#include "io.h"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <iomanip>

namespace ann {

// -------------------------------------------------------
//  .fvecs loader
// -------------------------------------------------------
Dataset load_fvecs(const std::string& path, int max_n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    Dataset ds;
    int32_t dim = 0;

    // Read first dim to detect dimensionality
    if (!f.read(reinterpret_cast<char*>(&dim), 4))
        throw std::runtime_error("Empty fvecs file: " + path);
    f.seekg(0);

    ds.dim = dim;
    int bytes_per_vec = 4 + dim * 4;

    // Get file size to compute N
    f.seekg(0, std::ios::end);
    std::streamsize file_size = f.tellg();
    f.seekg(0);

    int total_n = static_cast<int>(file_size / bytes_per_vec);
    int n = (max_n > 0) ? std::min(total_n, max_n) : total_n;

    ds.n = n;
    ds.data.resize((size_t)n * dim);

    for (int i = 0; i < n; ++i) {
        int32_t d;
        f.read(reinterpret_cast<char*>(&d), 4);
        if (d != dim)
            throw std::runtime_error("Inconsistent dim in fvecs at row " + std::to_string(i));
        f.read(reinterpret_cast<char*>(ds.data.data() + (size_t)i * dim), dim * 4);
    }
    return ds;
}

// -------------------------------------------------------
//  .bvecs loader (converts uint8 → float)
// -------------------------------------------------------
Dataset load_bvecs(const std::string& path, int max_n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    int32_t dim = 0;
    f.read(reinterpret_cast<char*>(&dim), 4);
    f.seekg(0);

    int bytes_per_vec = 4 + dim;
    f.seekg(0, std::ios::end);
    std::streamsize file_size = f.tellg();
    f.seekg(0);

    int total_n = static_cast<int>(file_size / bytes_per_vec);
    int n = (max_n > 0) ? std::min(total_n, max_n) : total_n;

    Dataset ds;
    ds.n   = n;
    ds.dim = dim;
    ds.data.resize((size_t)n * dim);

    std::vector<uint8_t> buf(dim);
    for (int i = 0; i < n; ++i) {
        int32_t d;
        f.read(reinterpret_cast<char*>(&d), 4);
        f.read(reinterpret_cast<char*>(buf.data()), dim);
        float* dst = ds.data.data() + (size_t)i * dim;
        for (int j = 0; j < dim; ++j) dst[j] = static_cast<float>(buf[j]);
    }
    return ds;
}

// -------------------------------------------------------
//  .ivecs loader
// -------------------------------------------------------
IDataset load_ivecs(const std::string& path, int max_n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    int32_t dim = 0;
    f.read(reinterpret_cast<char*>(&dim), 4);
    f.seekg(0);

    int bytes_per_vec = 4 + dim * 4;
    f.seekg(0, std::ios::end);
    std::streamsize file_size = f.tellg();
    f.seekg(0);

    int total_n = static_cast<int>(file_size / bytes_per_vec);
    int n = (max_n > 0) ? std::min(total_n, max_n) : total_n;

    IDataset ds;
    ds.n   = n;
    ds.dim = dim;
    ds.data.resize((size_t)n * dim);

    for (int i = 0; i < n; ++i) {
        int32_t d;
        f.read(reinterpret_cast<char*>(&d), 4);
        f.read(reinterpret_cast<char*>(ds.data.data() + (size_t)i * dim), dim * 4);
    }
    return ds;
}

// -------------------------------------------------------
//  .fvecs saver
// -------------------------------------------------------
void save_fvecs(const std::string& path, const Dataset& ds) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write: " + path);
    int32_t dim = ds.dim;
    for (int i = 0; i < ds.n; ++i) {
        f.write(reinterpret_cast<const char*>(&dim), 4);
        f.write(reinterpret_cast<const char*>(ds.row(i)), dim * 4);
    }
}

// -------------------------------------------------------
//  Pretty print
// -------------------------------------------------------
void print_dataset_info(const std::string& name, const Dataset& ds) {
    double mb = (double)ds.data.size() * 4 / (1024.0 * 1024.0);
    std::cout << "[dataset] " << name
              << "  n=" << ds.n
              << "  dim=" << ds.dim
              << "  mem=" << std::fixed << std::setprecision(1) << mb << " MB\n";
}

} // namespace ann