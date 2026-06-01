#pragma once
#include "distance.h"
#include <vector>
#include <string>
#include <random>
#include <mutex>
#include <atomic>
#include <queue>
#include <unordered_set>
#include <functional>

namespace ann {

// -------------------------------------------------------
//  Configuration
// -------------------------------------------------------
struct HNSWConfig {
    int    M              = 16;    // max neighbors per node (layers > 0)
    int    M_max0         = 32;    // max neighbors at layer 0  (usually 2*M)
    int    ef_construction = 200;  // beam width during build
    int    ef_search       = 50;   // beam width during query (tunable after build)
    int    max_layers      = 16;   // hard cap on layer count
    Metric metric          = Metric::L2;
    unsigned seed          = 42;
};

// -------------------------------------------------------
//  Candidate pair: (distance, node_id)
// -------------------------------------------------------
using Candidate = std::pair<float, int>;

struct MaxHeap {   // max on top — used for result set
    using type = std::priority_queue<Candidate>;
};
struct MinHeap {   // min on top — used for candidates to explore
    using type = std::priority_queue<Candidate,
                                     std::vector<Candidate>,
                                     std::greater<Candidate>>;
};

// -------------------------------------------------------
//  HNSW index
// -------------------------------------------------------
class HNSW {
public:
    // Construct empty index
    explicit HNSW(int dim, HNSWConfig cfg = {});

    // Build from a flat float array: data[i*dim .. (i+1)*dim - 1]
    void build(const float* data, int n,
               std::function<void(int,int)> progress_cb = nullptr);

    // Insert a single vector (thread-unsafe; fine for sequential build)
    void insert(const float* vec);

    // K-NN query. Returns up to k node IDs (approximate).
    std::vector<int> search(const float* query, int k) const;

    // K-NN query that also returns distances
    std::vector<Candidate> search_with_distances(const float* query, int k) const;

    // Save / load index to binary file
    void save(const std::string& path) const;
    void load(const std::string& path);

    // Accessors
    int  size()       const { return (int)vecs_.size() / dim_; }
    int  dim()        const { return dim_; }
    int  num_layers() const { return max_layer_ + 1; }
    void set_ef_search(int ef) { cfg_.ef_search = ef; }
    int  get_ef_search()  const { return cfg_.ef_search; }
    const HNSWConfig& config() const { return cfg_; }

    // Get raw vector for node i
    const float* vec_ptr(int i) const {
        return vecs_.data() + (size_t)i * dim_;
    }

private:
    // ---- Core algorithm internals ----

    // Greedy beam search at one layer
    // Returns up to ef candidates as a MAX-heap
    MaxHeap::type search_layer(const float* query,
                               int          ep_id,
                               int          ef,
                               int          layer) const;

    // SELECT-NEIGHBORS (simple — take M closest)
    std::vector<int> select_neighbors_simple(MaxHeap::type candidates,
                                             int           M) const;

    // SELECT-NEIGHBORS (heuristic — Algorithm 4 from paper)
    // Prefers diverse neighbors that improve graph connectivity
    std::vector<int> select_neighbors_heuristic(MaxHeap::type candidates,
                                                int           M,
                                                int           layer,
                                                bool          extend_cands = true,
                                                bool          keep_pruned  = true) const;

    // Shrink an existing neighbor list to M using heuristic
    void shrink_neighbors(std::vector<int>& nbrs, int node_id,
                          int M, int layer);

    // Assign a random layer using geometric distribution (paper §4.1)
    int random_layer();

    // ---- Storage ----

    int         dim_;
    HNSWConfig  cfg_;
    DistFn      dist_fn_;

    // Flat vector storage — cache friendly
    std::vector<float> vecs_;  // size = n * dim_

    // adj_[node_id][layer] = sorted neighbor ids
    std::vector<std::vector<std::vector<int>>> adj_;

    int entry_point_ = -1;
    int max_layer_   = -1;

    mutable std::mt19937 rng_;

    // Serialization magic bytes
    static constexpr uint32_t MAGIC = 0xABCD1234;
    static constexpr uint32_t VERSION = 2;
};

// -------------------------------------------------------
//  Brute-force exact search (for recall evaluation)
// -------------------------------------------------------
std::vector<int> brute_force_search(const float* db,  int n_db,
                                    const float* query,
                                    int k, int dim,
                                    Metric metric = Metric::L2);

// -------------------------------------------------------
//  Evaluation helpers
// -------------------------------------------------------

// recall@k: fraction of gt neighbors found in approx result
float compute_recall(const std::vector<int>& approx,
                     const int*              gt,
                     int                     k);

// Run full recall evaluation over a query set
struct RecallResult {
    float mean_recall;
    float min_recall;
    float max_recall;
    double qps;
    int ef_search;
};

RecallResult evaluate_recall(HNSW&        index,
                             const float* queries, int n_q,
                             const int*   gt,
                             int          k,
                             int          ef_search);

} // namespace ann