#include "hnsw.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace ann {

// -------------------------------------------------------
//  Constructor
// -------------------------------------------------------
HNSW::HNSW(int dim, HNSWConfig cfg)
    : dim_(dim), cfg_(cfg), rng_(cfg.seed) {
    if (cfg_.M_max0 <= 0) cfg_.M_max0 = 2 * cfg_.M;
    dist_fn_ = get_dist_fn(cfg_.metric);
}

// -------------------------------------------------------
//  random_layer
//  Geometric distribution: P(layer >= L) = exp(-L / ln(M))
// -------------------------------------------------------
int HNSW::random_layer() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double ml = 1.0 / std::log(static_cast<double>(cfg_.M));
    int layer = static_cast<int>(-std::log(dist(rng_)) * ml);
    return std::min(layer, cfg_.max_layers - 1);
}

// -------------------------------------------------------
//  search_layer
//  Greedy beam search at one layer.
//  Returns a MAX-heap of (distance, id) with up to ef entries.
// -------------------------------------------------------
MaxHeap::type HNSW::search_layer(const float* query,
                                  int          ep_id,
                                  int          ef,
                                  int          layer) const {
    // visited array via generation counter — O(1) mark/check
    // We use a flat vector that lives in the hot path
    static thread_local std::vector<uint32_t> visited_gen;
    static thread_local uint32_t              generation = 0;

    int n = size();
    if ((int)visited_gen.size() < n) visited_gen.assign(n, 0);
    ++generation;
    if (generation == 0) {
        std::fill(visited_gen.begin(), visited_gen.end(), 0);
        ++generation;
    }

    float d0 = dist_fn_(query, vec_ptr(ep_id), dim_);
    visited_gen[ep_id] = generation;

    MaxHeap::type results;   // max-heap: farthest on top
    MinHeap::type cands;     // min-heap: closest on top

    results.push({d0, ep_id});
    cands.push({d0, ep_id});

    while (!cands.empty()) {
        auto [cd, cid] = cands.top();
        cands.pop();

        // Pruning: if closest candidate is farther than worst result, stop
        if (cd > results.top().first && (int)results.size() >= ef)
            break;

        const auto& neighbors = adj_[cid][layer];
        for (int nbr : neighbors) {
            if (nbr < 0 || nbr >= n) continue;
            if (visited_gen[nbr] == generation) continue;
            visited_gen[nbr] = generation;

            float nd = dist_fn_(query, vec_ptr(nbr), dim_);

            if ((int)results.size() < ef || nd < results.top().first) {
                cands.push({nd, nbr});
                results.push({nd, nbr});
                if ((int)results.size() > ef)
                    results.pop();  // evict farthest
            }
        }
    }
    return results;
}

// -------------------------------------------------------
//  select_neighbors_simple
//  Takes M closest from a max-heap
// -------------------------------------------------------
std::vector<int> HNSW::select_neighbors_simple(MaxHeap::type candidates,
                                                int           M) const {
    // Drain max-heap into sorted vector (closest first after reverse)
    std::vector<Candidate> tmp;
    tmp.reserve(candidates.size());
    while (!candidates.empty()) {
        tmp.push_back(candidates.top());
        candidates.pop();
    }
    // Max-heap has farthest on top; we want closest M
    // Sort ascending by distance
    std::sort(tmp.begin(), tmp.end(),
              [](const Candidate& a, const Candidate& b){ return a.first < b.first; });

    std::vector<int> result;
    int take = std::min((int)tmp.size(), M);
    result.reserve(take);
    for (int i = 0; i < take; ++i) result.push_back(tmp[i].second);
    return result;
}

// -------------------------------------------------------
//  select_neighbors_heuristic
//  Algorithm 4 from the HNSW paper.
//  Prefers diverse neighbors that improve long-range connectivity.
// -------------------------------------------------------
std::vector<int> HNSW::select_neighbors_heuristic(MaxHeap::type candidates,
                                                   int           M,
                                                   int           layer,
                                                   bool          extend_cands,
                                                   bool          keep_pruned) const {
    // Collect candidates into a min-heap (closest first)
    MinHeap::type W;
    while (!candidates.empty()) {
        W.push({candidates.top().first, candidates.top().second});
        candidates.pop();
    }

    // Optionally extend with neighbors of candidates
    if (extend_cands) {
        // copy W to iterate (can't iterate priority_queue)
        auto W_copy = W;
        while (!W_copy.empty()) {
            int eid = W_copy.top().second; W_copy.pop();
            if (eid < 0 || eid >= size()) continue;
            for (int nbr : adj_[eid][layer]) {
                if (nbr < 0 || nbr >= size()) continue;
                W.push({dist_fn_(vec_ptr(W.top().second), vec_ptr(nbr), dim_), nbr});
            }
        }
    }

    std::vector<int> result;
    result.reserve(M);
    MinHeap::type discarded;

    while (!W.empty() && (int)result.size() < M) {
        auto [d_cand, e] = W.top(); W.pop();

        // Check if this candidate is closer to the query than to any already-selected neighbor
        bool good = true;
        for (int r : result) {
            float d_to_r = dist_fn_(vec_ptr(e), vec_ptr(r), dim_);
            if (d_to_r < d_cand) {
                good = false;
                break;
            }
        }
        if (good)
            result.push_back(e);
        else
            discarded.push({d_cand, e});
    }

    // Fill remaining slots from discarded if allowed
    if (keep_pruned) {
        while (!discarded.empty() && (int)result.size() < M) {
            result.push_back(discarded.top().second);
            discarded.pop();
        }
    }
    return result;
}

// -------------------------------------------------------
//  shrink_neighbors
//  When bidirectional linking causes a node to exceed M_max,
//  trim its list using the heuristic.
// -------------------------------------------------------
void HNSW::shrink_neighbors(std::vector<int>& nbrs, int node_id,
                             int M, int layer) {
    // Build a max-heap of the existing neighbors
    MaxHeap::type cands;
    for (int nbr : nbrs) {
        float d = dist_fn_(vec_ptr(node_id), vec_ptr(nbr), dim_);
        cands.push({d, nbr});
    }
    nbrs = select_neighbors_simple(std::move(cands), M);
}

// -------------------------------------------------------
//  insert
//  Full Algorithm 1 from the paper.
// -------------------------------------------------------
void HNSW::insert(const float* vec) {
    int id = size();

    // Append vector to flat storage
    vecs_.insert(vecs_.end(), vec, vec + dim_);

    // Ensure adj_ is large enough
    adj_.emplace_back(cfg_.max_layers);

    int node_layer = random_layer();

    // First node: just set as entry point
    if (entry_point_ == -1) {
        entry_point_ = id;
        max_layer_   = node_layer;
        return;
    }

    int ep = entry_point_;

    // Phase 1: Greedy descent from max_layer_ → node_layer+1
    // Use ef=1 (greedy, no beam) for fast descent
    for (int lc = max_layer_; lc > node_layer; --lc) {
        auto res = search_layer(vec, ep, 1, lc);
        if (!res.empty()) ep = res.top().second;
    }

    // Phase 2: Insert at each layer from min(node_layer, max_layer_) → 0
    for (int lc = std::min(node_layer, max_layer_); lc >= 0; --lc) {
        int M_at_layer = cfg_.M;

        auto candidates = search_layer(vec, ep, cfg_.ef_construction, lc);

        // Select M neighbors for new node
        std::vector<int> neighbors =
    select_neighbors_simple(candidates, M_at_layer);

        adj_[id][lc] = neighbors;
        // Bidirectional links: also add id to each neighbor's list
        for (int nbr : neighbors) {
            auto& nbr_list = adj_[nbr][lc];
            nbr_list.push_back(id);

            // Shrink if exceeded capacity
            int cap = (lc == 0) ? cfg_.M_max0 : cfg_.M;
            if ((int)nbr_list.size() > cap) {
                shrink_neighbors(nbr_list, nbr, cap, lc);
            }
        }

        // Move entry point to best candidate for next layer
        if (!candidates.empty()) {
    MaxHeap::type tmp = candidates;
    std::vector<Candidate> v;

    while (!tmp.empty()) {
        v.push_back(tmp.top());
        tmp.pop();
    }

    if (!v.empty())
        ep = v.back().second;
}
}

    // Update entry point if new node has higher layer
    if (node_layer > max_layer_) {
        max_layer_   = node_layer;
        entry_point_ = id;
    }
}

// -------------------------------------------------------
//  build
//  Insert all vectors; optional progress callback
// -------------------------------------------------------
void HNSW::build(const float* data, int n,
                 std::function<void(int,int)> progress_cb) {
    vecs_.reserve((size_t)n * dim_);
    adj_.reserve(n);

    for (int i = 0; i < n; ++i) {
        insert(data + (size_t)i * dim_);
        if (progress_cb && (i % 10000 == 0 || i == n - 1))
            progress_cb(i + 1, n);
    }
}

// -------------------------------------------------------
//  search
//  K-NN query — Algorithm 5 from the paper.
// -------------------------------------------------------
std::vector<int> HNSW::search(const float* query, int k) const {
    if (entry_point_ < 0) return {};

    int ep = entry_point_;

    // Greedy descent: layers max_layer_ → 1 with ef=1
    for (int lc = max_layer_; lc > 0; --lc) {
        auto res = search_layer(query, ep, 1, lc);
        if (!res.empty()) ep = res.top().second;
    }

    // Layer 0: full beam search with ef_search
    auto candidates = search_layer(query, ep, std::max(cfg_.ef_search,k), 0);

    // Extract top-k (drain max-heap, reverse)
    int take = std::min(k, (int)candidates.size());
    std::vector<Candidate> tmp;
    tmp.reserve(candidates.size());
    while (!candidates.empty()) {
        tmp.push_back(candidates.top());
        candidates.pop();
    }
    // tmp is sorted farthest→closest; reverse for closest first
    std::sort(tmp.begin(), tmp.end(),
              [](const Candidate& a, const Candidate& b){ return a.first < b.first; });

    std::vector<int> result;
    result.reserve(take);
    for (int i = 0; i < take; ++i) result.push_back(tmp[i].second);
    return result;
}

// -------------------------------------------------------
//  search_with_distances
// -------------------------------------------------------
std::vector<Candidate> HNSW::search_with_distances(const float* query, int k) const {
    if (entry_point_ < 0) return {};

    int ep = entry_point_;
    for (int lc = max_layer_; lc > 0; --lc) {
        auto res = search_layer(query, ep, 1, lc);
        if (!res.empty()) ep = res.top().second;
    }

    auto candidates = search_layer(query, ep, std::max(cfg_.ef_search,k), 0);
    int take = std::min(k, (int)candidates.size());

    std::vector<Candidate> tmp;
    tmp.reserve(candidates.size());
    while (!candidates.empty()) {
        tmp.push_back(candidates.top());
        candidates.pop();
    }
    std::sort(tmp.begin(), tmp.end(),
              [](const Candidate& a, const Candidate& b){ return a.first < b.first; });
    tmp.resize(take);
    return tmp;
}

// -------------------------------------------------------
//  save / load (binary format)
//  Layout: MAGIC | VERSION | dim | n | metric | cfg | vecs | adj
// -------------------------------------------------------
void HNSW::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write index: " + path);

    auto write = [&](const void* d, size_t bytes){ f.write((const char*)d, bytes); };

    write(&MAGIC,   4);
    write(&VERSION, 4);
    write(&dim_,    4);
    int n = size();
    write(&n, 4);

    int metric_int = static_cast<int>(cfg_.metric);
    write(&metric_int,           4);
    write(&cfg_.M,               4);
    write(&cfg_.M_max0,          4);
    write(&cfg_.ef_construction, 4);
    write(&cfg_.ef_search,       4);
    write(&cfg_.max_layers,      4);
    write(&entry_point_,         4);
    write(&max_layer_,           4);

    // Vectors (flat)
    write(vecs_.data(), vecs_.size() * 4);

    // Adjacency list: for each node, for each layer, write count then ids
    for (int i = 0; i < n; ++i) {
        int nlayers = (int)adj_[i].size();
        write(&nlayers, 4);
        for (int l = 0; l < nlayers; ++l) {
            int cnt = (int)adj_[i][l].size();
            write(&cnt, 4);
            if (cnt > 0) write(adj_[i][l].data(), cnt * 4);
        }
    }
}

void HNSW::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open index: " + path);

    auto read = [&](void* d, size_t bytes){ f.read((char*)d, bytes); };

    uint32_t magic, version;
    read(&magic,   4);
    read(&version, 4);
    if (magic != MAGIC)
        throw std::runtime_error("Bad magic bytes — not an ann-engine index file");
    if (version != VERSION)
        throw std::runtime_error("Index version mismatch (got " +
                                 std::to_string(version) + ", expected " +
                                 std::to_string(VERSION) + ")");

    int n;
    read(&dim_, 4);
    read(&n,    4);

    int metric_int;
    read(&metric_int,           4);
    read(&cfg_.M,               4);
    read(&cfg_.M_max0,          4);
    read(&cfg_.ef_construction, 4);
    read(&cfg_.ef_search,       4);
    read(&cfg_.max_layers,      4);
    read(&entry_point_,         4);
    read(&max_layer_,           4);

    cfg_.metric = static_cast<Metric>(metric_int);
    dist_fn_    = get_dist_fn(cfg_.metric);

    vecs_.resize((size_t)n * dim_);
    read(vecs_.data(), vecs_.size() * 4);

    adj_.resize(n);
    for (int i = 0; i < n; ++i) {
        int nlayers;
        read(&nlayers, 4);
        adj_[i].resize(nlayers);
        for (int l = 0; l < nlayers; ++l) {
            int cnt;
            read(&cnt, 4);
            adj_[i][l].resize(cnt);
            if (cnt > 0) read(adj_[i][l].data(), cnt * 4);
        }
    }
}

// -------------------------------------------------------
//  brute_force_search
// -------------------------------------------------------
std::vector<int> brute_force_search(const float* db,  int n_db,
                                    const float* query,
                                    int k, int dim,
                                    Metric metric) {
    DistFn fn = get_dist_fn(metric);
    std::vector<Candidate> dists;
    dists.reserve(n_db);
    for (int i = 0; i < n_db; ++i)
        dists.push_back({fn(query, db + (size_t)i * dim, dim), i});

    int take = std::min(k, n_db);
    std::partial_sort(dists.begin(), dists.begin() + take, dists.end(),
                      [](const Candidate& a, const Candidate& b){ return a.first < b.first; });

    std::vector<int> result;
    result.reserve(take);
    for (int i = 0; i < take; ++i) result.push_back(dists[i].second);
    return result;
}

// -------------------------------------------------------
//  compute_recall
// -------------------------------------------------------
float compute_recall(const std::vector<int>& approx,
                     const int*              gt,
                     int                     k) {
    // Build ground-truth set
    // gt[0..k-1] are the true nearest neighbors
    int n_gt = std::min(k, (int)approx.size());
    int hits = 0;
    for (int i = 0; i < n_gt; ++i) {
        for (int j = 0; j < k; ++j) {
            if (approx[i] == gt[j]) { ++hits; break; }
        }
    }
    return static_cast<float>(hits) / k;
}

// -------------------------------------------------------
//  evaluate_recall
// -------------------------------------------------------
RecallResult evaluate_recall(HNSW&        index,
                             const float* queries, int n_q,
                             const int*   gt,
                             int          k,
                             int          ef_search,
                             int          gt_stride) {
    index.set_ef_search(ef_search);

    float total = 0.f, min_r = 1.f, max_r = 0.f;
    int gt_k = (gt_stride > 0) ? gt_stride : 100;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int q = 0; q < n_q; ++q) {
        auto res = index.search(queries + (size_t)q * index.dim(), k);
        float r  = compute_recall(res, gt + (size_t)q * gt_k, k);
        total   += r;
        min_r    = std::min(min_r, r);
        max_r    = std::max(max_r, r);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    RecallResult rr;
    rr.mean_recall = total / n_q;
    rr.min_recall  = min_r;
    rr.max_recall  = max_r;
    rr.qps         = n_q / (ms / 1000.0);
    rr.ef_search   = ef_search;
    return rr;
}

} // namespace ann
