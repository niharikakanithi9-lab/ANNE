#include "hnsw.h"
#include "io.h"
#include <CLI/CLI.hpp>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>

using namespace ann;

// -------------------------------------------------------
//  Progress bar helper
// -------------------------------------------------------
static void print_progress(int done, int total) {
    int pct   = done * 100 / total;
    int width = 40;
    int filled= width * done / total;
    std::cout << "\r  [";
    for (int i = 0; i < filled; ++i) std::cout << '=';
    if (filled < width) { std::cout << '>'; ++filled; }
    for (int i = filled; i < width; ++i) std::cout << ' ';
    std::cout << "] " << std::setw(3) << pct << "%  "
              << done << "/" << total;
    std::cout.flush();
    if (done == total) std::cout << '\n';
}

// -------------------------------------------------------
//  subcommand: build
// -------------------------------------------------------
int cmd_build(int argc, char** argv) {
    CLI::App app{"Build an HNSW index from a .fvecs dataset"};

    std::string data_path, out_path, metric_str = "l2";
    int M = 16, ef_c = 200, max_n = -1;

    app.add_option("--data", data_path,   "Input .fvecs file")->required();
    app.add_option("--out",  out_path,    "Output index file")->required();
    app.add_option("--M",    M,           "Max neighbors per node (default 16)");
    app.add_option("--ef-construction", ef_c, "Build beam width (default 200)");
    app.add_option("--metric", metric_str,"Distance metric: l2|cosine|dot");
    app.add_option("--max-n", max_n,      "Limit vectors to load (default: all)");

    try { app.parse(argc, argv); }
    catch (const CLI::ParseError& e) { return app.exit(e); }

    std::cout << "Loading dataset...\n";
    Dataset ds = load_fvecs(data_path, max_n);
    print_dataset_info(data_path, ds);

    HNSWConfig cfg;
    cfg.M              = M;
    cfg.M_max0         = 2 * M;
    cfg.ef_construction= ef_c;
    cfg.metric         = metric_from_string(metric_str);

    HNSW index(ds.dim, cfg);

    std::cout << "Building HNSW index (M=" << M
              << ", ef_construction=" << ef_c << ")...\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    index.build(ds.data.data(), ds.n, print_progress);
    auto t1 = std::chrono::high_resolution_clock::now();

    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "  Built in " << std::fixed << std::setprecision(1)
              << secs << "s  (" << ds.n << " vectors, "
              << index.num_layers() << " layers)\n";

    index.save(out_path);
    std::cout << "Index saved to " << out_path << "\n";
    return 0;
}

// -------------------------------------------------------
//  subcommand: query
// -------------------------------------------------------
int cmd_query(int argc, char** argv) {
    CLI::App app{"Run K-NN queries against a built index"};

    std::string index_path, query_path;
    int k = 10, ef = 50, n_print = 5;

    app.add_option("--index", index_path, "Index file")->required();
    app.add_option("--query", query_path, "Query .fvecs file")->required();
    app.add_option("--k",     k,          "Number of neighbors (default 10)");
    app.add_option("--ef",    ef,         "ef_search (default 50)");
    app.add_option("--print", n_print,    "Number of results to print");

    try { app.parse(argc, argv); }
    catch (const CLI::ParseError& e) { return app.exit(e); }

    std::cout << "Loading index from " << index_path << "...\n";
    HNSW index(1);  // dim overwritten by load
    index.load(index_path);
    std::cout << "  Loaded: n=" << index.size()
              << " dim=" << index.dim() << "\n";

    Dataset qs = load_fvecs(query_path);
    print_dataset_info("queries", qs);

    index.set_ef_search(ef);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int q = 0; q < qs.n; ++q) {
        auto res = index.search_with_distances(qs.row(q), k);

        if (q < n_print) {
            std::cout << "Query " << q << ": ";
            for (auto& [d, id] : res)
                std::cout << id << "(d=" << std::fixed
                          << std::setprecision(1) << d << ") ";
            std::cout << '\n';
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double qps = qs.n / (ms / 1000.0);

    std::cout << std::fixed << std::setprecision(0)
              << "Throughput: " << qps << " QPS ("
              << qs.n << " queries in "
              << std::setprecision(1) << ms << " ms)\n";
    return 0;
}

// -------------------------------------------------------
//  subcommand: bench
// -------------------------------------------------------
int cmd_bench(int argc, char** argv) {
    CLI::App app{"Benchmark recall@k vs QPS by sweeping ef_search"};

    std::string index_path, query_path, gt_path;
    int k = 10;
    std::vector<int> ef_values = {10, 20, 40, 80, 120, 200, 300, 500};

    app.add_option("--index", index_path, "Index file")->required();
    app.add_option("--query", query_path, "Query .fvecs")->required();
    app.add_option("--gt",    gt_path,    "Ground truth .ivecs")->required();
    app.add_option("--k",     k,          "Recall@k (default 10)");
    app.add_option("--ef",    ef_values,  "ef_search values to sweep");

    try { app.parse(argc, argv); }
    catch (const CLI::ParseError& e) { return app.exit(e); }

    std::cout << "Loading index...\n";
    HNSW index(1);
    index.load(index_path);
    std::cout << "  n=" << index.size() << "  dim=" << index.dim() << "\n\n";

    Dataset  qs = load_fvecs(query_path);
    IDataset gt = load_ivecs(gt_path);

    if (qs.n != gt.n)
        throw std::runtime_error("Query and ground-truth row count mismatch");

    print_dataset_info("queries", qs);

    std::cout << "\n";
    std::cout << std::setw(8)  << "ef_search"
              << std::setw(14) << "recall@" + std::to_string(k)
              << std::setw(12) << "QPS"
              << std::setw(12) << "ms/query"
              << "\n";
    std::cout << std::string(46, '-') << "\n";

    for (int ef : ef_values) {
        RecallResult r = evaluate_recall(
            index,
            qs.data.data(), qs.n,
            gt.data.data(),
            k, ef
        );
        double ms_per_q = 1000.0 / r.qps;
        std::cout << std::setw(8)  << ef
                  << std::setw(14) << std::fixed << std::setprecision(4) << r.mean_recall
                  << std::setw(12) << std::setprecision(0) << r.qps
                  << std::setw(12) << std::setprecision(3) << ms_per_q
                  << "\n";
    }

    // Also run brute-force baseline
    std::cout << "\n[Brute force baseline]\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    float bf_recall = 0.f;
    for (int q = 0; q < qs.n; ++q) {
        auto res = brute_force_search(
            index.vec_ptr(0), index.size(),
            qs.row(q), k, qs.dim, index.config().metric
        );
        bf_recall += compute_recall(res, gt.row(q), k);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double bf_qps = qs.n / (ms / 1000.0);
    std::cout << "  recall@" << k << ": " << bf_recall / qs.n
              << "  QPS: " << std::setprecision(0) << bf_qps << "\n";
    return 0;
}

// -------------------------------------------------------
//  subcommand: info
// -------------------------------------------------------
int cmd_info(int argc, char** argv) {
    CLI::App app{"Print info about a saved index"};

    std::string index_path;
    app.add_option("--index", index_path, "Index file")->required();

    try { app.parse(argc, argv); }
    catch (const CLI::ParseError& e) { return app.exit(e); }

    HNSW index(1);
    index.load(index_path);

    const auto& cfg = index.config();
    std::cout << "=== ann-engine index info ===\n"
              << "  Vectors   : " << index.size()       << "\n"
              << "  Dimension : " << index.dim()        << "\n"
              << "  Layers    : " << index.num_layers() << "\n"
              << "  M         : " << cfg.M              << "\n"
              << "  M_max0    : " << cfg.M_max0         << "\n"
              << "  ef_constr : " << cfg.ef_construction<< "\n"
              << "  ef_search : " << cfg.ef_search      << "\n"
              << "  Metric    : " << metric_to_string(cfg.metric) << "\n"
              << "  Memory    : "
              << std::fixed << std::setprecision(1)
              << ((double)index.size() * index.dim() * 4 / 1024.0 / 1024.0)
              << " MB (vectors only)\n";
    return 0;
}

// -------------------------------------------------------
//  main dispatcher
// -------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ann-engine <subcommand> [options]\n"
                  << "  build   Build an HNSW index\n"
                  << "  query   Run K-NN queries\n"
                  << "  bench   Benchmark recall vs QPS\n"
                  << "  info    Print index metadata\n\n"
                  << "Run: ann-engine <subcommand> --help\n";
        return 1;
    }

    std::string cmd = argv[1];
    // shift argv for subcommand parsing
    int   sub_argc = argc - 1;
    char** sub_argv = argv + 1;

    try {
        if      (cmd == "build") return cmd_build(sub_argc, sub_argv);
        else if (cmd == "query") return cmd_query(sub_argc, sub_argv);
        else if (cmd == "bench") return cmd_bench(sub_argc, sub_argv);
        else if (cmd == "info")  return cmd_info (sub_argc, sub_argv);
        else {
            std::cerr << "Unknown subcommand: " << cmd << "\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}