#include "db.hpp"
#include "bm25.hpp"
#include "dense.hpp"
#include "hybrid.hpp"
#include "embed_store.hpp"
#include <iostream>
#include <chrono>
#include <string>
#include <future>

static double now_us() {
    using namespace std::chrono;
    return duration<double, std::micro>(
        high_resolution_clock::now().time_since_epoch()).count();
}

static void print_results(const std::vector<SearchResult>& results, double us) {
    std::cout << "  latency: " << us << " us\n";
    for (int i = 0; i < (int)results.size(); i++)
        std::cout << "  " << (i+1) << ". [" << results[i].doc_id << "] "
                  << results[i].title << "  (score=" << results[i].score << ")\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <bm25|semantic|hybrid> \"query\"\n";
        return 1;
    }

    std::string mode  = argv[1];
    std::string query = argv[2];
    std::string db    = "data/corpus.db";

    // ── BM25 ──────────────────────────────────────────────────────────────────
    auto docs = load_documents(db);
    BM25Index bm25;
    for (auto& d : docs) bm25.add(d.id, d.title, d.body);
    bm25.build();

    if (mode == "bm25") {
        bm25.search(query, 5);  // warm-up
        double t0 = now_us();
        auto results = bm25.search(query, 5);
        double t1 = now_us();
        std::cout << "BM25  \"" << query << "\"\n";
        print_results(results, t1 - t0);
        return 0;
    }

    // ── Dense ─────────────────────────────────────────────────────────────────
    std::cout << "Loading embeddings...\n";
    auto dense = load_dense_index(db);

    if (mode == "semantic") {
        auto q_vec = embed_query(query);
        if (q_vec.empty()) { std::cerr << "Ollama embedding failed\n"; return 1; }
        dense.search(q_vec, 5);  // warm-up
        double t0 = now_us();
        auto results = dense.search(q_vec, 5);
        double t1 = now_us();
        std::cout << "Semantic \"" << query << "\"\n";
        print_results(results, t1 - t0);
        return 0;
    }

    // ── Hybrid ────────────────────────────────────────────────────────────────
    if (mode == "hybrid") {
        auto q_vec = embed_query(query);
        if (q_vec.empty()) { std::cerr << "Ollama embedding failed\n"; return 1; }

        // Run BM25 and Dense concurrently — they touch separate data structures
        double t0 = now_us();
        auto bm25_fut  = std::async(std::launch::async, [&]{ return bm25.search(query, 20); });
        auto dense_res = dense.search(q_vec, 20);
        auto bm25_res  = bm25_fut.get();
        std::vector<std::pair<int,std::string>> all_docs;
        all_docs.reserve(docs.size());
        for (auto& d : docs) all_docs.emplace_back(d.id, d.title);
        auto results = rrf_fuse(bm25_res, dense_res, all_docs, 5);
        double t1 = now_us();

        std::cout << "Hybrid \"" << query << "\"\n";
        print_results(results, t1 - t0);
        return 0;
    }

    std::cerr << "Unknown mode: " << mode << "\n";
    return 1;
}
