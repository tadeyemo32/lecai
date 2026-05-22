#pragma once
#include "bm25.hpp"
#include <vector>
#include <unordered_map>
#include <algorithm>

// Reciprocal Rank Fusion: score(d) = sum_r 1/(k + rank_r(d))
// k=60 is the standard constant that dampens the effect of very high rankings
static constexpr double RRF_K = 60.0;

inline std::vector<SearchResult> rrf_fuse(
    const std::vector<SearchResult>& bm25_results,
    const std::vector<SearchResult>& dense_results,
    const std::vector<std::pair<int,std::string>>& all_docs,  // (doc_id, title)
    int k = 5)
{
    // Accumulate RRF scores keyed by doc_id
    std::unordered_map<int, double> rrf_scores;
    rrf_scores.reserve(bm25_results.size() + dense_results.size());

    for (int rank = 0; rank < static_cast<int>(bm25_results.size()); rank++)
        rrf_scores[bm25_results[rank].doc_id] += 1.0 / (RRF_K + rank + 1);

    for (int rank = 0; rank < static_cast<int>(dense_results.size()); rank++)
        rrf_scores[dense_results[rank].doc_id] += 1.0 / (RRF_K + rank + 1);

    // Build title lookup from all_docs
    std::unordered_map<int, std::string> titles;
    titles.reserve(all_docs.size());
    for (auto& [id, title] : all_docs) titles[id] = title;

    std::vector<SearchResult> fused;
    fused.reserve(rrf_scores.size());
    for (auto& [id, score] : rrf_scores)
        fused.push_back({id, -1, titles.count(id) ? titles.at(id) : "", score});

    int take = std::min(k, static_cast<int>(fused.size()));
    std::partial_sort(fused.begin(), fused.begin() + take, fused.end(),
        [](const SearchResult& a, const SearchResult& b){ return a.score > b.score; });
    fused.resize(take);
    return fused;
}
