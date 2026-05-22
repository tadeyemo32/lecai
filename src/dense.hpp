#pragma once
#include <Accelerate/Accelerate.h>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "bm25.hpp"   // reuse SearchResult

// Cosine similarity via Accelerate (SIMD vectorised, free on macOS)
// Both vectors must be L2-normalised — then cosine = dot product
static inline float dot_f32(const float* a, const float* b, int n) {
    float result = 0.0f;
    vDSP_dotpr(a, 1, b, 1, &result, static_cast<vDSP_Length>(n));
    return result;
}

static inline void l2_normalise(float* v, int n) {
    float sq = 0.0f;
    vDSP_svesq(v, 1, &sq, static_cast<vDSP_Length>(n));
    float norm = std::sqrt(sq);
    if (norm > 1e-9f) {
        float inv = 1.0f / norm;
        vDSP_vsmul(v, 1, &inv, v, 1, static_cast<vDSP_Length>(n));
    }
}

class DenseIndex {
public:
    explicit DenseIndex(int dim) : dim_(dim) {}

    // Add a pre-fetched embedding (will be L2-normalised in place)
    void add(int doc_id, const std::string& title, std::vector<float> vec) {
        if (static_cast<int>(vec.size()) != dim_)
            throw std::runtime_error("Embedding dimension mismatch");
        l2_normalise(vec.data(), dim_);
        doc_ids_.push_back(doc_id);
        titles_.push_back(title);
        // Store contiguously for cache-friendly scan
        matrix_.insert(matrix_.end(), vec.begin(), vec.end());
    }

    // Returns top-k by cosine similarity
    std::vector<SearchResult> search(const std::vector<float>& raw_query, int k = 5) {
        if (doc_ids_.empty()) return {};

        // Normalise query once
        query_buf_ = raw_query;
        l2_normalise(query_buf_.data(), dim_);

        int N = static_cast<int>(doc_ids_.size());
        scores_.resize(N);

        // Single matrix multiply: scores[N×1] = matrix[N×dim] · query[dim×1]
        // Accelerate dispatches this across all CPU cores internally.
        vDSP_mmul(matrix_.data(), 1,
                  query_buf_.data(), 1,
                  scores_.data(), 1,
                  static_cast<vDSP_Length>(N),
                  1,
                  static_cast<vDSP_Length>(dim_));

        // Partial sort indices
        idx_buf_.resize(N);
        std::iota(idx_buf_.begin(), idx_buf_.end(), 0);
        int take = std::min(k, N);
        std::partial_sort(idx_buf_.begin(), idx_buf_.begin() + take, idx_buf_.end(),
            [this](int a, int b){ return scores_[a] > scores_[b]; });

        std::vector<SearchResult> results;
        results.reserve(take);
        for (int i = 0; i < take; i++) {
            int idx = idx_buf_[i];
            results.push_back({doc_ids_[idx], idx, titles_[idx],
                               static_cast<double>(scores_[idx])});
        }
        return results;
    }

    int size() const { return static_cast<int>(doc_ids_.size()); }

private:
    int                  dim_;
    std::vector<int>     doc_ids_;
    std::vector<std::string> titles_;
    std::vector<float>   matrix_;     // [N * dim] row-major, L2-normalised
    std::vector<float>   scores_;     // scratch
    std::vector<float>   query_buf_;  // scratch
    std::vector<int>     idx_buf_;    // scratch
};
