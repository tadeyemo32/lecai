#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <numeric>

static constexpr double BM25_K1 = 1.5;
static constexpr double BM25_B  = 0.75;

struct SearchResult {
    int         doc_id;    // original DB id
    int         idx;       // 0-based index into corpus
    std::string title;
    double      score;
};

inline std::vector<std::string> tokenise(const std::string& text) {
    std::vector<std::string> tokens;
    tokens.reserve(text.size() / 5);
    std::string tok;
    tok.reserve(16);
    for (unsigned char c : text) {
        if (std::isalpha(c))
            tok += static_cast<char>(std::tolower(c));
        else if (!tok.empty()) {
            tokens.push_back(tok);
            tok.clear();
        }
    }
    if (!tok.empty()) tokens.push_back(tok);
    return tokens;
}

class BM25Index {
public:
    // Compact posting: doc index (not DB id) + term freq + doc length
    struct Posting {
        uint32_t doc_idx;
        uint16_t term_freq;
        uint16_t doc_len;   // capped at 65535 tokens — fine for abstracts
    };

    struct PostingList {
        double               idf;      // precomputed at build time
        std::vector<Posting> entries;
    };

    void add(int doc_id, const std::string& title, const std::string& text) {
        int idx = static_cast<int>(doc_ids_.size());
        doc_ids_.push_back(doc_id);
        titles_.push_back(title);

        auto tokens = tokenise(title + " " + text);
        int len = static_cast<int>(tokens.size());
        doc_lengths_.push_back(len);
        total_len_ += len;

        std::unordered_map<std::string, int> tf;
        tf.reserve(tokens.size());
        for (auto& t : tokens) tf[t]++;

        for (auto& [term, freq] : tf) {
            term_index_[term].entries.push_back({
                static_cast<uint32_t>(idx),
                static_cast<uint16_t>(std::min(freq, 65535)),
                static_cast<uint16_t>(std::min(len, 65535))
            });
        }
    }

    // Must call once after all add() calls — finalises IDF and avgdl
    void build() {
        int N = static_cast<int>(doc_ids_.size());
        avgdl_ = N > 0 ? static_cast<double>(total_len_) / N : 1.0;

        for (auto& [term, pl] : term_index_) {
            double df = static_cast<double>(pl.entries.size());
            pl.idf = std::log((N - df + 0.5) / (df + 0.5) + 1.0);
        }

        // Pre-allocate the score scratch buffer (reused across queries)
        score_buf_.assign(N, 0.0);
    }

    std::vector<SearchResult> search(const std::string& query, int k = 5) const {
        // Reset only touched cells — avoid O(N) memset on every query
        std::fill(score_buf_.begin(), score_buf_.end(), 0.0);

        auto q_terms = tokenise(query);

        for (auto& term : q_terms) {
            auto it = term_index_.find(term);
            if (it == term_index_.end()) continue;

            const PostingList& pl = it->second;
            for (const Posting& p : pl.entries) {
                double tf_norm = (p.term_freq * (BM25_K1 + 1.0)) /
                    (p.term_freq + BM25_K1 * (1.0 - BM25_B + BM25_B * p.doc_len / avgdl_));
                score_buf_[p.doc_idx] += pl.idf * tf_norm;
            }
        }

        // Collect non-zero results
        std::vector<SearchResult> results;
        results.reserve(64);
        int N = static_cast<int>(doc_ids_.size());
        for (int i = 0; i < N; i++) {
            if (score_buf_[i] > 0.0)
                results.push_back({doc_ids_[i], i, titles_[i], score_buf_[i]});
        }

        int take = std::min(k, static_cast<int>(results.size()));
        std::partial_sort(results.begin(), results.begin() + take, results.end(),
            [](const SearchResult& a, const SearchResult& b){ return a.score > b.score; });
        results.resize(take);
        return results;
    }

    int size() const { return static_cast<int>(doc_ids_.size()); }
    int doc_id_at(int idx) const { return doc_ids_[idx]; }
    const std::string& title_at(int idx) const { return titles_[idx]; }

private:
    std::unordered_map<std::string, PostingList> term_index_;
    std::vector<int>         doc_ids_;
    std::vector<std::string> titles_;
    std::vector<int>         doc_lengths_;
    size_t                   total_len_ = 0;
    double                   avgdl_     = 1.0;
    mutable std::vector<double> score_buf_;  // scratch, reused each query
};
