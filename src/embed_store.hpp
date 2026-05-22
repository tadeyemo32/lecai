#pragma once
#include <sqlite3.h>
#include <vector>
#include <string>
#include <stdexcept>
#include "dense.hpp"

static constexpr int EMBED_DIM = 768;

// Load all embeddings from SQLite into a DenseIndex.
// Embeddings are stored as raw float32 BLOBs written by embed.py.
inline DenseIndex load_dense_index(const std::string& db_path) {
    sqlite3* db = nullptr;
    if (sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK)
        throw std::runtime_error("Cannot open DB: " + db_path);

    const char* sql = R"(
        SELECT d.id, d.title, e.vector
        FROM embeddings e
        JOIN documents d ON d.id = e.doc_id
        ORDER BY d.id
    )";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        sqlite3_close(db);
        throw std::runtime_error("Failed to prepare embeddings query");
    }

    DenseIndex idx(EMBED_DIM);
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int         doc_id = sqlite3_column_int(stmt, 0);
        std::string title  = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        const void* blob   = sqlite3_column_blob(stmt, 2);
        int         bytes  = sqlite3_column_bytes(stmt, 2);

        int expected = EMBED_DIM * static_cast<int>(sizeof(float));
        if (bytes != expected)
            throw std::runtime_error("Bad blob size for doc " + std::to_string(doc_id));

        std::vector<float> vec(EMBED_DIM);
        std::memcpy(vec.data(), blob, bytes);
        idx.add(doc_id, title, std::move(vec));
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return idx;
}

// Fetch a single embedding from Ollama for a query string (runtime, not cached)
// Returns empty vector on failure.
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <sstream>

inline std::vector<float> embed_query(const std::string& text) {
    // Build HTTP POST body
    std::string escaped;
    escaped.reserve(text.size());
    for (char c : text) {
        if (c == '"')  escaped += "\\\"";
        else if (c == '\\') escaped += "\\\\";
        else if (c == '\n') escaped += "\\n";
        else escaped += c;
    }
    std::string body = R"({"model":"nomic-embed-text","prompt":")" + escaped + "\"}";

    std::string request =
        "POST /api/embeddings HTTP/1.0\r\n"
        "Host: localhost\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "\r\n" + body;

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return {};

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(11434);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(sock); return {};
    }

    send(sock, request.data(), request.size(), 0);

    std::string response;
    char buf[4096];
    ssize_t n;
    while ((n = recv(sock, buf, sizeof(buf), 0)) > 0)
        response.append(buf, n);
    close(sock);

    // Find JSON body after \r\n\r\n
    auto pos = response.find("\r\n\r\n");
    if (pos == std::string::npos) return {};
    std::string json = response.substr(pos + 4);

    // Parse "embedding":[...] — minimal hand-rolled parser
    auto start = json.find("\"embedding\":[");
    if (start == std::string::npos) return {};
    start += 13;
    auto end = json.find(']', start);
    if (end == std::string::npos) return {};

    std::vector<float> vec;
    vec.reserve(EMBED_DIM);
    std::istringstream ss(json.substr(start, end - start));
    std::string token;
    while (std::getline(ss, token, ',')) {
        try { vec.push_back(std::stof(token)); }
        catch (...) {}
    }
    return vec;
}
