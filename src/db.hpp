#pragma once
#include <sqlite3.h>
#include <string>
#include <vector>
#include <stdexcept>

struct Document {
    int         id;
    std::string title;
    std::string body;
};

inline std::vector<Document> load_documents(const std::string& db_path) {
    sqlite3* db = nullptr;
    if (sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK)
        throw std::runtime_error("Cannot open DB: " + db_path);

    const char* sql = "SELECT id, title, body FROM documents ORDER BY id";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        sqlite3_close(db);
        throw std::runtime_error("Failed to prepare statement");
    }

    std::vector<Document> docs;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Document d;
        d.id    = sqlite3_column_int(stmt, 0);
        d.title = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        d.body  = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        docs.push_back(std::move(d));
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return docs;
}
