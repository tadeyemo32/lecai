CXX      := clang++
CXXFLAGS := -std=c++23 -O3 -march=native -Isrc
LDFLAGS  := -lsqlite3 -framework Accelerate

BIN := lecai

SRC := src/main.cpp

$(BIN): $(SRC) src/db.hpp src/bm25.hpp
	$(CXX) $(CXXFLAGS) $(SRC) $(LDFLAGS) -o $(BIN)

run: $(BIN)
	./$(BIN) bm25   "Formula One world championship"
	./$(BIN) semantic "Formula One world championship"
	./$(BIN) hybrid "Formula One world championship"

eval: $(BIN)
	python3 eval.py

ui:
	cd ui && npm start

clean:
	rm -f $(BIN)

.PHONY: run ui clean
