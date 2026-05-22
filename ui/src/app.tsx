import React, { useState, useCallback } from "react";
import { render, Box, Text, useInput, useApp, useStdin } from "ink";
import TextInput from "ink-text-input";
import { execFileSync } from "child_process";
import { resolve } from "path";

type Mode = "bm25" | "semantic" | "hybrid";

interface Result {
  rank: number;
  docId: string;
  title: string;
  score: string;
}

interface SearchState {
  results: Result[];
  latency: string;
  error: string | null;
}

const MODES: Mode[] = ["bm25", "semantic", "hybrid"];
const MODE_LABELS: Record<Mode, string> = {
  bm25:     "BM25     (keyword)",
  semantic: "Semantic (embeddings)",
  hybrid:   "Hybrid   (RRF)",
};
const MODE_COLORS: Record<Mode, string> = {
  bm25:     "cyan",
  semantic: "magenta",
  hybrid:   "green",
};

const BIN = resolve("../lecai");

function runSearch(mode: Mode, query: string): SearchState {
  try {
    const out = execFileSync(BIN, [mode, query], {
      encoding: "utf8",
      timeout: 10_000,
      cwd: resolve(".."),
    });

    // Parse latency line:  "  latency: 7.16701 us"
    const latMatch = out.match(/latency:\s*([\d.]+)\s*(\w+)/);
    const latency  = latMatch ? `${parseFloat(latMatch[1]).toFixed(1)} ${latMatch[2]}` : "?";

    // Parse result lines: "  1. [232] Formula One  (score=14.03)"
    const results: Result[] = [];
    const lineRe = /^\s+(\d+)\.\s+\[(\d+)\]\s+(.+?)\s+\(score=([\d.e+\-]+)\)/gm;
    let m: RegExpExecArray | null;
    while ((m = lineRe.exec(out)) !== null) {
      results.push({ rank: +m[1], docId: m[2], title: m[3], score: parseFloat(m[4]).toFixed(4) });
    }

    return { results, latency, error: null };
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return { results: [], latency: "—", error: msg.slice(0, 120) };
  }
}

function ModeTab({ mode, active }: { mode: Mode; active: boolean }) {
  return (
    <Box marginRight={2}>
      <Text
        color={active ? MODE_COLORS[mode] : "gray"}
        bold={active}
        underline={active}
      >
        {active ? "▶ " : "  "}{MODE_LABELS[mode]}
      </Text>
    </Box>
  );
}

function ResultRow({ r, color }: { r: Result; color: string }) {
  return (
    <Box>
      <Text color="gray">{String(r.rank).padStart(2)}. </Text>
      <Text color={color} bold>{r.title.padEnd(42).slice(0, 42)}</Text>
      <Text color="gray">  score </Text>
      <Text>{r.score}</Text>
      <Text color="gray">  id={r.docId}</Text>
    </Box>
  );
}

function App() {
  const { exit } = useApp();
  const { isRawModeSupported } = useStdin();
  const [modeIdx, setModeIdx]   = useState(0);
  const [query, setQuery]       = useState("");
  const [state, setState]       = useState<SearchState | null>(null);
  const [searching, setSearching] = useState(false);

  const mode = MODES[modeIdx]!;

  useInput((input, key) => {
    if (key.ctrl && input === "c") { exit(); return; }
    if (key.tab || (key.rightArrow && !query)) {
      setModeIdx(i => (i + 1) % MODES.length);
      setState(null);
    }
    if (key.leftArrow && !query) {
      setModeIdx(i => (i + MODES.length - 1) % MODES.length);
      setState(null);
    }
  }, { isActive: isRawModeSupported });

  const handleSubmit = useCallback((q: string) => {
    const trimmed = q.trim();
    if (!trimmed) return;
    setSearching(true);
    setState(null);
    // Run synchronously — execFileSync blocks, but for a CLI tool this is fine
    const result = runSearch(mode, trimmed);
    setState(result);
    setSearching(false);
  }, [mode]);

  const color = MODE_COLORS[mode];

  return (
    <Box flexDirection="column" padding={1}>

      {/* Header */}
      <Box marginBottom={1}>
        <Text bold color="white">⚡ LEC AI  </Text>
        <Text color="gray">Sports Retrieval  ·  240 docs  ·  nomic-embed-text 768d</Text>
      </Box>

      {/* Mode tabs */}
      <Box marginBottom={1}>
        <Text color="gray">Mode (Tab/←→): </Text>
        {MODES.map(m => <ModeTab key={m} mode={m} active={m === mode} />)}
      </Box>

      {/* Search input */}
      <Box borderStyle="round" borderColor={color} paddingX={1} marginBottom={1}>
        <Text color={color} bold>› </Text>
        {isRawModeSupported ? (
          <TextInput
            value={query}
            onChange={setQuery}
            onSubmit={handleSubmit}
            placeholder="type a query and press Enter…"
          />
        ) : (
          <Text color="gray">Run in a real terminal to enable interactive input.</Text>
        )}
      </Box>

      {/* Status / results */}
      {searching && (
        <Text color="yellow">  Searching…</Text>
      )}

      {state && !state.error && (
        <Box flexDirection="column">
          <Box marginBottom={1}>
            <Text color="gray">  Top-5  ·  latency </Text>
            <Text color="green" bold>{state.latency}</Text>
            <Text color="gray">  ·  mode </Text>
            <Text color={color} bold>{mode.toUpperCase()}</Text>
          </Box>
          {state.results.length === 0
            ? <Text color="yellow">  No results found.</Text>
            : state.results.map(r => <ResultRow key={r.rank} r={r} color={color} />)
          }
        </Box>
      )}

      {state?.error && (
        <Box flexDirection="column">
          <Text color="red">  Error: {state.error}</Text>
          <Text color="gray">  Make sure `make` was run from the project root first.</Text>
        </Box>
      )}

      {/* Footer */}
      <Box marginTop={1}>
        <Text color="gray" dimColor>Enter=search  Tab=cycle mode  Ctrl+C=quit</Text>
      </Box>

    </Box>
  );
}

if (!process.stdin.isTTY) {
  console.log("lecai UI requires an interactive terminal. Run: make ui");
  process.exit(0);
}

render(<App />);
