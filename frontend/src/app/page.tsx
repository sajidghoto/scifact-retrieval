
"use client";

import { useEffect, useState } from "react";
import type { FormEvent } from "react";

type SearchResult = {
  doc_id: string;
  title: string;
  text: string;
  score: number;
};

type Stats = {
  documents: number;
  chunks: number;
};

const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<"bm25" | "vector" | "hybrid">("bm25");

  useEffect(() => {
    fetch(`${apiUrl}/stats`)
      .then(async (response) => {
        if (!response.ok) throw new Error("Dataset statistics are unavailable.");
        return response.json() as Promise<Stats>;
      })
      .then(setStats)
      .catch(() => setStats(null));
  }, []);

  async function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError("");
    try {
      const response = await fetch(
        `${apiUrl}/search?query=${encodeURIComponent(query.trim())}&limit=10&mode=${mode}`,
      );
      const payload = (await response.json()) as
        | { results: SearchResult[] }
        | { detail: string };
      if (!response.ok || !("results" in payload)) {
        throw new Error("detail" in payload ? payload.detail : "Search failed.");
      }
      setResults(payload.results);
    } catch (reason) {
      setResults([]);
      setError(reason instanceof Error ? reason.message : "Search failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-5xl flex-col gap-10 px-6 py-16">
      <header className="max-w-3xl space-y-4">
        <p className="text-sm font-medium uppercase tracking-widest text-sky-700">
          Research Engineering
        </p>
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
          SciFact Retrieval System
        </h1>
        <p className="text-lg text-gray-600">
          Search the locally ingested SciFact corpus with BM25, vector, or
          frozen hybrid retrieval.
        </p>
      </header>

      <section className="space-y-4">
        <form className="flex flex-col gap-3 sm:flex-row" onSubmit={submitSearch}>
          <label className="sr-only" htmlFor="query">
            Search query
          </label>
          <input
            className="min-h-12 flex-1 rounded-lg border border-gray-300 bg-white px-4 text-gray-900 shadow-sm outline-none ring-sky-600 focus:ring-2"
            id="query"
            onChange={(event) => setQuery(event.target.value)}
            placeholder="e.g. What causes respiratory infections?"
            value={query}
          />
          <button
            className="min-h-12 rounded-lg bg-sky-700 px-6 font-semibold text-white hover:bg-sky-800 disabled:cursor-not-allowed disabled:opacity-60"
            disabled={loading || !query.trim()}
            type="submit"
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </form>
        <label className="flex items-center gap-2 text-sm text-gray-600" htmlFor="mode">
          Retrieval mode
          <select
            className="rounded border border-gray-300 bg-white px-2 py-1 text-gray-900"
            id="mode"
            onChange={(event) =>
              setMode(event.target.value as "bm25" | "vector" | "hybrid")
            }
            value={mode}
          >
            <option value="bm25">BM25</option>
            <option value="vector">Vector</option>
            <option value="hybrid">Hybrid</option>
          </select>
        </label>
        {stats && (
          <p className="text-sm text-gray-500">
            {stats.documents.toLocaleString()} documents ·{" "}
            {stats.chunks.toLocaleString()} chunks indexed
          </p>
        )}
        {error && <p className="text-sm text-red-700">{error}</p>}
      </section>

      <section aria-live="polite" className="space-y-5">
        {results.map((result) => (
          <article className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm" key={result.doc_id}>
            <div className="flex items-start justify-between gap-4">
              <h2 className="text-lg font-semibold">{result.title || "Untitled document"}</h2>
              <span className="shrink-0 text-sm text-gray-500">
                {result.score.toFixed(3)}
              </span>
            </div>
            <p className="mt-1 text-xs text-gray-500">Document {result.doc_id}</p>
            <p className="mt-3 line-clamp-4 text-gray-700">{result.text}</p>
          </article>
        ))}
        {!loading && !error && query && results.length === 0 && (
          <p className="text-gray-600">No matching documents found.</p>
        )}
      </section>
    </main>
  );
}
