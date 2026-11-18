
"use client";

import { useEffect, useState } from "react";
import type { FormEvent } from "react";

type SearchResult = {
  doc_id: string;
  title: string;
  text: string;
  score: number;
};

type RetrievalMode = "bm25" | "vector" | "hybrid";

type GeneratedAnswer = {
  claim: string;
  mode: RetrievalMode;
  verdict: "supports" | "contradicts" | "insufficient";
  cited_doc_ids: string[];
  rationale: string;
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
  const [busyAction, setBusyAction] = useState<"search" | "generate" | null>(null);
  const [generatedAnswer, setGeneratedAnswer] = useState<GeneratedAnswer | null>(null);
  const [mode, setMode] = useState<RetrievalMode>("bm25");

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

    setBusyAction("search");
    setError("");
    setGeneratedAnswer(null);
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
      setBusyAction(null);
    }
  }

  async function submitGeneration() {
    if (!query.trim()) return;

    setBusyAction("generate");
    setError("");
    setGeneratedAnswer(null);
    try {
      const response = await fetch(`${apiUrl}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ claim: query.trim(), mode }),
      });
      const payload = (await response.json()) as
        | (GeneratedAnswer & { results: SearchResult[] })
        | { detail: string };
      if (!response.ok || !("verdict" in payload)) {
        throw new Error(
          "detail" in payload ? payload.detail : "Answer generation failed.",
        );
      }
      setGeneratedAnswer(payload);
      setResults(payload.results);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Answer generation failed.");
    } finally {
      setBusyAction(null);
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
          frozen hybrid retrieval, then generate an evidence-grounded answer
          from the top five results.
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
            onChange={(event) => {
              setQuery(event.target.value);
              setGeneratedAnswer(null);
            }}
            placeholder="e.g. Antibiotics reduce mortality in bacterial infections."
            value={query}
          />
          <button
            className="min-h-12 rounded-lg bg-sky-700 px-6 font-semibold text-white hover:bg-sky-800 disabled:cursor-not-allowed disabled:opacity-60"
            disabled={busyAction !== null || !query.trim()}
            type="submit"
          >
            {busyAction === "search" ? "Searching..." : "Search"}
          </button>
          <button
            className="min-h-12 rounded-lg border border-sky-700 px-6 font-semibold text-sky-800 hover:bg-sky-50 disabled:cursor-not-allowed disabled:opacity-60"
            disabled={busyAction !== null || !query.trim()}
            onClick={submitGeneration}
            type="button"
          >
            {busyAction === "generate" ? "Generating..." : "Generate answer"}
          </button>
        </form>
        <label className="flex items-center gap-2 text-sm text-gray-600" htmlFor="mode">
          Retrieval mode
          <select
            className="rounded border border-gray-300 bg-white px-2 py-1 text-gray-900"
            id="mode"
            onChange={(event) => {
              setMode(event.target.value as RetrievalMode);
              setGeneratedAnswer(null);
            }}
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

      {generatedAnswer && (
        <section
          aria-live="polite"
          className="space-y-3 rounded-xl border border-sky-200 bg-sky-50 p-5"
        >
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-lg font-semibold">Evidence-grounded answer</h2>
            <span className="rounded-full bg-white px-3 py-1 text-sm font-medium capitalize text-sky-900">
              {generatedAnswer.verdict}
            </span>
          </div>
          <p className="text-gray-800">{generatedAnswer.rationale}</p>
          <p className="text-sm text-gray-600">
            Cited documents:{" "}
            {generatedAnswer.cited_doc_ids.length > 0
              ? generatedAnswer.cited_doc_ids.join(", ")
              : "None"}
          </p>
        </section>
      )}

      <section aria-live="polite" className="space-y-5">
        {results.map((result) => (
          <article
            className={`rounded-xl border bg-white p-5 shadow-sm ${
              generatedAnswer?.cited_doc_ids.includes(result.doc_id)
                ? "border-sky-400"
                : "border-gray-200"
            }`}
            key={result.doc_id}
          >
            <div className="flex items-start justify-between gap-4">
              <h2 className="text-lg font-semibold">{result.title || "Untitled document"}</h2>
              <span className="shrink-0 text-sm text-gray-500">
                {result.score.toFixed(3)}
              </span>
            </div>
            <p className="mt-1 text-xs text-gray-500">Document {result.doc_id}</p>
            <p className="mt-3 line-clamp-4 text-gray-700">{result.text}</p>
            {generatedAnswer?.cited_doc_ids.includes(result.doc_id) && (
              <p className="mt-3 text-xs font-semibold uppercase tracking-wide text-sky-800">
                Cited as evidence
              </p>
            )}
          </article>
        ))}
        {!busyAction && !error && query && results.length === 0 && (
          <p className="text-gray-600">No matching documents found.</p>
        )}
      </section>
    </main>
  );
}
