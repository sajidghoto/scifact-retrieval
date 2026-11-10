"""Create a read-only review of the queries missed by all three retrievers."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
FAILURES_PATH = ROOT_DIR / "results" / "failures_hybrid.md"
QUERIES_PATH = ROOT_DIR / "data" / "raw" / "scifact" / "queries.jsonl"
CORPUS_PATH = ROOT_DIR / "data" / "raw" / "scifact" / "corpus.jsonl"
QRELS_PATH = ROOT_DIR / "data" / "raw" / "scifact" / "qrels" / "test.tsv"
OUTPUT_PATH = ROOT_DIR / "results" / "failure_review.md"

# This is the standard compact English stopword list used by this review.
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he he'd he'll he's
    her here here's hers herself him himself his how how's i i'd i'll i'm i've
    if in into is isn't it it's its itself just me more most mustn't my myself
    no nor not of off on once only or other ought our ours ourselves out over
    own same shan't she she'd she'll she's should shouldn't so some such than
    that that's the their theirs them themselves then there there's these they
    they'd they'll they're they've this those through to too under until up very
    was wasn't we we'd we'll we're we've were weren't what what's when when's
    where where's which while who who's whom why why's with won't would wouldn't
    you you'd you'll you're you've your yours yourself yourselves
    """.split()
)
WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def term_overlap(claim: str, abstract_title: str, abstract_text: str) -> tuple[list[str], list[str]]:
    """Return unique claim terms found and missing in one abstract."""
    terms = []
    for word in WORD_PATTERN.findall(claim.lower()):
        if len(word) >= 3 and word not in STOPWORDS and word not in terms:
            terms.append(word)
    abstract = f"{abstract_title} {abstract_text}"
    found = [
        term
        for term in terms
        if re.search(rf"\b{re.escape(term)}\b", abstract, flags=re.IGNORECASE)
    ]
    return found, [term for term in terms if term not in found]


def read_failure_data() -> tuple[
    list[str],
    dict[str, str],
    dict[str, dict[str, dict[str, str]]],
    dict[str, dict[str, str]],
]:
    failure_text = FAILURES_PATH.read_text(encoding="utf-8")
    ids_match = re.search(r"All-three miss query IDs:\s*(.+)", failure_text)
    if not ids_match:
        raise ValueError("Could not find all-three query IDs in failures_hybrid.md")
    query_ids = [value.strip() for value in ids_match.group(1).split(",")]

    ranks: dict[str, dict[str, str]] = {}
    sections = re.split(r"(?m)^## Query ", failure_text)[1:]
    for section in sections:
        query_id = section.splitlines()[0].strip()
        rank_match = re.search(
            r"- Best relevant rank:\s+- hybrid: (.+)\s+- vector: (.+)\s+- bm25: (.+)",
            section,
        )
        if not rank_match:
            raise ValueError(f"Could not find ranks for query {query_id}")
        ranks[query_id] = {
            "hybrid": rank_match.group(1).strip(),
            "vector": rank_match.group(2).strip(),
            "BM25": rank_match.group(3).strip(),
        }

    queries = {}
    with QUERIES_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            queries[str(record["_id"])] = record["text"]

    corpus = {}
    with CORPUS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            corpus[str(record["_id"])] = record

    relevant: dict[str, list[str]] = {query_id: [] for query_id in query_ids}
    with QRELS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            query_id, corpus_id, score = line.rstrip("\n").split("\t")
            if query_id in relevant and score == "1":
                relevant[query_id].append(corpus_id)

    for query_id in query_ids:
        if query_id not in queries or query_id not in ranks:
            raise ValueError(f"Missing query or rank data for query {query_id}")
        for corpus_id in relevant[query_id]:
            if corpus_id not in corpus:
                raise ValueError(f"Missing corpus document {corpus_id}")

    return query_ids, queries, {
        query_id: {corpus_id: corpus[corpus_id] for corpus_id in corpus_ids}
        for query_id, corpus_ids in relevant.items()
    }, ranks


def render_review() -> str:
    query_ids, queries, relevant, ranks = read_failure_data()
    stopwords = " ".join(sorted(STOPWORDS))
    lines = [
        "# Failure review",
        "",
        f"Query count: {len(query_ids)}",
        "Overlap lists are exact word matches only (no stemming or spelling normalisation).",
        f"Stopword list used: {stopwords}",
        "",
    ]
    for query_id in query_ids:
        lines.extend([f"## Query {query_id}", "", f"Claim: {queries[query_id]}", ""])
        lines.extend(
            [
                "Best relevant rank:",
                f"- hybrid: {ranks[query_id]['hybrid']}",
                f"- vector: {ranks[query_id]['vector']}",
                f"- BM25: {ranks[query_id]['BM25']}",
                "",
            ]
        )
        for corpus_id, document in relevant[query_id].items():
            found, missing = term_overlap(
                queries[query_id], document["title"], document["text"]
            )
            lines.extend(
                [
                    f"### Relevant document {corpus_id}",
                    "",
                    f"Title: {document['title']}",
                    "",
                    f"Abstract: {document['text']}",
                    "",
                    f"claim terms found: {found}",
                    f"claim terms missing: {missing}",
                    "",
                ]
            )
    return "\n".join(lines)


def main() -> None:
    OUTPUT_PATH.write_text(render_review(), encoding="utf-8")
    query_count = int(
        re.search(
            r"Query count: (\d+)", OUTPUT_PATH.read_text(encoding="utf-8")
        ).group(1)
    )
    print(f"Wrote {OUTPUT_PATH} ({query_count} queries)")


if __name__ == "__main__":
    main()
