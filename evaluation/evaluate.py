"""Prototype evaluation for PrivacyLens.

Usage:
    python -m evaluation.evaluate                 # retrieval metrics only (no API key needed)
    python -m evaluation.evaluate --with-llm      # + answer metrics (needs OPENAI_API_KEY)
    python -m evaluation.evaluate --with-llm --json results.json

Metrics (all on a small, hand-annotated set — prototype numbers, NOT general
performance guarantees):
* Retrieval hit rate@k: share of answerable questions where at least one
  retrieved chunk comes from an annotated supporting section.
* Citation validity rate: share of model-proposed citations that matched a
  retrieved chunk (measured before invalid IDs are removed).
* Answer correctness: evidence status is one of the expected statuses AND (for
  answerable questions) the answer mentions an expected key fact AND cites at
  least one chunk from a supporting section.
* Correct abstention rate: share of unanswerable / out-of-scope questions
  answered with evidence_status "not_found".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import get_settings  # noqa: E402
from core.analyzer import process_policy  # noqa: E402
from core.embeddings import load_embedder  # noqa: E402
from core.qa import answer_question, retrieve_for_question  # noqa: E402

QUESTIONS_PATH = Path(__file__).with_name("questions.json")


def _supports(chunk_heading: str | None, sections: list[str]) -> bool:
    return bool(chunk_heading) and any(s.lower() in chunk_heading.lower() for s in sections)


def _rate(num: int, den: int) -> str:
    return f"{num}/{den} = {num / den:.0%}" if den else "n/a"


def run(with_llm: bool, k: int | None = None) -> dict:
    settings = get_settings()
    k = k or settings.qa_top_k
    spec = json.loads(QUESTIONS_PATH.read_text())
    policy_text = (ROOT / spec["policy"]).read_text(encoding="utf-8")
    policy = process_policy(policy_text, load_embedder(settings.embedding_model), settings=settings)

    llm = None
    if with_llm:
        from core.llm import LLMClient

        llm = LLMClient.from_settings(settings)

    rows = []
    for q in spec["questions"]:
        chunks = retrieve_for_question(policy, q["question"], None, k)
        row = {
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "retrieved": [f"{c.id} ({c.heading})" for c in chunks],
        }
        if q["supporting_sections"]:
            row["retrieval_hit"] = any(_supports(c.heading, q["supporting_sections"]) for c in chunks)
        if llm is not None:
            ans = answer_question(policy, q["question"], llm, settings=settings)
            proposed = len(ans.source_ids) + len(ans.rejected_ids)
            status_ok = ans.evidence_status.value in q["expected_status"]
            mention_ok = (not q["answer_must_mention"]) or any(
                m.lower() in (ans.answer + " " + (ans.limitations or "")).lower() for m in q["answer_must_mention"]
            )
            cited_support = (not q["supporting_sections"]) or any(
                _supports(policy.index.get(s).heading, q["supporting_sections"]) for s in ans.source_ids
            )
            row.update(
                status=ans.evidence_status.value,
                answer=ans.answer,
                limitations=ans.limitations,
                citations_proposed=proposed,
                citations_valid=len(ans.source_ids),
                status_ok=status_ok,
                mention_ok=mention_ok,
                cited_support=cited_support,
                correct=status_ok and mention_ok and (cited_support or ans.evidence_status.value == "not_found"),
            )
        rows.append(row)

    answerable = [r for r in rows if "retrieval_hit" in r]
    metrics = {
        "retrieval_hit_rate": _rate(sum(r["retrieval_hit"] for r in answerable), len(answerable)),
        "k": k,
        "num_questions": len(rows),
    }
    if llm is not None:
        unanswerable = [r for r in rows if r["type"] in ("unanswerable", "out_of_scope")]
        metrics.update(
            citation_validity_rate=_rate(sum(r["citations_valid"] for r in rows), sum(r["citations_proposed"] for r in rows)),
            answer_correctness=_rate(sum(r["correct"] for r in rows), len(rows)),
            correct_abstention_rate=_rate(sum(r["status"] == "not_found" for r in unanswerable), len(unanswerable)),
            false_abstentions_on_answerable=sum(r["status"] == "not_found" for r in answerable),
            model=settings.openai_model,
        )
    return {"metrics": metrics, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-llm", action="store_true", help="also evaluate generated answers (needs OPENAI_API_KEY)")
    parser.add_argument("--k", type=int, default=None, help="chunks retrieved per question")
    parser.add_argument("--json", type=Path, default=None, help="write full results to this file")
    args = parser.parse_args()

    result = run(args.with_llm, args.k)
    print("PrivacyLens prototype evaluation (small hand-annotated set; not a general performance claim)\n")
    for row in result["rows"]:
        flags = []
        if "retrieval_hit" in row:
            flags.append("retrieval:" + ("hit" if row["retrieval_hit"] else "MISS"))
        if "correct" in row:
            flags.append(f"status:{row['status']}")
            flags.append("answer:" + ("ok" if row["correct"] else "WRONG"))
        print(f"{row['id']} [{row['type']}] {' '.join(flags)}\n    {row['question']}")
    print("\nMetrics:")
    for key, value in result["metrics"].items():
        print(f"  {key}: {value}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
        print(f"\nFull results written to {args.json}")


if __name__ == "__main__":
    main()
