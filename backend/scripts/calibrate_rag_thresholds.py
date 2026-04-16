from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.knowledge_base import TerrainKnowledgeBase


def analyze_similarity_distribution(eval_set_path: str = "tests/fixtures/rag_eval_set.json") -> None:
    path = Path(eval_set_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation set not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        cases = json.load(handle)

    kb = TerrainKnowledgeBase()
    similarities: list[float] = []
    diffs: list[float] = []
    print("Per-case retrieval:")
    for case in cases:
        results = kb.retrieve_with_distance(case["prompt"], top_k=2)
        if not results:
            print(f"- {case['prompt']} -> no_results")
            continue
        similarities.append(results[0]["similarity"])
        top_name = results[0]["name"]
        second = results[1]["similarity"] if len(results) >= 2 else None
        print(
            f"- {case['prompt']} -> top1={top_name} sim={results[0]['similarity']:.3f}"
            + (f", gap={results[0]['similarity'] - second:.3f}" if second is not None else "")
        )
        if len(results) >= 2:
            diffs.append(results[0]["similarity"] - results[1]["similarity"])

    if not similarities:
        raise RuntimeError("No similarity values collected")

    similarity_array = np.array(similarities, dtype=np.float32)
    print(
        "Similarity distribution:",
        f"min={similarity_array.min():.3f}",
        f"p25={np.percentile(similarity_array, 25):.3f}",
        f"median={np.median(similarity_array):.3f}",
        f"p75={np.percentile(similarity_array, 75):.3f}",
        f"max={similarity_array.max():.3f}",
    )
    print(f"Suggested min_similarity: {np.percentile(similarity_array, 25) - 0.05:.3f}")

    if diffs:
        diff_array = np.array(diffs, dtype=np.float32)
        print(
            "Top-1 vs Top-2 gap:",
            f"p25={np.percentile(diff_array, 25):.3f}",
            f"median={np.median(diff_array):.3f}",
            f"p75={np.percentile(diff_array, 75):.3f}",
        )


if __name__ == "__main__":
    analyze_similarity_distribution()
