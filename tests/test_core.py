"""Unit tests for metrics, graph ranking, and extraction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from extraction import build_extracted_docs, clean_turns, extract_turns
from graph import fit_vectorizer, pagerank, select_top_k, similarity_matrix
from metrics import evaluate, extractive_density, rouge_l, rouge_n
from vocab import Vocab, make_example


class MetricsTest(unittest.TestCase):
    def test_rouge_identical(self) -> None:
        self.assertEqual(
            rouge_n(hypothesis="the cat sat", reference="the cat sat", n=1),
            1.0,
        )
        self.assertEqual(
            rouge_l(hypothesis="the cat sat", reference="the cat sat"),
            1.0,
        )

    def test_rouge_disjoint(self) -> None:
        self.assertEqual(rouge_n(hypothesis="a b c", reference="x y z", n=2), 0.0)
        self.assertEqual(rouge_l(hypothesis="a b c", reference="x y z"), 0.0)

    def test_extractive_density(self) -> None:
        self.assertEqual(
            extractive_density(
                hypothesis="the cat sat",
                source="the cat sat on the mat",
            ),
            1.0,
        )
        self.assertEqual(
            extractive_density(hypothesis="zebra", source="the cat sat"),
            0.0,
        )

    def test_evaluate_keys(self) -> None:
        metrics = evaluate(
            hypothesis="the cat sat",
            reference="the cat sat",
            source="the cat sat on the mat",
        )
        self.assertEqual(metrics["rouge1"], 1.0)
        self.assertGreater(metrics["compression_ratio"], 0.0)


class GraphTest(unittest.TestCase):
    def test_complete_graph_is_uniform(self) -> None:
        similarity = np.ones((4, 4))
        np.fill_diagonal(similarity, 0.0)
        scores = pagerank(similarity=similarity, max_iter=200)
        np.testing.assert_allclose(scores, np.full(4, 0.25), atol=1e-8)

    def test_select_top_k_keeps_document_order(self) -> None:
        scores = np.array([0.1, 0.9, 0.5, 0.0])
        self.assertEqual(select_top_k(scores=scores, k=2), [1, 2])


class ExtractionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            {
                "id": "one",
                "turns": [
                    "Amanda: hello",
                    "Jerry: I baked cookies today",
                    "Amanda: bring them tomorrow",
                    "Jerry: ok",
                ],
                "summary": "Jerry baked cookies.",
            }
        ]
        self.vectorizer = fit_vectorizer(
            train_turns=[turn for doc in self.documents for turn in doc["turns"]]
        )

    def test_clean_turns_drops_greetings(self) -> None:
        cleaned = clean_turns(turns=self.documents[0]["turns"])
        self.assertEqual(len(cleaned), 3)
        self.assertNotIn("Amanda: hello", cleaned)

    def test_extract_limits_turns(self) -> None:
        scores = pagerank(
            similarity=similarity_matrix(
                vectorizer=self.vectorizer,
                turns=self.documents[0]["turns"],
            ),
            threshold=0.5,
        )
        text, indices = extract_turns(
            doc=self.documents[0],
            scores=scores,
            m=2,
        )
        self.assertEqual(len(indices), 2)
        self.assertEqual(len(text.splitlines()), 2)

    def test_build_extracted_docs_keeps_summary(self) -> None:
        extracted = build_extracted_docs(
            documents=self.documents,
            vectorizer=self.vectorizer,
            m=2,
            filter_noise=True,
        )
        self.assertEqual(extracted[0]["id"], "one")
        self.assertEqual(extracted[0]["summary"], "Jerry baked cookies.")
        self.assertLessEqual(len(extracted[0]["source"].splitlines()), 2)


class VocabTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        from collections import Counter

        vocab = Vocab(counter=Counter({"cat": 3, "sat": 2}), max_size=10)
        restored = Vocab.from_json(itos=vocab.to_json())
        self.assertEqual(vocab.itos, restored.itos)
        self.assertEqual(restored.w2i(word="cat"), vocab.w2i(word="cat"))

    def test_make_example(self) -> None:
        from collections import Counter

        vocab = Vocab(counter=Counter({"cat": 3, "sat": 2}), max_size=10)
        example = make_example(
            doc={"source": "the cat sat", "summary": "cat sat"},
            vocab=vocab,
        )
        self.assertEqual(len(example.enc), 3)
        self.assertEqual(example.dec_in[0], 2)
        self.assertGreaterEqual(len(example.oovs), 1)


if __name__ == "__main__":
    unittest.main()
