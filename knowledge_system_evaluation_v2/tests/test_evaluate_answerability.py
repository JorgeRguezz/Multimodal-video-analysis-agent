from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from knowledge_system_evaluation_v2.phase_1_2_3_4_scripts import (
    evaluate_answerability as evaluator,
)


CLAIMS = [
    {
        "claim_id": "C1",
        "text": "The combined evidence establishes the claim.",
        "search_query": "combined evidence",
    }
]


def evidence(evidence_id: str, text: str = "Evidence text.") -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "chunk_id": f"chunk-{evidence_id}",
        "video_name": "video",
        "segment_ids": [],
        "source": "bm25",
        "final_score": 1.0,
        "text": text,
    }


def decision(
    status: str,
    evidence_ids: list[str],
    contributing_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "decisions": [
            {
                "claim_id": "C1",
                "status": status,
                "evidence_ids": evidence_ids,
                "contributing_evidence_ids": contributing_ids or [],
                "reason": "Synthetic judgment.",
            }
        ],
        "requires_review": False,
        "review_reason": "",
    }


class ValidateBatchPayloadTests(unittest.TestCase):
    def test_supported_requires_a_valid_citation(self) -> None:
        normalized, violations, fatal = evaluator.validate_batch_payload(
            decision("supported", ["E001"]),
            CLAIMS,
            {"E001"},
            "initial",
            1,
        )

        self.assertEqual([], violations)
        self.assertEqual(set(), fatal)
        self.assertEqual("supported", normalized[0]["status"])
        self.assertEqual(["E001"], normalized[0]["evidence_ids"])

    def test_invalid_contributor_is_removed_without_becoming_decisive(self) -> None:
        normalized, violations, fatal = evaluator.validate_batch_payload(
            decision("insufficient", [], ["E001", "E999"]),
            CLAIMS,
            {"E001"},
            "expanded",
            2,
        )

        self.assertTrue(violations)
        self.assertEqual(set(), fatal)
        self.assertEqual(["E001"], normalized[0]["contributing_evidence_ids"])
        self.assertEqual("insufficient", normalized[0]["status"])

    def test_duplicate_claim_is_conservatively_insufficient(self) -> None:
        payload = decision("supported", ["E001"])
        payload["decisions"].append(dict(payload["decisions"][0]))
        normalized, violations, fatal = evaluator.validate_batch_payload(
            payload,
            CLAIMS,
            {"E001"},
            "initial",
            1,
        )

        self.assertTrue(violations)
        self.assertEqual({"C1"}, fatal)
        self.assertEqual("insufficient", normalized[0]["status"])


class ClassificationTests(unittest.TestCase):
    def test_contradicted_claims_require_review_without_overriding_status(self) -> None:
        status, requires_review = evaluator.classify_answerability(
            supported=0,
            total_claims=2,
            has_review_reasons=False,
            contradicted=1,
        )

        self.assertEqual("None", status)
        self.assertTrue(requires_review)

    def test_partial_with_contradiction_remains_partial(self) -> None:
        status, requires_review = evaluator.classify_answerability(
            supported=1,
            total_claims=3,
            has_review_reasons=False,
            contradicted=1,
        )

        self.assertEqual("Partial", status)
        self.assertTrue(requires_review)


class JudgeBatchRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_semantic_retry_recovers_without_review(self) -> None:
        invalid = decision("supported", ["E999"])
        valid = decision("supported", ["E001"])
        mocked_call = AsyncMock(side_effect=[invalid, valid])

        with patch.object(evaluator, "call_local_json", new=mocked_call):
            decisions, review, diagnostics = await evaluator.judge_batches(
                "Question",
                CLAIMS,
                [evidence("E001")],
                1,
                stage="initial",
            )

        self.assertEqual([], review)
        self.assertEqual("supported", decisions[0]["status"])
        self.assertEqual(1, diagnostics["semantic_retry_count"])
        self.assertEqual(0, diagnostics["semantic_failure_count"])
        self.assertTrue(diagnostics["validation_events"][0]["recovered"])

    async def test_repeated_invalid_citation_forces_review(self) -> None:
        invalid = decision("supported", ["E999"])
        mocked_call = AsyncMock(side_effect=[invalid, invalid])

        with patch.object(evaluator, "call_local_json", new=mocked_call):
            decisions, review, diagnostics = await evaluator.judge_batches(
                "Question",
                CLAIMS,
                [evidence("E001")],
                1,
                stage="initial",
            )

        self.assertTrue(review)
        self.assertEqual("insufficient", decisions[0]["status"])
        self.assertEqual([], decisions[0]["evidence_ids"])
        self.assertEqual(1, diagnostics["semantic_retry_count"])
        self.assertEqual(1, diagnostics["semantic_failure_count"])


class ConsolidationTests(unittest.IsolatedAsyncioTestCase):
    def test_shortlist_preserves_rank_and_limits_size_and_budget(self) -> None:
        claim_result = {
            "contributing_evidence_ids": ["E003", "E001", "E002"],
            "contradicting_evidence_ids": [],
        }
        records = [
            evidence("E001", "a" * 100),
            evidence("E002", "b" * 100),
            evidence("E003", "c" * 100),
        ]
        two_record_budget = len(evaluator.render_evidence(records[:2]))

        selected = evaluator.select_consolidation_evidence(
            claim_result,
            records,
            top_k=3,
            char_budget=two_record_budget,
        )

        self.assertEqual(["E001", "E002"], [item["evidence_id"] for item in selected])
        self.assertLessEqual(len(evaluator.render_evidence(selected)), two_record_budget)

    async def test_cross_batch_contributors_can_jointly_support_claim(self) -> None:
        screened = [
            {
                "claim_id": "C1",
                "status": "insufficient",
                "evidence_ids": [],
                "contributing_evidence_ids": ["E001"],
                "reason": "First material part.",
                "stage": "initial",
                "batch_number": 1,
            },
            {
                "claim_id": "C1",
                "status": "insufficient",
                "evidence_ids": [],
                "contributing_evidence_ids": ["E006"],
                "reason": "Second material part.",
                "stage": "expanded",
                "batch_number": 1,
            },
        ]
        claim_results, _ = evaluator.aggregate(CLAIMS, screened)
        records = [evidence("E001"), evidence("E006")]
        selected = evaluator.select_consolidation_evidence(
            claim_results[0],
            records,
            top_k=10,
            char_budget=30000,
        )
        mocked_call = AsyncMock(
            return_value=decision("supported", ["E001", "E006"])
        )

        with patch.object(evaluator, "call_local_json", new=mocked_call):
            consolidated, review, _ = await evaluator.judge_batches(
                "Question",
                CLAIMS,
                selected,
                len(selected),
                stage="consolidation",
            )

        final_results, _ = evaluator.aggregate(CLAIMS, screened + consolidated)
        self.assertEqual([], review)
        self.assertEqual(2, len(selected))
        self.assertEqual("supported", final_results[0]["status"])
        self.assertEqual(
            ["E001", "E006"],
            final_results[0]["supporting_evidence_ids"],
        )


if __name__ == "__main__":
    unittest.main()
