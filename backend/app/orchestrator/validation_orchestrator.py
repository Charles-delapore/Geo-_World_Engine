from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from app.core.geometry_metrics import compute_metric_report
from app.core.semantic_mapper import compute_consistency_score
from app.core.spatial_critic import evaluate_terrain_consistency
from app.core.spatial_relation_graph import SpatialRelationGraph, SRGEntity, SRGEdge, EntityType, TopologicalPredicate, FuzzyQuantifier

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    passed: bool
    metric_report: dict = field(default_factory=dict)
    consistency: dict = field(default_factory=dict)
    srg_consistency: dict = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "metric_report": self.metric_report,
            "consistency": self.consistency,
            "srg_consistency": self.srg_consistency,
            "issues": self.issues,
        }


class ValidationOrchestrator:
    def validate(
        self,
        elevation: np.ndarray,
        plan: dict,
        topology_intent: dict | None = None,
    ) -> ValidationResult:
        issues: list[str] = []
        ti = topology_intent or plan.get("topology_intent") or {}

        metric_report = {}
        try:
            metric_report = compute_metric_report(elevation, ti)
        except Exception as exc:
            issues.append(f"Metric report failed: {exc}")
            logger.warning("Metric report failed: %s", exc)

        consistency = {}
        try:
            plan_params = {
                "land_ratio": float((plan.get("profile") or {}).get("land_ratio") or 0.44),
                "target_land_component_count": int(ti.get("target_land_component_count") or 1),
            }
            consistency = compute_consistency_score(
                plan.get("prompt", ""),
                plan_params,
                elevation,
            )
        except Exception as exc:
            issues.append(f"Consistency score failed: {exc}")
            logger.warning("Consistency score failed: %s", exc)

        srg_consistency = {}
        try:
            srg_data = plan.get("srg")
            if srg_data:
                srg = _reconstruct_srg(srg_data)
                srg_consistency = evaluate_terrain_consistency(elevation, srg, ti)
        except Exception as exc:
            issues.append(f"SRG consistency failed: {exc}")
            logger.warning("SRG consistency failed: %s", exc)

        passed = len(issues) == 0
        return ValidationResult(
            passed=passed,
            metric_report=metric_report,
            consistency=consistency or {},
            srg_consistency=srg_consistency or {},
            issues=issues,
        )


def _reconstruct_srg(srg_data: dict) -> SpatialRelationGraph:
    graph = SpatialRelationGraph(
        raw_text=srg_data.get("raw_text", ""),
        cost_steps=srg_data.get("cost_steps", []),
    )
    for entity_data in srg_data.get("entities", []):
        entity_type_str = entity_data.get("entity_type", "continent")
        try:
            entity_type = EntityType(entity_type_str)
        except ValueError:
            entity_type = EntityType.CONTINENT
        graph.add_entity(SRGEntity(
            name=entity_data.get("name", ""),
            entity_type=entity_type,
            position=entity_data.get("position"),
            attributes=entity_data.get("attributes", {}),
            text_span=entity_data.get("text_span", ""),
        ))
    for edge_data in srg_data.get("edges", []):
        predicate_str = edge_data.get("predicate", "disjoint")
        try:
            predicate = TopologicalPredicate(predicate_str)
        except ValueError:
            predicate = TopologicalPredicate.DISJOINT
        quantifier = None
        if edge_data.get("quantifier"):
            try:
                quantifier = FuzzyQuantifier(edge_data["quantifier"])
            except ValueError:
                pass
        graph.add_edge(SRGEdge(
            subject=edge_data.get("subject", ""),
            predicate=predicate,
            object=edge_data.get("object", ""),
            quantifier=quantifier,
            confidence=edge_data.get("confidence", 1.0),
            text_evidence=edge_data.get("text_evidence", ""),
        ))
    return graph
