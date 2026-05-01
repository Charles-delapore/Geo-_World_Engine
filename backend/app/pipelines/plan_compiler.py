from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.core.llm_parser import parse_with_rag
from app.core.spatial_relation_graph import extract_srg, srg_to_topology_intent
from app.core.spatial_critic import critique_with_iteration

logger = logging.getLogger(__name__)


@dataclass
class ConstraintNode:
    node_id: str
    node_type: str
    position: Optional[str] = None
    attributes: dict = field(default_factory=dict)
    source: str = "text"


@dataclass
class ConstraintEdge:
    subject_id: str
    predicate: str
    object_id: str
    quantifier: Optional[str] = None
    confidence: float = 1.0


@dataclass
class ConstraintGraph:
    nodes: list[ConstraintNode] = field(default_factory=list)
    edges: list[ConstraintEdge] = field(default_factory=list)
    raw_text: str = ""

    def add_node(self, node: ConstraintNode) -> None:
        existing = [n for n in self.nodes if n.node_id == node.node_id]
        if not existing:
            self.nodes.append(node)
        else:
            if node.position and not existing[0].position:
                existing[0].position = node.position
            existing[0].attributes.update(node.attributes)

    def add_edge(self, edge: ConstraintEdge) -> None:
        duplicate = any(
            e.subject_id == edge.subject_id
            and e.predicate == edge.predicate
            and e.object_id == edge.object_id
            for e in self.edges
        )
        if not duplicate:
            self.edges.append(edge)

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type,
                    "position": n.position,
                    "attributes": n.attributes,
                    "source": n.source,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "subject_id": e.subject_id,
                    "predicate": e.predicate,
                    "object_id": e.object_id,
                    "quantifier": e.quantifier,
                    "confidence": e.confidence,
                }
                for e in self.edges
            ],
            "raw_text": self.raw_text,
        }

    @classmethod
    def from_srg(cls, srg_data: dict) -> "ConstraintGraph":
        graph = cls(raw_text=srg_data.get("raw_text", ""))
        for entity in srg_data.get("entities", []):
            graph.add_node(ConstraintNode(
                node_id=entity.get("name", ""),
                node_type=entity.get("entity_type", "continent"),
                position=entity.get("position"),
                attributes=entity.get("attributes", {}),
                source="text",
            ))
        for edge in srg_data.get("edges", []):
            graph.add_edge(ConstraintEdge(
                subject_id=edge.get("subject", ""),
                predicate=edge.get("predicate", "disjoint"),
                object_id=edge.get("object", ""),
                quantifier=edge.get("quantifier"),
                confidence=edge.get("confidence", 1.0),
            ))
        return graph


class PlanCompiler:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def compile(
        self,
        prompt: str,
        params: dict | None = None,
        asset_ids: list[str] | None = None,
    ) -> tuple[dict, ConstraintGraph]:
        plan = parse_with_rag(
            user_prompt=prompt,
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
        )

        srg = extract_srg(prompt)
        constraint_graph = ConstraintGraph.from_srg(srg.to_dict())
        plan["constraint_graph"] = constraint_graph.to_dict()

        if asset_ids:
            plan = self._inject_assets(plan, asset_ids, params or {})
            for node in plan.get("_asset_graph_nodes", []):
                constraint_graph.add_node(ConstraintNode(
                    node_id=node["node_id"],
                    node_type=node["node_type"],
                    position=node.get("position"),
                    attributes=node.get("attributes", {}),
                    source=node.get("source", "asset"),
                ))
            for edge in plan.get("_asset_graph_edges", []):
                constraint_graph.add_edge(ConstraintEdge(
                    subject_id=edge["subject_id"],
                    predicate=edge["predicate"],
                    object_id=edge["object_id"],
                    quantifier=edge.get("quantifier"),
                    confidence=edge.get("confidence", 1.0),
                ))
            plan.pop("_asset_graph_nodes", None)
            plan.pop("_asset_graph_edges", None)
            plan["constraint_graph"] = constraint_graph.to_dict()

        return plan, constraint_graph

    def _inject_assets(self, plan: dict, asset_ids: list[str], params: dict) -> dict:
        from app.pipelines.asset_normalizer import AssetNormalizer

        normalizer = AssetNormalizer()
        asset_results = []
        for asset_id in asset_ids:
            try:
                asset_meta = self._resolve_asset_meta(asset_id, params)
                result = normalizer.normalize(asset_id, asset_meta.get("type", "image"))
                asset_results.append(result)
                logger.info("Normalized asset %s type=%s", asset_id, result.get("type"))
            except Exception as exc:
                logger.warning("Failed to normalize asset %s: %s", asset_id, exc)

        if asset_results:
            asset_graph = normalizer.assets_to_constraint_graph(asset_results)
            plan["_asset_graph_nodes"] = [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type,
                    "position": n.position,
                    "attributes": n.attributes,
                    "source": n.source,
                }
                for n in asset_graph.nodes
            ]
            plan["_asset_graph_edges"] = [
                {
                    "subject_id": e.subject_id,
                    "predicate": e.predicate,
                    "object_id": e.object_id,
                    "quantifier": e.quantifier,
                    "confidence": e.confidence,
                }
                for e in asset_graph.edges
            ]
            existing_nodes = plan.get("constraint_graph", {}).get("nodes", [])
            for node in asset_graph.nodes:
                existing_nodes.append({
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "position": node.position,
                    "attributes": node.attributes,
                    "source": node.source,
                })
            plan = normalizer.inject_into_plan(plan, asset_results)

        plan["asset_ids"] = asset_ids
        return plan

    @staticmethod
    def _resolve_asset_meta(asset_id: str, params: dict) -> dict:
        assets_meta = params.get("assets_meta") or {}
        if asset_id in assets_meta:
            return assets_meta[asset_id]
        return {"type": "image"}
