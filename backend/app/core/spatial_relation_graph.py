from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TopologicalPredicate(str, Enum):
    DISJOINT = "disjoint"
    TOUCHES = "touches"
    CROSSES = "crosses"
    OVERLAPS = "overlaps"
    WITHIN = "within"
    CONTAINS = "contains"
    SEPARATED_BY = "separated_by"
    ADJACENT_TO = "adjacent_to"
    NORTH_OF = "north_of"
    SOUTH_OF = "south_of"
    EAST_OF = "east_of"
    WEST_OF = "west_of"
    ENCLOSES = "encloses"
    ENCLOSED_BY = "enclosed_by"
    FLANKS = "flanks"


class EntityType(str, Enum):
    CONTINENT = "continent"
    MOUNTAIN = "mountain"
    SEA = "sea"
    OCEAN = "ocean"
    STRAIT = "strait"
    INLAND_SEA = "inland_sea"
    RIVER = "river"
    ISLAND = "island"
    ARCHIPELAGO = "archipelago"
    PENINSULA = "peninsula"
    PLAIN = "plain"
    DESERT = "desert"
    BAY = "bay"
    GULF = "gulf"


class FuzzyQuantifier(str, Enum):
    IMMEDIATELY = "immediately"
    CLOSE = "close"
    NEAR = "near"
    FAR = "far"
    VERY_FAR = "very_far"
    ALONG = "along"
    ACROSS = "across"
    BETWEEN = "between"


ENTITY_PATTERNS: list[tuple[EntityType, list[str]]] = [
    (EntityType.ARCHIPELAGO, ["archipelago", "islands", "群岛", "列岛", "岛链"]),
    (EntityType.PENINSULA, ["peninsula", "半岛"]),
    (EntityType.INLAND_SEA, ["inland sea", "inner sea", "enclosed sea", "内海", "内陆海"]),
    (EntityType.STRAIT, ["strait", "海峡"]),
    (EntityType.GULF, ["gulf", "海湾"]),
    (EntityType.BAY, ["bay", "湾"]),
    (EntityType.OCEAN, ["ocean", "海洋", "大洋"]),
    (EntityType.SEA, ["sea", "海"]),
    (EntityType.MOUNTAIN, ["mountain", "mountains", "range", "ridge", "peaks", "山", "山脉", "高山", "山脊"]),
    (EntityType.RIVER, ["river", "rivers", "stream", "河", "河流", "水系"]),
    (EntityType.DESERT, ["desert", "arid", "沙漠", "干旱"]),
    (EntityType.PLAIN, ["plain", "plains", "flatland", "平原", "平坦"]),
    (EntityType.ISLAND, ["island", "islands", "岛"]),
    (EntityType.CONTINENT, ["continent", "continents", "landmass", "mainland", "大陆", "陆地"]),
]

PREDICATE_PATTERNS: list[tuple[TopologicalPredicate, list[str]]] = [
    (TopologicalPredicate.SEPARATED_BY, ["隔开", "分隔", "separated by", "divided by", "split by", "隔海"]),
    (TopologicalPredicate.ADJACENT_TO, ["相邻", "毗邻", "adjacent", "neighboring", "bordering", "next to", "beside"]),
    (TopologicalPredicate.NORTH_OF, ["北侧", "北部", "north of", "north side", "以北", "北面"]),
    (TopologicalPredicate.SOUTH_OF, ["南侧", "南部", "south of", "south side", "以南", "南面"]),
    (TopologicalPredicate.EAST_OF, ["东侧", "东部", "east of", "east side", "以东", "东面"]),
    (TopologicalPredicate.WEST_OF, ["西侧", "西部", "west of", "west side", "以西", "西面"]),
    (TopologicalPredicate.ENCLOSES, ["包围", "环绕", "encloses", "surrounds", "wraps"]),
    (TopologicalPredicate.ENCLOSED_BY, ["被包围", "被环绕", "enclosed by", "surrounded by"]),
    (TopologicalPredicate.FLANKS, ["两侧", "flanking", "flanks", "两侧是"]),
    (TopologicalPredicate.WITHIN, ["内部", "之中", "within", "inside", "在…内"]),
    (TopologicalPredicate.CONTAINS, ["包含", "含有", "contains", "has"]),
    (TopologicalPredicate.CROSSES, ["穿过", "横跨", "crosses", "crossing", "横贯"]),
    (TopologicalPredicate.OVERLAPS, ["重叠", "overlap", "overlapping"]),
    (TopologicalPredicate.TOUCHES, ["接触", "相连", "touches", "connected", "相接"]),
    (TopologicalPredicate.DISJOINT, ["分离", "不相邻", "disjoint", "separated", "不相连"]),
]

QUANTIFIER_PATTERNS: list[tuple[FuzzyQuantifier, list[str]]] = [
    (FuzzyQuantifier.IMMEDIATELY, ["紧邻", "紧挨", "紧靠", "immediately", "directly", "right next to"]),
    (FuzzyQuantifier.CLOSE, ["近", "附近", "close to", "nearby", "not far"]),
    (FuzzyQuantifier.NEAR, ["附近", "旁边", "near", "beside", "by"]),
    (FuzzyQuantifier.FAR, ["远", "遥远", "far", "distant", "remote"]),
    (FuzzyQuantifier.VERY_FAR, ["很远", "极远", "very far", "extremely far"]),
    (FuzzyQuantifier.ALONG, ["沿着", "沿", "along", "following"]),
    (FuzzyQuantifier.ACROSS, ["横跨", "跨越", "across", "spanning"]),
    (FuzzyQuantifier.BETWEEN, ["之间", "中间", "between", "in between"]),
]

POSITION_KEYWORDS: dict[str, list[str]] = {
    "northwest": ["northwest", "西北", "西北部"],
    "north": ["north", "北部", "北方", "北侧"],
    "northeast": ["northeast", "东北", "东北部"],
    "west": ["west", "西部", "西侧", "西面"],
    "center": ["center", "central", "中央", "中心", "中部"],
    "east": ["east", "东部", "东侧", "东面"],
    "southwest": ["southwest", "西南", "西南部"],
    "south": ["south", "南部", "南方", "南侧"],
    "southeast": ["southeast", "东南", "东南部"],
}


@dataclass
class SRGEntity:
    name: str
    entity_type: EntityType
    position: Optional[str] = None
    attributes: dict = field(default_factory=dict)
    text_span: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "entity_type": self.entity_type.value,
            "position": self.position,
            "attributes": self.attributes,
            "text_span": self.text_span,
        }


@dataclass
class SRGEdge:
    subject: str
    predicate: TopologicalPredicate
    object: str
    quantifier: Optional[FuzzyQuantifier] = None
    confidence: float = 1.0
    text_evidence: str = ""

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate.value,
            "object": self.object,
            "quantifier": self.quantifier.value if self.quantifier else None,
            "confidence": self.confidence,
            "text_evidence": self.text_evidence,
        }


@dataclass
class SpatialRelationGraph:
    entities: list[SRGEntity] = field(default_factory=list)
    edges: list[SRGEdge] = field(default_factory=list)
    raw_text: str = ""
    cost_steps: list[str] = field(default_factory=list)

    def add_entity(self, entity: SRGEntity) -> None:
        existing = [e for e in self.entities if e.name == entity.name]
        if not existing:
            self.entities.append(entity)
        else:
            if entity.position and not existing[0].position:
                existing[0].position = entity.position
            existing[0].attributes.update(entity.attributes)

    def add_edge(self, edge: SRGEdge) -> None:
        duplicate = any(
            e.subject == edge.subject
            and e.predicate == edge.predicate
            and e.object == edge.object
            for e in self.edges
        )
        if not duplicate:
            self.edges.append(edge)

    def get_entity(self, name: str) -> Optional[SRGEntity]:
        for e in self.entities:
            if e.name == name:
                return e
        return None

    def get_entities_by_type(self, entity_type: EntityType) -> list[SRGEntity]:
        return [e for e in self.entities if e.entity_type == entity_type]

    def get_edges_for_entity(self, name: str) -> list[SRGEdge]:
        return [e for e in self.edges if e.subject == name or e.object == name]

    def to_dict(self) -> dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "edges": [e.to_dict() for e in self.edges],
            "raw_text": self.raw_text,
            "cost_steps": self.cost_steps,
        }

    def to_topology_intent(self) -> dict:
        return srg_to_topology_intent(self)


def extract_srg(text: str) -> SpatialRelationGraph:
    graph = SpatialRelationGraph(raw_text=text)
    normalized = _normalize(text)

    graph.cost_steps.append(f"Step1: 识别地理实体")
    entities = _extract_entities(normalized, text)
    for entity in entities:
        graph.add_entity(entity)
    graph.cost_steps.append(f"  发现 {len(entities)} 个实体: {[e.name for e in entities]}")

    graph.cost_steps.append(f"Step2: 解析全局方位基准(统一以正北为0°)")
    _resolve_global_orientation(graph, normalized)

    graph.cost_steps.append(f"Step3: 推导实体间的相对拓扑与距离关系")
    edges = _extract_edges(normalized, text, graph)
    for edge in edges:
        graph.add_edge(edge)
    graph.cost_steps.append(f"  发现 {len(edges)} 条拓扑关系")

    graph.cost_steps.append(f"Step4: 验证全局布局一致性")
    violations = _validate_consistency(graph)
    if violations:
        graph.cost_steps.append(f"  一致性警告: {violations}")
        _repair_inconsistencies(graph, violations)
    else:
        graph.cost_steps.append(f"  全局布局一致")

    return graph


def srg_to_topology_intent(graph: SpatialRelationGraph) -> dict:
    continents = graph.get_entities_by_type(EntityType.CONTINENT)
    mountains = graph.get_entities_by_type(EntityType.MOUNTAIN)
    seas = graph.get_entities_by_type(EntityType.SEA) + graph.get_entities_by_type(EntityType.OCEAN)
    inland_seas = graph.get_entities_by_type(EntityType.INLAND_SEA)
    straits = graph.get_entities_by_type(EntityType.STRAIT)
    archipelagos = graph.get_entities_by_type(EntityType.ARCHIPELAGO)
    peninsulas = graph.get_entities_by_type(EntityType.PENINSULA)
    islands = graph.get_entities_by_type(EntityType.ISLAND)

    intent: dict = {
        "srg_entities": [e.to_dict() for e in graph.entities],
        "srg_edges": [e.to_dict() for e in graph.edges],
        "continents": [],
        "mountains": [],
        "sea_zones": [],
        "inland_seas": [],
        "island_chains": [],
        "peninsulas": [],
        "topology_predicates": [],
        "must_disconnect_pairs": [],
        "forbid_cross_cut": False,
    }

    for c in continents:
        pos = c.position or "center"
        if any(existing.get("position") == pos for existing in intent["continents"]):
            continue
        intent["continents"].append({
            "position": pos,
            "size": c.attributes.get("size", 0.38),
        })

    for m in mountains:
        intent["mountains"].append({
            "location": m.position or "center",
            "height": m.attributes.get("height", 0.7),
            "orientation": m.attributes.get("orientation"),
        })

    for s in seas + straits:
        pos = s.position or "center"
        if pos not in intent["sea_zones"]:
            intent["sea_zones"].append(pos)

    for ils in inland_seas:
        intent["inland_seas"].append({
            "position": ils.position or "center",
            "connection": ils.attributes.get("connection", "enclosed"),
        })

    for arch in archipelagos:
        intent["island_chains"].append({
            "position": arch.position or "center",
            "density": arch.attributes.get("density", 0.66),
        })

    for pen in peninsulas:
        intent["peninsulas"].append({
            "location": pen.position or "west",
            "size": pen.attributes.get("size", 0.18),
        })

    for edge in graph.edges:
        intent["topology_predicates"].append(edge.to_dict())
        if edge.predicate == TopologicalPredicate.SEPARATED_BY:
            pair = sorted([edge.subject, edge.object])
            if pair not in intent["must_disconnect_pairs"]:
                intent["must_disconnect_pairs"].append(pair)

    has_single = len(continents) == 1 and not archipelagos and not islands
    has_two = len(continents) >= 2
    has_arch = bool(archipelagos) or len(islands) >= 3

    if has_single:
        intent["kind"] = "single_island"
        intent["forbid_cross_cut"] = True
        intent["exact_landmass_count"] = 1
    elif has_arch:
        intent["kind"] = "archipelago_chain"
        intent["min_land_component_count"] = 3
    elif has_two:
        separated = any(
            e.predicate == TopologicalPredicate.SEPARATED_BY
            for e in graph.edges
        )
        if separated:
            intent["kind"] = "two_continents_with_rift_sea"
            intent["forbid_cross_cut"] = True
            intent["exact_landmass_count"] = 2
        elif inland_seas:
            intent["kind"] = "central_enclosed_inland_sea"
            intent["forbid_cross_cut"] = True
        else:
            intent["kind"] = "two_continents_with_rift_sea"
            intent["forbid_cross_cut"] = True
            intent["exact_landmass_count"] = 2
    elif peninsulas:
        intent["kind"] = "peninsula_coast"
    elif inland_seas:
        intent["kind"] = "central_enclosed_inland_sea"
        intent["forbid_cross_cut"] = True
    else:
        intent["kind"] = "single_island"
        intent["forbid_cross_cut"] = True
        intent["exact_landmass_count"] = 1

    return intent


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("-", " ")).strip()


def _extract_entities(normalized: str, original: str) -> list[SRGEntity]:
    entities: list[SRGEntity] = []
    seen_names: set[str] = set()

    _extract_compound_continents(normalized, entities, seen_names)

    for entity_type, patterns in ENTITY_PATTERNS:
        for pattern in patterns:
            for match in re.finditer(re.escape(pattern), normalized):
                span_start = max(0, match.start() - 30)
                span_end = min(len(normalized), match.end() + 30)
                context = normalized[span_start:span_end]
                position = _extract_position_from_context(context)
                name = f"{entity_type.value}_{position}" if position else f"{entity_type.value}_{len(seen_names) + 1}"
                if name in seen_names:
                    continue
                seen_names.add(name)
                attributes = _extract_entity_attributes(entity_type, context)
                entities.append(SRGEntity(
                    name=name,
                    entity_type=entity_type,
                    position=position,
                    attributes=attributes,
                    text_span=context,
                ))

    _merge_continent_entities(entities, normalized, seen_names)
    return entities


def _extract_compound_continents(normalized: str, entities: list[SRGEntity], seen_names: set[str]) -> None:
    compound_patterns: list[tuple[list[str], list[str]]] = [
        (["west", "east"], [r"东西.{0,6}(?:大陆|陆地|continent)", r"east.?west.{0,6}(?:continent|landmass)", r"一东一西"]),
        (["north", "south"], [r"南北.{0,6}(?:大陆|陆地|continent)", r"north.?south.{0,6}(?:continent|landmass)", r"一南一北"]),
        (["northwest", "southeast"], [r"西北.{0,4}东南.{0,4}(?:大陆|陆地)"]),
        (["northeast", "southwest"], [r"东北.{0,4}西南.{0,4}(?:大陆|陆地)"]),
    ]

    for positions, patterns in compound_patterns:
        for pattern in patterns:
            if re.search(pattern, normalized):
                for pos in positions:
                    name = f"continent_{pos}"
                    if name not in seen_names:
                        seen_names.add(name)
                        context_window = normalized[max(0, normalized.find(pos[:2] if len(pos) >= 2 else pos) - 20):]
                        attributes = _extract_entity_attributes(EntityType.CONTINENT, context_window)
                        entities.append(SRGEntity(
                            name=name,
                            entity_type=EntityType.CONTINENT,
                            position=pos,
                            attributes=attributes,
                            text_span=normalized,
                        ))
                break


def _extract_position_from_context(context: str) -> Optional[str]:
    for canonical, keywords in POSITION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in context:
                return canonical
    return None


def _extract_entity_attributes(entity_type: EntityType, context: str) -> dict:
    attrs: dict = {}
    if entity_type == EntityType.CONTINENT:
        if re.search(r"巨|大|辽阔|广阔|massive|huge|giant|large|vast", context):
            attrs["size"] = 0.55
        elif re.search(r"小|狭|窄|tiny|small|narrow", context):
            attrs["size"] = 0.28
        else:
            attrs["size"] = 0.38
    elif entity_type == EntityType.MOUNTAIN:
        if re.search(r"极高|高耸|towering|very\s+high|lofty", context):
            attrs["height"] = 0.92
        elif re.search(r"高|山|high|tall|mountain", context):
            attrs["height"] = 0.78
        elif re.search(r"低|矮|丘|gentle|low|hill", context):
            attrs["height"] = 0.45
        else:
            attrs["height"] = 0.65
        if re.search(r"东西|east.?west|横向", context):
            attrs["orientation"] = "east-west"
        elif re.search(r"南北|north.?south|纵向", context):
            attrs["orientation"] = "north-south"
    elif entity_type == EntityType.INLAND_SEA:
        if re.search(r"海峡|strait", context):
            attrs["connection"] = "strait"
        elif re.search(r"连接外海|通向|open|outlet", context):
            attrs["connection"] = "east ocean"
        else:
            attrs["connection"] = "enclosed"
    elif entity_type == EntityType.ARCHIPELAGO:
        if re.search(r"密集|dense|packed", context):
            attrs["density"] = 0.88
        elif re.search(r"稀疏|sparse|scattered", context):
            attrs["density"] = 0.44
        else:
            attrs["density"] = 0.66
    elif entity_type == EntityType.PENINSULA:
        if re.search(r"大|large|big|巨大", context):
            attrs["size"] = 0.28
        elif re.search(r"小|small|tiny|narrow", context):
            attrs["size"] = 0.12
        else:
            attrs["size"] = 0.18
    return attrs


def _merge_continent_entities(entities: list[SRGEntity], normalized: str, seen_names: set[str]) -> None:
    continent_entities = [e for e in entities if e.entity_type == EntityType.CONTINENT]
    if len(continent_entities) <= 1:
        return

    positions = {e.position for e in continent_entities if e.position}
    if {"west", "east"} <= positions or re.search(r"东西|east.?west|一东一西", normalized):
        for e in entities:
            if e.entity_type == EntityType.CONTINENT:
                if e.position is None:
                    if "west" not in positions:
                        e.position = "west"
                        e.name = f"continent_west"
                    elif "east" not in positions:
                        e.position = "east"
                        e.name = f"continent_east"
    elif {"north", "south"} <= positions or re.search(r"南北|north.?south|一南一北", normalized):
        for e in entities:
            if e.entity_type == EntityType.CONTINENT:
                if e.position is None:
                    if "north" not in positions:
                        e.position = "north"
                        e.name = f"continent_north"
                    elif "south" not in positions:
                        e.position = "south"
                        e.name = f"continent_south"


def _resolve_global_orientation(graph: SpatialRelationGraph, normalized: str) -> None:
    for entity in graph.entities:
        if entity.position:
            continue
        for edge in graph.edges:
            if edge.subject == entity.name:
                obj = graph.get_entity(edge.object)
                if obj and obj.position:
                    entity.position = _infer_position_from_predicate(
                        edge.predicate, obj.position, inverse=False
                    )
            elif edge.object == entity.name:
                subj = graph.get_entity(edge.subject)
                if subj and subj.position:
                    entity.position = _infer_position_from_predicate(
                        edge.predicate, subj.position, inverse=True
                    )


def _infer_position_from_predicate(
    predicate: TopologicalPredicate, reference_pos: str, inverse: bool
) -> Optional[str]:
    mapping = {
        TopologicalPredicate.NORTH_OF: {"direct": "north", "ref": "south"},
        TopologicalPredicate.SOUTH_OF: {"direct": "south", "ref": "north"},
        TopologicalPredicate.EAST_OF: {"direct": "east", "ref": "west"},
        TopologicalPredicate.WEST_OF: {"direct": "west", "ref": "east"},
    }
    if predicate not in mapping:
        return None
    if inverse:
        return mapping[predicate]["ref"]
    return mapping[predicate]["direct"]


def _extract_edges(normalized: str, original: str, graph: SpatialRelationGraph) -> list[SRGEdge]:
    edges: list[SRGEdge] = []

    for predicate, patterns in PREDICATE_PATTERNS:
        for pattern in patterns:
            for match in re.finditer(re.escape(pattern), normalized):
                span_start = max(0, match.start() - 60)
                span_end = min(len(normalized), match.end() + 60)
                context = normalized[span_start:span_end]

                subject_name, object_name = _identify_subject_object(context, graph, predicate)
                if subject_name and object_name:
                    quantifier = _extract_quantifier(context)
                    edges.append(SRGEdge(
                        subject=subject_name,
                        predicate=predicate,
                        object=object_name,
                        quantifier=quantifier,
                        confidence=0.85,
                        text_evidence=context,
                    ))

    _infer_implicit_edges(graph, edges, normalized)
    return edges


def _identify_subject_object(
    context: str, graph: SpatialRelationGraph, predicate: TopologicalPredicate
) -> tuple[Optional[str], Optional[str]]:
    entities_in_context = []
    for entity in graph.entities:
        if entity.name in context or (entity.position and entity.position in context):
            entities_in_context.append(entity)
        elif entity.text_span and entity.text_span[:20] in context:
            entities_in_context.append(entity)

    for entity_type, _ in ENTITY_PATTERNS:
        type_entities = [e for e in entities_in_context if e.entity_type == entity_type]
        if len(type_entities) >= 2:
            return type_entities[0].name, type_entities[1].name

    if len(entities_in_context) >= 2:
        return entities_in_context[0].name, entities_in_context[1].name

    if predicate == TopologicalPredicate.SEPARATED_BY:
        continents = graph.get_entities_by_type(EntityType.CONTINENT)
        seas = graph.get_entities_by_type(EntityType.SEA) + graph.get_entities_by_type(EntityType.OCEAN)
        if len(continents) >= 2 and seas:
            return continents[0].name, continents[1].name
        elif len(continents) >= 2:
            return continents[0].name, continents[1].name

    return None, None


def _extract_quantifier(context: str) -> Optional[FuzzyQuantifier]:
    for quantifier, patterns in QUANTIFIER_PATTERNS:
        for pattern in patterns:
            if pattern in context:
                return quantifier
    return None


def _infer_implicit_edges(
    graph: SpatialRelationGraph, edges: list[SRGEdge], normalized: str
) -> None:
    continents = graph.get_entities_by_type(EntityType.CONTINENT)
    mountains = graph.get_entities_by_type(EntityType.MOUNTAIN)
    seas = graph.get_entities_by_type(EntityType.SEA) + graph.get_entities_by_type(EntityType.OCEAN)
    inland_seas = graph.get_entities_by_type(EntityType.INLAND_SEA)

    if len(continents) >= 2:
        has_separation = any(
            e.predicate == TopologicalPredicate.SEPARATED_BY
            for e in edges
            if e.subject in {c.name for c in continents} and e.object in {c.name for c in continents}
        )
        if not has_separation:
            if re.search(r"隔开|分隔|separated|between|split|divid|隔海|相望", normalized):
                edges.append(SRGEdge(
                    subject=continents[0].name,
                    predicate=TopologicalPredicate.SEPARATED_BY,
                    object=continents[1].name,
                    confidence=0.85,
                    text_evidence="inferred from separation context",
                ))
            elif re.search(r"两侧|两边|对面|opposite|either side", normalized):
                edges.append(SRGEdge(
                    subject=continents[0].name,
                    predicate=TopologicalPredicate.SEPARATED_BY,
                    object=continents[1].name,
                    confidence=0.7,
                    text_evidence="inferred from opposite sides context",
                ))

        if len(continents) == 2:
            c1, c2 = continents[0], continents[1]
            if c1.position and c2.position:
                if c1.position == "west" and c2.position == "east":
                    has_directional = any(
                        e.predicate == TopologicalPredicate.EAST_OF
                        for e in edges
                        if e.subject == c1.name and e.object == c2.name
                    )
                    if not has_directional:
                        edges.append(SRGEdge(
                            subject=c1.name,
                            predicate=TopologicalPredicate.EAST_OF,
                            object=c2.name,
                            confidence=0.6,
                            text_evidence="inferred from west/east positions",
                        ))
                elif c1.position == "north" and c2.position == "south":
                    has_directional = any(
                        e.predicate == TopologicalPredicate.SOUTH_OF
                        for e in edges
                        if e.subject == c1.name and e.object == c2.name
                    )
                    if not has_directional:
                        edges.append(SRGEdge(
                            subject=c1.name,
                            predicate=TopologicalPredicate.SOUTH_OF,
                            object=c2.name,
                            confidence=0.6,
                            text_evidence="inferred from north/south positions",
                        ))

    for mtn in mountains:
        has_location = any(
            e.subject == mtn.name or e.object == mtn.name
            for e in edges
        )
        if not has_location and mtn.position:
            for cont in continents:
                if cont.position == mtn.position:
                    edges.append(SRGEdge(
                        subject=mtn.name,
                        predicate=TopologicalPredicate.WITHIN,
                        object=cont.name,
                        confidence=0.7,
                        text_evidence="inferred from position overlap",
                    ))
                    break

    for ils in inland_seas:
        has_enclosure = any(
            e.predicate in (TopologicalPredicate.ENCLOSED_BY, TopologicalPredicate.ENCLOSES)
            and (e.subject == ils.name or e.object == ils.name)
            for e in edges
        )
        if not has_enclosure and continents:
            edges.append(SRGEdge(
                subject=ils.name,
                predicate=TopologicalPredicate.ENCLOSED_BY,
                object=continents[0].name,
                confidence=0.65,
                text_evidence="inferred from inland sea type",
            ))


def _validate_consistency(graph: SpatialRelationGraph) -> list[str]:
    violations: list[str] = []

    for edge in graph.edges:
        subj = graph.get_entity(edge.subject)
        obj = graph.get_entity(edge.object)
        if not subj or not obj:
            continue

        if edge.predicate == TopologicalPredicate.SEPARATED_BY:
            if subj.position and obj.position:
                if subj.position == obj.position:
                    violations.append(
                        f"{subj.name} and {obj.name} are separated but share position {subj.position}"
                    )

        if edge.predicate in (TopologicalPredicate.NORTH_OF, TopologicalPredicate.SOUTH_OF):
            if subj.position and obj.position:
                if subj.position == obj.position:
                    violations.append(
                        f"{subj.name} is {edge.predicate.value} of {obj.name} but same position"
                    )

        if edge.predicate in (TopologicalPredicate.EAST_OF, TopologicalPredicate.WEST_OF):
            if subj.position and obj.position:
                if subj.position == obj.position:
                    violations.append(
                        f"{subj.name} is {edge.predicate.value} of {obj.name} but same position"
                    )

    for i, e1 in enumerate(graph.edges):
        for e2 in graph.edges[i + 1:]:
            if (e1.subject == e2.subject and e1.object == e2.object):
                if e1.predicate == TopologicalPredicate.NORTH_OF and e2.predicate == TopologicalPredicate.SOUTH_OF:
                    violations.append(
                        f"Contradiction: {e1.subject} is both north and south of {e1.object}"
                    )
                if e1.predicate == TopologicalPredicate.EAST_OF and e2.predicate == TopologicalPredicate.WEST_OF:
                    violations.append(
                        f"Contradiction: {e1.subject} is both east and west of {e1.object}"
                    )

    return violations


def _repair_inconsistencies(graph: SpatialRelationGraph, violations: list[str]) -> None:
    for violation in violations:
        if "same position" in violation and "separated" in violation:
            for edge in graph.edges:
                if edge.predicate == TopologicalPredicate.SEPARATED_BY:
                    subj = graph.get_entity(edge.subject)
                    obj = graph.get_entity(edge.object)
                    if subj and obj and subj.position == obj.position:
                        if subj.position in ("west", "east"):
                            obj.position = "east" if subj.position == "west" else "west"
                        elif subj.position in ("north", "south"):
                            obj.position = "south" if subj.position == "north" else "north"
                        else:
                            subj.position = "west"
                            obj.position = "east"

        if "Contradiction" in violation:
            edges_to_remove = []
            for i, e1 in enumerate(graph.edges):
                for e2 in graph.edges[i + 1:]:
                    if e1.subject == e2.subject and e1.object == e2.object:
                        if (e1.predicate == TopologicalPredicate.NORTH_OF and e2.predicate == TopologicalPredicate.SOUTH_OF) or \
                           (e1.predicate == TopologicalPredicate.EAST_OF and e2.predicate == TopologicalPredicate.WEST_OF):
                            if e1.confidence <= e2.confidence:
                                edges_to_remove.append(e1)
                            else:
                                edges_to_remove.append(e2)
            for edge in edges_to_remove:
                if edge in graph.edges:
                    graph.edges.remove(edge)
