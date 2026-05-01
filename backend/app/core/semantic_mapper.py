from __future__ import annotations

import math
import re
from typing import Tuple

INTENSITY_WORDS: list[tuple[str, float]] = [
    (r"极(?:其|为|度|端)?(?:大|高|多|强|深|陡|崎岖|辽阔|巨大|宏伟)", 1.0),
    (r"非常|特别|十分|极其|超级|无比|格外|异常", 0.92),
    (r"很|颇|甚|相当|挺|比较|较|蛮|特", 0.78),
    (r"略(?:微|为|稍)?|稍(?:微|稍)?|微(?:微|稍)?|有点|一些|一点|些许|些许", 0.35),
    (r"极(?:其|为|度|端)?(?:小|低|少|弱|浅|平|缓|窄|细|薄)", 0.15),
    (r"very\s+|extremely\s+|super\s+|incredibly\s+", 0.92),
    (r"quite\s+|rather\s+|fairly\s+|pretty\s+", 0.78),
    (r"somewhat\s+|slightly\s+|a\s+bit\s+|mildly\s+", 0.35),
    (r"barely\s+|hardly\s+|minimally\s+", 0.15),
]

BASE_POSITIONS: dict[str, Tuple[float, float]] = {
    "northwest": (0.22, 0.22),
    "north": (0.20, 0.50),
    "northeast": (0.22, 0.78),
    "west": (0.50, 0.22),
    "center": (0.50, 0.50),
    "east": (0.50, 0.78),
    "southwest": (0.78, 0.22),
    "south": (0.80, 0.50),
    "southeast": (0.78, 0.78),
}

OFFSET_MAP: dict[str, Tuple[float, float]] = {
    "偏北": (-0.08, 0.0), "偏南": (0.08, 0.0),
    "偏东": (0.0, 0.08), "偏西": (0.0, -0.08),
    "略北": (-0.05, 0.0), "略南": (0.05, 0.0),
    "略东": (0.0, 0.05), "略西": (0.0, -0.05),
    "靠北": (-0.10, 0.0), "靠南": (0.10, 0.0),
    "靠东": (0.0, 0.10), "靠西": (0.0, -0.10),
    "往北": (-0.06, 0.0), "往南": (0.06, 0.0),
    "往东": (0.0, 0.06), "往西": (0.0, -0.06),
    "north-ish": (-0.05, 0.0), "south-ish": (0.05, 0.0),
    "east-ish": (0.0, 0.05), "west-ish": (0.0, -0.05),
    "slightly north": (-0.05, 0.0), "slightly south": (0.05, 0.0),
    "slightly east": (0.0, 0.05), "slightly west": (0.0, -0.05),
}


def resolve_position_continuous(position: str) -> Tuple[float, float]:
    base = BASE_POSITIONS.get(position, (0.5, 0.5))
    dy, dx = 0.0, 0.0
    lower = position.lower().replace(" ", "")
    for key, (oy, ox) in OFFSET_MAP.items():
        if key in lower:
            dy += oy
            dx += ox
    y = max(0.08, min(0.92, base[0] + dy))
    x = max(0.08, min(0.92, base[1] + dx))
    return (round(y, 3), round(x, 3))


def extract_intensity(text: str, default: float = 0.5) -> float:
    lower = text.lower()
    for pattern, value in INTENSITY_WORDS:
        if re.search(pattern, lower):
            return value
    return default


def map_size_continuous(text: str, position: str = "") -> float:
    intensity = extract_intensity(text, 0.5)
    base = 0.44
    if re.search(r"巨|大|辽阔|广阔|massive|huge|giant|large|vast|big", text.lower()):
        base = 0.44 + 0.18 * intensity
    elif re.search(r"小|狭|窄|tiny|small|narrow|slim|little", text.lower()):
        base = 0.44 - 0.16 * intensity
    else:
        base = 0.44 + 0.08 * (intensity - 0.5)
    if position in ("west", "east", "north", "south"):
        base = max(0.20, min(0.75, base))
    else:
        base = max(0.15, min(0.70, base))
    return round(base, 3)


def map_height_continuous(text: str) -> float:
    intensity = extract_intensity(text, 0.5)
    lower = text.lower()
    if re.search(r"极高|高耸|towering|very\s+high|lofty|soaring", lower):
        return round(0.78 + 0.22 * intensity, 3)
    if re.search(r"高|山|rugged|high|tall|mountain", lower):
        return round(0.55 + 0.35 * intensity, 3)
    if re.search(r"低|矮|丘|gentle|low|small|hill", lower):
        return round(0.30 + 0.25 * intensity, 3)
    return round(0.45 + 0.35 * intensity, 3)


def map_ruggedness_continuous(text: str) -> float:
    intensity = extract_intensity(text, 0.5)
    lower = text.lower()
    if re.search(r"崎岖|险峻|rugged|steep|rough|mountainous", lower):
        return round(0.55 + 0.40 * intensity, 3)
    if re.search(r"平坦|平原|flat|plain|smooth|gentle", lower):
        return round(0.15 + 0.20 * intensity, 3)
    if re.search(r"丘陵|起伏|rolling|hilly", lower):
        return round(0.35 + 0.20 * intensity, 3)
    return round(0.35 + 0.30 * intensity, 3)


def map_coast_complexity_continuous(text: str) -> float:
    intensity = extract_intensity(text, 0.5)
    lower = text.lower()
    if re.search(r"曲折|蜿蜒|破碎|复杂|indented|complex|jagged|rugged\s+coast", lower):
        return round(0.60 + 0.35 * intensity, 3)
    if re.search(r"平直|光滑|简单|straight|smooth|simple", lower):
        return round(0.15 + 0.20 * intensity, 3)
    return round(0.35 + 0.25 * intensity, 3)


def map_moisture_continuous(text: str) -> float:
    intensity = extract_intensity(text, 0.5)
    lower = text.lower()
    if re.search(r"湿润|潮湿|多雨|lush|wet|rainy|humid|moist", lower):
        return round(1.0 + 0.60 * intensity, 3)
    if re.search(r"干旱|沙漠|干燥|arid|dry|desert|parched", lower):
        return round(0.35 + 0.25 * intensity, 3)
    return round(0.80 + 0.40 * intensity, 3)


def map_temperature_bias_continuous(text: str) -> float:
    intensity = extract_intensity(text, 0.5)
    lower = text.lower()
    if re.search(r"炎热|热带|酷热|tropical|hot|scorching", lower):
        return round(4.0 + 8.0 * intensity, 1)
    if re.search(r"温暖|温带|warm|temperate|mild", lower):
        return round(1.0 + 4.0 * intensity, 1)
    if re.search(r"寒冷|冰冻|极寒|frozen|cold|frigid|glacial|arctic", lower):
        return round(-4.0 - 10.0 * intensity, 1)
    if re.search(r"凉爽|凉|cool|chilly", lower):
        return round(-1.0 - 4.0 * intensity, 1)
    return 0.0


def map_land_ratio_continuous(landform_class: str, text: str = "") -> float:
    intensity = extract_intensity(text, 0.5) if text else 0.5
    base_map = {
        "single_island": 0.38,
        "archipelago": 0.32,
        "peninsula": 0.42,
        "supercontinent": 0.62,
        "two_continents": 0.46,
        "generic": 0.44,
    }
    base = base_map.get(landform_class, 0.44)
    lower = text.lower() if text else ""
    if re.search(r"更多陆|陆多|more\s+land|landlocked", lower):
        base += 0.08 * intensity
    elif re.search(r"更多水|海多|more\s+water|oceanic", lower):
        base -= 0.08 * intensity
    return round(max(0.15, min(0.80, base)), 3)


def compute_consistency_score(
    prompt: str,
    plan_params: dict,
    elevation: "numpy.ndarray",
) -> dict:
    import numpy as np

    scores: dict[str, float] = {}
    lower = prompt.lower()

    total_land = float(np.mean(elevation > 0))
    plan_land = plan_params.get("land_ratio", 0.44)
    land_diff = abs(total_land - plan_land)
    scores["land_ratio_match"] = round(max(0.0, 1.0 - land_diff * 3.0), 3)

    if re.search(r"崎岖|rugged|steep|mountain", lower):
        grad_y, grad_x = np.gradient(elevation)
        slope = np.sqrt(grad_y ** 2 + grad_x ** 2)
        mean_slope = float(np.mean(slope[elevation > 0])) if np.any(elevation > 0) else 0.0
        scores["ruggedness_match"] = round(min(1.0, mean_slope * 15.0), 3)

    if re.search(r"平坦|flat|plain", lower):
        grad_y, grad_x = np.gradient(elevation)
        slope = np.sqrt(grad_y ** 2 + grad_x ** 2)
        mean_slope = float(np.mean(slope[elevation > 0])) if np.any(elevation > 0) else 0.0
        scores["flatness_match"] = round(max(0.0, 1.0 - mean_slope * 20.0), 3)

    if re.search(r"岛|island|archipelago", lower):
        from scipy import ndimage
        labeled, n_components = ndimage.label(elevation > 0)
        plan_count = plan_params.get("target_land_component_count", 1)
        if n_components > 0 and plan_count > 0:
            ratio = min(n_components, plan_count) / max(n_components, plan_count)
            scores["island_count_match"] = round(ratio, 3)

    if scores:
        scores["overall"] = round(sum(scores.values()) / len(scores), 3)
    return scores


def extract_spatial_features(text: str) -> dict:
    lower = text.lower()
    features: dict = {}

    features["has_inland_sea"] = bool(re.search(
        r"内海|中间有海|内陆海|被(?:陆|陆地)包围.{0,3}(?:海|水)|(?:中间|中央|中部).{0,3}(?:海|水域|湖)",
        text
    ))
    features["has_open_sea"] = bool(re.search(
        r"开(?:阔|放)(?:海|洋|水)|外海|被(?:开阔海|大洋|大?海)(?:隔开|分隔|分开)",
        text
    ))
    features["is_split_by_sea"] = bool(re.search(
        r"(?:被|隔着|隔开|分隔|分开).{0,6}(?:海|海洋|水)|(?:海|海洋|水).{0,6}(?:隔开|分开|分隔|分割)",
        text
    ))
    features["is_single_continent"] = bool(re.search(
        r"(?:一块|单个|单一|只有|仅有|只有一块|孤立).{0,6}(?:大陆|陆地|大岛)|(?:一|单)个?(?:大陆|陆地|大岛)|四面环海.{0,4}(?:大陆|陆地)",
        text
    ))
    features["is_dual_continent"] = bool(re.search(
        r"(?:两|二)(?:块|片|个|座).{0,3}(?:大陆|陆地)|一东一西|一南一北|东西两|南北两|两块",
        text
    ))
    features["continent_count"] = _extract_continent_count(text)

    features["has_mountains"] = bool(re.search(
        r"山(?:脉|系|峰|脊|地|区|峦|岳)|mountains?|orogen|ridge",
        text
    ))
    features["mountain_location"] = _resolve_mountain_location(text)
    features["mountain_orientation"] = _resolve_mountain_orientation(text)

    features["has_peninsula"] = bool(re.search(
        r"半岛|peninsula|cape|promontory",
        text
    ))
    features["peninsula_location"] = _resolve_peninsula_location(text)

    features["has_archipelago"] = bool(re.search(
        r"群岛|archipelago|island.chain|island.group|岛链|列岛",
        text
    ))
    features["archipelago_location"] = _resolve_archipelago_location(text)

    features["climate_zone"] = _resolve_climate_zone(text)
    features["terrain_style"] = _resolve_terrain_style(text)

    return features


def _extract_continent_count(text: str) -> int | None:
    lower = text.lower()
    if re.search(r"四(?:块|片|个|方|大).{0,3}(?:大陆|陆地)", lower):
        return 4
    if re.search(r"三(?:块|片|个).{0,3}(?:大陆|陆地)", lower):
        return 3
    if re.search(r"(?:两|二)(?:块|片|个).{0,3}(?:大陆|陆地)|一东一西|一南一北", lower):
        return 2
    if re.search(r"(?:一|单)(?:块|片|个).{0,3}(?:大陆|陆地)|四面环海|单个大陆|孤立大陆", lower):
        return 1
    if re.search(r"没有大陆|没有大块陆地|全是岛|只有岛|全是群岛|ocean.world", lower):
        return 0
    return None


def _resolve_mountain_location(text: str) -> str | None:
    if re.search(r"东(?:部|侧|面|边|海岸|缘).{0,4}(?:有|的|是|山脉|山系|山)", text):
        return "east"
    if re.search(r"西(?:部|侧|面|边|海岸|缘).{0,4}(?:有|的|是|山脉|山系|山)", text):
        return "west"
    if re.search(r"北(?:部|侧|面|边|缘|境).{0,4}(?:有|的|是|山脉|山系|山)", text):
        return "north"
    if re.search(r"南(?:部|侧|面|边|缘|境).{0,4}(?:有|的|是|山脉|山系|山)", text):
        return "south"
    if re.search(r"(?:中央|中部|中间|中心|腹地).{0,4}(?:有|的|是|山脉|山系|山)", text):
        return "center"
    if re.search(r"(?:山脉|山系|山).{0,4}纵贯|(?:纵贯|脊梁|脊椎).{0,4}大陆", text):
        return "center"
    if re.search(r"东北.{0,4}(?:有|是|山脉|山系|山)", text):
        return "northeast"
    if re.search(r"西北.{0,4}(?:有|是|山脉|山系|山)", text):
        return "northwest"
    if re.search(r"东南.{0,4}(?:有|是|山脉|山系|山)", text):
        return "southeast"
    if re.search(r"西南.{0,4}(?:有|是|山脉|山系|山)", text):
        return "southwest"
    return None


def _resolve_mountain_orientation(text: str) -> str | None:
    if re.search(r"东西向|横向|东西走向|east.west", text):
        return "east-west"
    if re.search(r"南北向|纵向|南北走向|north.south|纵贯|脊梁", text):
        return "north-south"
    if re.search(r"弧形|弧形走向|arc|弧形山脉|山弧", text):
        return "arc"
    return None


def _resolve_peninsula_location(text: str) -> str | None:
    if re.search(r"东(?:侧|部|面|方|边|海岸).{0,4}(?:伸|延|有|是).{0,4}半岛", text):
        return "east"
    if re.search(r"西(?:侧|部|面|方|边|海岸).{0,4}(?:伸|延|有|是).{0,4}半岛", text):
        return "west"
    if re.search(r"北(?:侧|部|面|方|边|海岸).{0,4}(?:伸|延|有|是).{0,4}半岛", text):
        return "north"
    if re.search(r"南(?:侧|部|面|方|边|海岸).{0,4}(?:伸|延|有|是).{0,4}半岛", text):
        return "south"
    if re.search(r"东西两.{0,6}半岛|两侧各.{0,3}半岛|东西双半岛", text):
        return "east-west"
    return None


def _resolve_archipelago_location(text: str) -> str | None:
    if re.search(r"东(?:北|南|部|面|方)(?:方向|海域|海|洋).{0,4}群岛", text):
        where = re.search(r"东(?:北|南|部|面|方)", text)
        if where:
            orient = where.group()
            if "东北" in orient or orient == "东北":
                return "northeast"
            if "东南" in orient or orient == "东南":
                return "southeast"
            return "east"
    if re.search(r"西(?:北|南|部|面|方)(?:方向|海域|海|洋).{0,4}群岛", text):
        where = re.search(r"西(?:北|南|部|面|方)", text)
        if where:
            orient = where.group()
            if "西北" in orient or orient == "西北":
                return "northwest"
            if "西南" in orient or orient == "西南":
                return "southwest"
            return "west"
    if re.search(r"南(?:部|面|方|海域|海|洋).{0,4}群岛", text):
        return "south"
    if re.search(r"北(?:部|面|方|海域|海|洋).{0,4}群岛", text):
        return "north"
    if re.search(r"中(?:央|部|间|心).{0,4}群岛|内海.{0,4}群岛", text):
        return "center"
    return None


def _resolve_climate_zone(text: str) -> str | None:
    lower = text.lower()
    if re.search(r"地中海(?:气候)?|mediterranean|dry.summer", lower):
        return "mediterranean"
    if re.search(r"热带|赤道|tropical|hot|equatorial|炎热|酷热", lower):
        return "tropical"
    if re.search(r"极地|极寒|寒带|冰封|frozen|polar|arctic|ice|cold|冰冻|寒冷|冻土", lower):
        return "polar"
    if re.search(r"温带|温和|temperate|mild|凉爽|凉", lower):
        return "temperate"
    if re.search(r"亚热带|subtropical", lower):
        return "tropical"
    if re.search(r"干旱|沙漠|arid|dry|desert|干燥", lower):
        return "arid"
    if re.search(r"湿润|多雨|humid|wet|rainy|潮湿|lush", lower):
        return "humid"
    if re.search(r"季风|monsoon", lower):
        return "monsoon"
    return None


def _resolve_terrain_style(text: str) -> str | None:
    if re.search(r"高原|台地|plateau|tableland|high.plain|高平", text):
        return "plateau"
    if re.search(r"盆地|depression|basin|碗状|低洼|凹陷", text):
        return "basin"
    if re.search(r"裂谷|rift|rift.valley|大地裂|分裂谷", text):
        return "rift"
    if re.search(r"峡湾|fjord|fiord|冰川谷|峡湾群", text):
        return "fjord"
    if re.search(r"三角洲|delta|estuary|河口|河口地", text):
        return "delta"
    if re.search(r"环礁|珊瑚|coral|reef|atoll|choral", text):
        return "reef"
    if re.search(r"湖泊|湖群|大?湖|lake|inland.lake", text):
        return "lake-basin"
    if re.search(r"沼泽|湿地|泥炭|swamp|marsh|wetland|bog|fen", text):
        return "wetland"
    if re.search(r"火山|volcanic|volcano|crater|lava", text):
        return "volcanic"
    if re.search(r"丘陵|起伏|rolling|hilly|undulating", text):
        return "hills"
    if re.search(r"冻土|苔原|冰缘|tundra|periglacial|permafrost", text):
        return "tundra"
    if re.search(r"峡谷|深切|沟壑|gorge|canyon|ravine|V形谷|U形谷", text):
        return "canyon"
    if re.search(r"溶蚀|天坑|karst|dissolution|喀斯特|岩溶", text):
        return "karst"
    return None


def extract_srg(text: str) -> dict:
    entities: list[dict] = []
    relations: list[dict] = []

    if re.search(r"大陆|continent|landmass|陆地", text):
        count_match = re.search(r"(一|1|单)(块|片|个|座)", text)
        if count_match or re.search(r"四面环海|只有一|单一大陆|island.continent", text):
            entities.append({"class": "continent", "id": "C1", "count": 1})
        two_match = re.search(r"(两|二|2)(块|片|个|座)", text)
        if two_match:
            entities.append({"class": "continent", "id": "C1", "position": "west"})
            entities.append({"class": "continent", "id": "C2", "position": "east"})
        three_match = re.search(r"(三|3)(块|片|个|座)", text)
        if three_match:
            entities.append({"class": "continent", "id": "C1", "position": "west"})
            entities.append({"class": "continent", "id": "C2", "position": "center"})
            entities.append({"class": "continent", "id": "C3", "position": "east"})
        if not entities:
            entities.append({"class": "continent", "id": "C0", "implicit": True})

    if re.search(r"山(?:脉|系|峰|脊|地|区|峦|岳)|mountains?|orogen|ridge", text):
        entities.append({"class": "mountain_range", "id": "M1"})

    if re.search(r"内海|inland.sea|mediterranean|enclosed.sea", text):
        entities.append({"class": "inland_sea", "id": "S1"})

    if re.search(r"外海|open.sea|开阔海|开阔洋|open.ocean", text):
        entities.append({"class": "open_sea", "id": "O1"})

    if re.search(r"群岛|archipelago|island.chain|island.group", text):
        entities.append({"class": "archipelago", "id": "A1"})

    if re.search(r"半岛|peninsula|cape|promontory", text):
        entities.append({"class": "peninsula", "id": "P1"})

    if re.search(r"海峡|strait|narrows|channel|passage", text):
        entities.append({"class": "strait", "id": "ST1"})

    sea_refs = [e for e in entities if e["class"] in ("inland_sea", "open_sea", "strait")]
    continent_refs = [e for e in entities if e["class"] == "continent"]
    mountain_refs = [e for e in entities if e["class"] == "mountain_range"]
    peninsula_refs = [e for e in entities if e["class"] == "peninsula"]
    archipelago_refs = [e for e in entities if e["class"] == "archipelago"]

    if len(continent_refs) == 2 and sea_refs:
        relations.append({
            "subject": "C1", "predicate": "separated_by",
            "object": sea_refs[0]["id"], "through": "sea",
        })
        relations.append({
            "subject": "C2", "predicate": "separated_by",
            "object": sea_refs[0]["id"], "through": "sea",
        })

    if sea_refs and continent_refs:
        for sea in sea_refs:
            if sea["class"] == "inland_sea":
                mid_idx = len(continent_refs) // 2
                relations.append({
                    "subject": sea["id"], "predicate": "within",
                    "object": continent_refs[mid_idx]["id"],
                })

    if mountain_refs and continent_refs:
        mountain_loc = _resolve_mountain_location(text)
        for mt in mountain_refs:
            if mountain_loc:
                target = continent_refs[0]["id"]
                relations.append({
                    "subject": mt["id"], "predicate": "located_at",
                    "object": target, "location": mountain_loc,
                })

    for pen in peninsula_refs:
        pen_loc = _resolve_peninsula_location(text)
        if pen_loc and continent_refs:
            relations.append({
                "subject": pen["id"], "predicate": "attached_to",
                "object": continent_refs[0]["id"], "location": pen_loc,
            })

    for arc in archipelago_refs:
        arc_loc = _resolve_archipelago_location(text)
        if arc_loc:
            relations.append({
                "subject": arc["id"], "predicate": "near",
                "object": "ocean", "location": arc_loc,
            })

    return {"entities": entities, "relations": relations}


def extract_topo_relations(text: str) -> list[str]:
    topo_predicates: list[str] = []
    if re.search(r"(相接|邻接|毗连|接壤|相邻|紧挨|相邻而居)", text):
        topo_predicates.append("touches")
    if re.search(r"(横穿|贯穿|切割|穿越|跨越|穿过|横跨|纵贯|纵穿)", text):
        topo_predicates.append("crosses")
    if re.search(r"(环绕|包围|围住|四面环|环绕着|包围着)", text):
        topo_predicates.append("contains")
    if re.search(r"(内部|之中|里面|之内|内).{0,3}(大陆|岛屿|陆)", text):
        topo_predicates.append("within")
    if re.search(r"(重叠|叠置|部分覆盖|半覆盖)", text):
        topo_predicates.append("overlaps")
    if re.search(r"(不相交|分离|隔离|远距离|互不相连)", text):
        topo_predicates.append("disjoint")
    if re.search(r"(沿着|顺沿|依傍|沿岸|沿边|延岸|沿海)", text):
        topo_predicates.append("along")
    if re.search(r"(之间|中间|间隔|两.{0,2}之间)", text):
        topo_predicates.append("between")
    return topo_predicates


def compute_plan_confidence(spatial_features: dict, extracted_plan: dict) -> dict:
    checks: list[dict] = []

    fc_count = spatial_features.get("continent_count")
    pc_count = extracted_plan.get("continent_count") or (len(extracted_plan.get("continents", [])) or None)
    if fc_count is not None and pc_count is not None:
        match = fc_count == pc_count
        checks.append({
            "dimension": "Contribution",
            "check": "continent_count",
            "expected": fc_count,
            "actual": pc_count,
            "match": match,
            "grade": "Excellent" if match else "Poor",
        })

    fc_inland_sea = spatial_features.get("has_inland_sea")
    pc_inland_sea = bool(extracted_plan.get("inland_seas"))
    if fc_inland_sea != pc_inland_sea:
        checks.append({
            "dimension": "Feasibility",
            "check": "inland_sea",
            "expected": fc_inland_sea,
            "actual": pc_inland_sea,
            "match": False,
            "grade": "Very Poor" if fc_inland_sea else "Neutral",
            "suggested_revision": "add_inland_sea" if fc_inland_sea else "remove_inland_sea",
        })

    fc_mountains = spatial_features.get("has_mountains")
    pc_mountains = bool(extracted_plan.get("mountains"))
    if fc_mountains != pc_mountains:
        checks.append({
            "dimension": "Contribution",
            "check": "mountains",
            "expected": fc_mountains,
            "actual": pc_mountains,
            "match": False,
            "grade": "Poor" if fc_mountains else "Neutral",
            "suggested_revision": "add_mountains" if fc_mountains else "remove_mountains",
        })

    fc_climate = spatial_features.get("climate_zone")
    pc_climate = extracted_plan.get("profile", {}).get("climate_zone")
    if fc_climate and pc_climate:
        fc_simple = fc_climate.replace("tropical", "tropical").replace("polar", "frozen")
        pc_simple = (pc_climate or "").lower()
        climate_match = fc_simple in pc_simple or pc_simple in fc_simple
        checks.append({
            "dimension": "Feasibility",
            "check": "climate_zone",
            "expected": fc_climate,
            "actual": pc_climate,
            "match": climate_match,
            "grade": "Excellent" if climate_match else "Good",
        })

    grades_order = {"Excellent": 5, "Good": 4, "Neutral": 3, "Poor": 2, "Very Poor": 1}
    overall_score = 0.5
    if checks:
        total = sum(grades_order.get(c["grade"], 3) for c in checks)
        overall_score = round(total / (len(checks) * 5), 3)

    return {"checks": checks, "overall_confidence": overall_score}


def validate_plan(prompt: str, plan: dict) -> dict:
    spatial_features = extract_spatial_features(prompt)
    confidence = compute_plan_confidence(spatial_features, plan)
    srg = extract_srg(prompt)
    topo = extract_topo_relations(prompt)

    feedback: dict = {
        "confidence": confidence,
        "spatial_features": spatial_features,
        "srg": srg,
        "topological_relations": topo,
        "requires_correction": confidence["overall_confidence"] < 0.6,
        "correction_hints": [],
    }

    for check in confidence.get("checks", []):
        if not check.get("match") and check.get("grade") in ("Poor", "Very Poor"):
            if check.get("suggested_revision"):
                feedback["correction_hints"].append(check["suggested_revision"])
            else:
                feedback["correction_hints"].append(
                    f"mismatch_{check['check']}_expected_{check['expected']}_actual_{check['actual']}"
                )

    return feedback
