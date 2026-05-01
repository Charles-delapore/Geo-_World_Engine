from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

from app.config import settings
from app.rag.failure_cases import FAILURE_CASES
from app.rag.recipe_store import RecipeDB
from app.rag.schemas import TerrainRecipe
from app.rag import terrain_descriptors as td

_EMBED_DIM = 1024
_N_HASHES = 7
_LATIN_PATTERN = re.compile(r"[a-z0-9]+")
_CJK_RANGE_LOW = "\u4e00"
_CJK_RANGE_HIGH = "\u9fff"
_PUNCT_PATTERN = re.compile(r"[，。！？；：、""''（）《》\\[\\]{}…—\\-\\s]+")


class TerrainKnowledgeBase:
    def __init__(self, db_url: str | None = None, use_failure_rerank: bool = True):
        self.db = RecipeDB(db_url or settings.RECIPE_DB_URL)
        self._recipes: list[TerrainRecipe] = []
        self._vectors: np.ndarray | None = None
        self._id_index: dict[str, int] = {}
        self._failure_cache: dict[str, dict] = {}
        self._use_rerank = use_failure_rerank
        self.reload()

    def reload(self) -> None:
        self._recipes = self.db.list_all()
        self._vectors = self._build_matrix(self._recipes) if self._recipes else None
        self._id_index = {r.id: idx for idx, r in enumerate(self._recipes)}
        self._build_failure_cache()

    def _build_failure_cache(self) -> None:
        self._failure_cache = {}
        for case in FAILURE_CASES:
            boost_ids = case.get("boost_recipe_ids", [])
            penalize_ids = case.get("penalize_recipe_ids", [])
            self._failure_cache[case["prompt"]] = {
                "boost_ids": boost_ids,
                "penalize_ids": penalize_ids,
            }

    def retrieve_with_distance(self, user_prompt: str, top_k: int = 3) -> list[dict]:
        if not self._recipes or self._vectors is None:
            return []
        query_vector = _embed_text(user_prompt)
        scores = self._vectors @ query_vector
        if self._use_rerank:
            _apply_failure_rerank(scores, user_prompt, self._id_index, self._failure_cache)
        ranked_indices = np.argsort(scores)[::-1][: max(1, min(top_k, len(self._recipes)))]
        results: list[dict] = []
        for index in ranked_indices:
            recipe = self._recipes[int(index)]
            similarity = float(scores[int(index)])
            results.append({
                "id": recipe.id,
                "name": recipe.name,
                "description": recipe.description,
                "world_plan": recipe.world_plan,
                "similarity": similarity,
                "source": recipe.source,
            })
        return results

    def _build_matrix(self, recipes: list[TerrainRecipe]) -> np.ndarray:
        if not recipes:
            return np.zeros((0, _EMBED_DIM), dtype=np.float32)
        vectors = [_embed_recipe(recipe) for recipe in recipes]
        matrix = np.vstack(vectors).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return (matrix / norms).astype(np.float32)


def _apply_failure_rerank(
    scores: np.ndarray,
    prompt: str,
    id_index: dict[str, int],
    failure_cache: dict[str, dict],
) -> None:
    from difflib import SequenceMatcher
    best_ratio = 0.0
    best_case: dict | None = None
    for case_prompt, case_info in failure_cache.items():
        ratio = SequenceMatcher(None, prompt.strip(), case_prompt.strip()).ratio()
        if ratio > best_ratio and ratio > 0.25:
            best_ratio = ratio
            best_case = case_info
    if best_case is None:
        return
    scale = min(1.0, best_ratio * 2.0)
    for bid in best_case.get("boost_ids", []):
        idx = id_index.get(bid)
        if idx is not None:
            scores[idx] += 0.12 * scale
    for pid in best_case.get("penalize_ids", []):
        idx = id_index.get(pid)
        if idx is not None:
            scores[idx] -= 0.10 * scale


def _embed_recipe(recipe: TerrainRecipe) -> np.ndarray:
    profile = (recipe.world_plan or {}).get("profile") or {}
    climate_hints = (recipe.world_plan or {}).get("climate_hints") or []
    feature_groups = (recipe.world_plan or {}).get("feature_groups") or []
    parts: list[str] = [
        recipe.name,
        recipe.description,
        " ".join(recipe.tags),
        profile.get("layout_template", ""),
        profile.get("sea_style", ""),
        profile.get("terrain_style", ""),
        profile.get("climate_zone", ""),
    ]
    for item in recipe.world_plan.get("continents", []):
        parts.append(str(item.get("position", item.get("location", ""))))
        parts.append(str(item.get("shape_hint", "")))
    for item in recipe.world_plan.get("mountains", []):
        parts.append(str(item.get("location", "")))
        parts.append(str(item.get("orientation", "")))
    for item in recipe.world_plan.get("inland_seas", []):
        parts.append(str(item.get("position", "")))
        parts.append(str(item.get("size_hint", "")))
    for fg in feature_groups:
        parts.append(str(fg.get("feature_type", "")))
        parts.append(str(fg.get("semantic_label", "")))
        parts.append(" ".join(str(p) for p in fg.get("positions", [])))
    parts.extend(str(hint) for hint in climate_hints)
    joined = " ".join(parts)
    for _ in range(3):
        joined = joined + " " + joined
    return _embed_text(joined)


def _embed_text(text: str) -> np.ndarray:
    tokens = _tokenize(text)
    if not tokens:
        return np.zeros(_EMBED_DIM, dtype=np.float32)
    total = len(tokens)
    weights: dict[str, float] = {}
    for token in tokens:
        weights[token] = weights.get(token, 0.0) + 1.0
    idf_boost: dict[str, float] = {
        tok: math.log1p(float(total) / max(count, 1.0))
        for tok, count in weights.items()
    }
    geo_boost = _geo_term_boost(tokens)
    vector = np.zeros(_EMBED_DIM, dtype=np.float32)
    for token, count in weights.items():
        w = (1.0 + math.log1p(count)) * idf_boost.get(token, 1.0)
        w *= geo_boost.get(token, 1.0)
        for h in range(_N_HASHES):
            bucket = _hash_token(token, _EMBED_DIM, seed=h)
            sign = 1.0 if _hash_token(token, _EMBED_DIM, seed=h + _N_HASHES) % 2 == 0 else -1.0
            vector[bucket] += sign * w
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _tokenize(text: str) -> list[str]:
    normalized = (text or "").strip()
    normalized = _PUNCT_PATTERN.sub(" ", normalized)
    latin_tokens = _LATIN_PATTERN.findall(normalized.lower())
    segments = re.split(r"[a-z0-9]+", normalized.lower())
    cjk_text = "".join(segments)
    cjk_tokens: list[str] = []
    cjk_chars = [c for c in cjk_text if _CJK_RANGE_LOW <= c <= _CJK_RANGE_HIGH]
    cjk_tokens.extend(cjk_chars)
    for i in range(len(cjk_chars) - 1):
        cjk_tokens.append(cjk_chars[i] + cjk_chars[i + 1])
    for i in range(len(cjk_chars) - 2):
        cjk_tokens.append(cjk_chars[i] + cjk_chars[i + 1] + cjk_chars[i + 2])
    cjk_tokens.extend(_compound_cjk_words(cjk_text))
    alias_tokens = _semantic_alias_tokens(normalized)
    all_tokens = latin_tokens + cjk_tokens + alias_tokens
    lat_part_tokens = _latin_part_tokens(" ".join(all_tokens))
    return all_tokens + lat_part_tokens


def _latin_part_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    aliases = {
        "mountains": ["mountain-range", "orogen", "highland"],
        "mountain": ["peak", "ridge", "uplift"],
        "continent": ["landmass", "plate"],
        "ocean": ["sea", "water-body"],
        "sea": ["ocean", "marine"],
        "island": ["isle", "isolated-land"],
        "peninsula": ["cape", "promontory"],
        "archipelago": ["island-chain", "island-group"],
        "desert": ["arid", "dryland"],
        "forest": ["woodland", "jungle"],
        "river": ["stream", "watercourse"],
        "lake": ["inland-water", "basin-water"],
        "valley": ["basin", "lowland"],
        "plateau": ["tableland", "high-plain"],
        "plain": ["flatland", "lowland"],
        "coast": ["shoreline", "seaboard"],
        "strait": ["channel", "narrows"],
        "bay": ["gulf", "inlet"],
        "cape": ["headland", "promontory"],
        "delta": ["estuary", "mouth"],
        "glacier": ["ice-sheet", "frozen"],
        "volcano": ["peak", "caldera"],
        "tundra": ["permafrost", "frozen-plain"],
        "savanna": ["grassland", "prairie"],
        "reef": ["atoll", "coral"],
        "fjord": ["inlet", "glacial-valley"],
        "ridge": ["ridgeline", "divide", "crest"],
        "slope": ["hillside", "gradient", "incline"],
        "basin": ["depression", "hollow", "lowland"],
        "canyon": ["gorge", "ravine", "chasm"],
        "cliff": ["bluff", "escarpment", "precipice"],
        "isthmus": ["land-bridge", "neck", "connection"],
        "archipelagos": ["island-chains", "island-groups"],
        "wetland": ["marsh", "swamp", "bog"],
        "rainforest": ["jungle", "tropical-forest"],
        "steppe": ["grassland", "prairie", "semi-arid-plain"],
        "mediterranean": ["inland-sea", "enclosed-basin"],
        "taiga": ["boreal-forest", "coniferous-belt"],
        "shelf": ["continental-shelf", "shallow-sea"],
        "trench": ["deep-sea", "subduction-trench"],
        "ridge-line": ["divide", "crest-line", "backbone"],
        "erosion": ["weathering", "denudation", "wearing-down"],
        "orogen": ["mountain-belt", "fold-mountains", "tectonic-range"],
        "arc": ["island-arc", "curved-chain", "volcanic-arc"],
        "rift": ["graben", "extensional-basin", "fault-valley"],
    }
    for phrase, expands in aliases.items():
        if phrase in text.lower():
            tokens.extend(expands)
    return tokens


def _compound_cjk_words(cjk_text: str) -> list[str]:
    patterns = [
        r"大陆", r"山脉", r"群岛", r"半岛", r"岛屿", r"平原",
        r"高原", r"盆地", r"沙漠", r"森林", r"草原", r"沼泽",
        r"内海", r"外海", r"海峡", r"海湾", r"海岸", r"洋流",
        r"冰川", r"火山", r"河流", r"湖泊", r"丘陵", r"峡谷",
        r"台地", r"冻土", r"绿洲", r"三角洲", r"珊瑚礁",
        r"东侧", r"西侧", r"南侧", r"北侧", r"东部", r"西部",
        r"南部", r"北部", r"中部", r"中央", r"东北", r"西北",
        r"东南", r"西南", r"中间", r"边缘", r"两侧", r"四周",
        r"沿海", r"内陆", r"极地", r"赤道", r"热带", r"温带",
        r"寒带", r"亚热带", r"干旱", r"湿润", r"多雨", r"季风",
        r"四面环海", r"被海隔开", r"隔着海", r"中间隔着",
        r"宽阔的", r"狭长的", r"巨大的", r"细小的", r"高耸的",
        r"低洼的", r"起伏的", r"平坦的", r"崎岖的", r"陡峭的",
        r"连绵的", r"孤立的", r"分散的", r"密集的", r"稀疏的",
        r"超大陆", r"次大陆", r"微型", r"巨型",
    ]
    patterns.extend(td.get_geomorphon_compound_words())
    patterns.extend(td.get_synthesis_compound_words())
    compounds: list[str] = []
    for pat in patterns:
        idx = 0
        while True:
            idx = cjk_text.find(pat, idx)
            if idx == -1:
                break
            compounds.append(pat)
            if len(pat) >= 1:
                compounds.append("CMP:" + pat)
            idx += 1
    return compounds


_HASH_CACHE: dict[str, int] = {}

def _hash_token(token: str, modulus: int, seed: int = 0) -> int:
    key = f"{seed}:{token}"
    if key in _HASH_CACHE:
        return _HASH_CACHE[key] % modulus
    digest = hashlib.sha256(f"{seed}:{token}".encode("utf-8")).hexdigest()
    val = int(digest[:16], 16)
    result = val % modulus
    _HASH_CACHE[key] = result
    return result


def _geo_term_boost(tokens: list[str]) -> dict[str, float]:
    high_boost_terms: set[str] = {
        "大陆", "山脉", "内海", "外海", "群岛", "半岛", "海峡",
        "超大陆", "被海隔开", "四面环海", "隔着海", "中间隔着",
        "中央内海", "巨型", "微",
        "裂谷", "地峡", "分水岭", "造山带", "缝合带",
        "山脊", "陡崖", "破火山口",
    }
    med_boost_terms: set[str] = {
        "高原", "盆地", "沙漠", "平原", "森林", "草原", "岛屿",
        "海湾", "海岸", "冰川", "火山", "河流", "湖泊",
        "山坡", "山麓", "洼地", "河谷", "河漫滩",
        "陨石坑", "针叶林", "落叶阔叶林",
    }
    geo_prefixes = {"continent", "mountain", "sea", "ocean", "island",
                     "peninsula", "archipelago", "strait", "desert",
                     "mountain-range", "orogen", "split",
                     "rift", "ridge", "valley", "plateau", "basin",
                     "cliff", "canyon", "isthmus", "fjord",
                     "caldera", "volcanic", "delta", "estuary",
                     "tundra", "boreal", "taiga", "savanna",
                     "rainforest", "steppe", "mangrove", "wetland",
                     "erosion", "weathering", "deposition",
                     "aeolian", "glacial", "fluvial", "karst",
                     "arc", "trench", "shelf", "atoll"}
    result: dict[str, float] = {}
    for tok in tokens:
        boost = 1.0
        if tok in high_boost_terms:
            boost = 2.5
        elif tok in med_boost_terms:
            boost = 1.8
        elif any(tok.startswith(p) for p in geo_prefixes):
            boost = 1.6
        elif tok.startswith("CMP:"):
            boost = 2.0
        if tok in {"东侧", "西侧", "南侧", "北侧", "东部", "西部",
                    "南部", "北部", "中部", "东北", "西北", "东南", "西南"}:
            boost = max(boost, 1.5)
        result[tok] = boost
    return result


def _semantic_alias_tokens(text: str) -> list[str]:
    aliases = _build_alias_map()
    derived: list[str] = []
    matched: set[str] = set()
    sorted_phrases = sorted(aliases.keys(), key=len, reverse=True)
    for phrase in sorted_phrases:
        if phrase in text and phrase not in matched:
            derived.extend(aliases[phrase])
            matched.add(phrase)
    derived.extend(_spatial_relation_aliases(text))
    derived = list(set(derived))
    if not derived:
        derived.extend(_fallback_spatial(text))
    return derived


def _build_alias_map() -> dict[str, list[str]]:
    base = _CORE_ALIASES.copy()
    for k, v in _EXTENDED_ALIASES.items():
        if k in base:
            base[k] = list(set(base[k] + v))
        else:
            base[k] = v
    return base


_CORE_ALIASES: dict[str, list[str]] = {
    "内海": ["inland-sea", "mediterranean", "enclosed-sea"],
    "中间有内海": ["inland-sea", "mediterranean", "central-basin", "split-continent"],
    "中央内海": ["inland-sea", "mediterranean", "central-basin"],
    "中部内海": ["inland-sea", "mediterranean", "central-basin"],
    "巨大内海": ["inland-sea", "mediterranean", "central-basin", "large-inland-sea"],
    "被海隔开": ["open-sea", "split-continent"],
    "被海洋隔开": ["open-sea", "split-continent", "ocean-corridor"],
    "隔着海": ["open-sea", "split-continent"],
    "隔着海洋": ["split-continent", "ocean-corridor"],
    "中间隔着海": ["open-sea", "split-continent", "ocean-corridor"],
    "中间被海隔开": ["open-sea", "split-continent", "ocean-corridor"],
    "中间被开阔海洋隔开": ["open-sea", "split-continent", "ocean-corridor"],
    "中间有开阔海洋": ["open-sea", "split-continent", "ocean-corridor", "open-ocean"],
    "两块陆地被海隔开": ["open-sea", "split-continent", "dual-continent"],
    "两块大陆被海隔开": ["split-continent", "dual-continent-east-west"],
    "开阔海洋": ["open-sea", "open-ocean"],
    "外海": ["open-sea", "open-ocean"],
    "海峡": ["strait", "narrow-connection", "narrows"],
    "半岛": ["peninsula"],
    "东侧伸出半岛": ["east-peninsula"],
    "西侧有半岛": ["west-peninsula"],
    "南侧有半岛": ["south-peninsula"],
    "北侧有半岛": ["north-peninsula"],
    "东部伸出半岛": ["east-peninsula"],
    "西部有半岛": ["west-peninsula"],
    "群岛": ["archipelago", "island-chain"],
    "四面环海": ["single-island", "island-continent"],
    "一块四面环海的大陆": ["single-island", "island-continent", "one-continent-island"],
    "四面都是海": ["single-island", "island-continent"],
    "被海洋包围": ["single-island", "island-continent"],
    "东部大陆": ["east-continent"],
    "西部大陆": ["west-continent"],
    "一东一西两块大陆": ["split-east-west", "dual-continent", "two-continents-east-west"],
    "一南一北两块大陆": ["split-north-south", "dual-continent", "two-continents-north-south"],
    "南北两块大陆": ["split-north-south", "dual-continent"],
    "东西两块大陆": ["split-east-west", "dual-continent"],
    "北部山脉": ["north-mountains"],
    "东部有山脉": ["east-mountains"],
    "东部大陆有山脉": ["east-mountains", "east-orogen"],
    "西侧有山脉": ["west-mountains", "west-orogen"],
    "西部有山脉": ["west-mountains", "west-orogen"],
    "南部有山脉": ["south-mountains", "south-rim"],
    "北部有山脉": ["north-mountains", "north-rim"],
    "北缘有山脉": ["north-mountains", "north-rim"],
    "西侧伸出半岛": ["west-peninsula"],
    "东侧伸出": ["east-extension"],
    "西侧伸出": ["west-extension"],
    "超大陆": ["supercontinent", "single-continent"],
    "巨型超大陆": ["supercontinent", "single-continent", "mega"],
    "干旱": ["arid", "dry", "desert-biome"],
    "热带": ["tropical", "warm"],
    "寒冷": ["frozen", "cold", "ice-biome"],
    "高山": ["alpine", "mountainous"],
    "丘陵": ["hilly", "rolling-hills"],
    "平原": ["flat", "lowland", "plain"],
    "盆地": ["basin", "depression"],
    "高原": ["plateau", "highland", "tableland"],
    "沙漠": ["desert", "arid"],
    "森林": ["forest", "woodland", "jungle"],
    "草原": ["grassland", "savanna", "prairie"],
    "沼泽": ["swamp", "marsh", "wetland"],
    "冰川": ["glacier", "ice-sheet", "frozen"],
    "火山": ["volcano", "volcanic", "peak"],
    "河流": ["river", "stream", "waterway"],
    "湖泊": ["lake", "inland-water", "basin-water"],
    "海岸": ["coast", "shore", "seaboard"],
    "海湾": ["bay", "gulf", "inlet"],
    "珊瑚礁": ["reef", "atoll", "coral", "barrier"],
    "三角洲": ["delta", "estuary", "river-mouth"],
    "极地": ["polar", "frozen", "ice-cap"],
    "温带": ["temperate"],
    "寒带": ["subpolar", "boreal"],
    "赤道": ["equatorial", "tropical"],
    "季风": ["monsoon", "seasonal-rain"],
    "多雨": ["rainy", "wet", "humid"],
    "湿润": ["humid", "wet"],
    "冻土": ["frozen", "permafrost", "cold", "tundra"],
    "冰冻": ["frozen", "ice", "glacier"],
    "冰蚀": ["glacial-erosion", "ice-scouring", "frozen"],
    "风蚀": ["aeolian", "wind-transport", "deflation"],
    "风积": ["aeolian", "dune-formation", "sand-deposit"],
    "海蚀": ["coastal-processes", "wave-erosion"],
    "溶蚀": ["karst", "dissolution", "chemical-weathering"],
    "山坡": ["slope", "hillside", "mountain-side"],
    "山脊": ["ridge", "ridgeline", "divide"],
    "山麓": ["footslope", "pediment", "lower-slope"],
    "洼地": ["hollow", "concave", "headwater-basin", "depression"],
    "河谷": ["valley", "river-valley", "fluvial-valley"],
    "河漫滩": ["floodplain", "alluvial-plain", "overbank"],
    "陡崖": ["cliff", "escarpment", "bluff"],
    "裂谷": ["rift-valley", "graben", "extensional-basin"],
    "陨石坑": ["impact-crater", "astrobleme", "meteor-crater"],
    "破火山口": ["caldera", "volcanic-collapse", "crater-lake"],
    "针叶林": ["coniferous", "evergreen", "taiga"],
    "落叶阔叶林": ["temperate-forest", "deciduous", "broadleaf"],
    "硬叶林": ["mediterranean-scrub", "chaparral", "sclerophyll"],
    "红树林": ["mangrove", "coastal-swamp", "tidal-forest"],
    "西风带": ["westerlies", "temperate-zone"],
    "信风": ["trade-winds", "tropical-easterlies"],
    "洋流": ["ocean-current", "marine-flow", "current"],
    "暖流": ["warm-current", "heat-transport"],
    "寒流": ["cold-current", "nutrient-upwelling"],
    "迎风": ["windward", "wind-facing"],
    "背风": ["leeward", "rain-shadow"],
    "阴坡": ["shaded-slope", "north-facing"],
    "阳坡": ["sunny-slope", "south-facing"],
    "分水岭": ["watershed-divide", "drainage-divide", "ridgeline"],
    "冰缘": ["periglacial", "frost-action"],
    "地峡": ["isthmus", "land-bridge", "neck"],
}


_EXTENDED_ALIASES: dict[str, list[str]] = {
    "单一": ["single-element", "solitary", "one"],
    "两块": ["two-elements", "dual", "pair"],
    "三块": ["three-elements", "triple", "triad"],
    "多块": ["multiple-elements", "multi", "many"],
    "狭长": ["elongated", "narrow", "linear"],
    "宽阔": ["wide", "broad", "expansive"],
    "巨大": ["huge", "large-sized", "mega"],
    "小型": ["small", "tiny", "compact"],
    "高耸": ["tall", "high-elevation", "alpine"],
    "低洼": ["low-elevation", "depressed", "basin"],
    "起伏": ["undulating", "rolling", "varied-terrain"],
    "平坦": ["flat", "level", "even"],
    "崎岖": ["rugged", "rough", "irregular"],
    "陡峭": ["steep", "sharp-relief", "cliff"],
    "连绵": ["continuous", "chain", "serial"],
    "孤立": ["isolated", "standalone", "solo"],
    "分散": ["scattered", "distributed", "sparse-layout"],
    "密集": ["dense", "clustered", "concentrated"],
    "稀疏": ["sparse", "scattered", "thinly-spread"],
    "微型": ["micro", "tiny", "mini"],
    "巨型": ["mega", "giant", "colossal"],
    "次大陆": ["subcontinent", "sub-landmass"],
    "边缘": ["edge", "fringe", "periphery"],
    "四周": ["all-around", "peripheral", "encircling"],
    "沿海": ["coastal", "littoral", "seaboard"],
    "内陆": ["inland", "interior", "heartland"],
    "东侧": ["east-side", "eastern-edge"],
    "西侧": ["west-side", "western-edge"],
    "南侧": ["south-side", "southern-edge"],
    "北侧": ["north-side", "northern-edge"],
    "东部": ["east-region", "eastern"],
    "西部": ["west-region", "western"],
    "南部": ["south-region", "southern"],
    "北部": ["north-region", "northern"],
    "中部": ["central-region", "center", "mid-section"],
    "中央": ["center", "core", "central"],
    "东北": ["northeast", "ne-corner"],
    "西北": ["northwest", "nw-corner"],
    "东南": ["southeast", "se-corner"],
    "西南": ["southwest", "sw-corner"],
    "中间": ["middle", "between", "interposed"],
    "两侧": ["both-sides", "bilateral"],
    "山脉纵贯": ["north-south-range", "spine-ridge", "ns-orogen"],
    "山脉横贯": ["east-west-range", "transverse-range", "ew-orogen"],
    "山系": ["mountain-system", "orogenic-belt", "range-complex"],
    "火山带": ["volcanic-arc", "volcanic-range"],
    "岛弧": ["island-arc", "arc-system"],
    "海沟": ["ocean-trench", "deep-trench"],
    "海底": ["seafloor", "ocean-floor", "submarine"],
    "陆桥": ["land-bridge", "isthmus", "connection"],
    "峡湾": ["fjord", "glacial-valley", "inlet"],
    "不规整": ["irregular", "uneven", "jagged"],
    "规整": ["regular", "ordered", "symmetric"],
    "南端": ["south-end", "south-tip"],
    "北端": ["north-end", "north-tip"],
    "东端": ["east-end", "east-tip"],
    "西端": ["west-end", "west-tip"],
    "沿海平原": ["coastal-plain", "coastal-lowland"],
    "内陆高原": ["inland-plateau", "central-plateau"],
    "北高南低": ["north-high-south-low", "north-steep"],
    "东高西低": ["east-high-west-low"],
    "西高东低": ["west-high-east-low"],
    "南高北低": ["south-high-north-low"],
    "大陆架": ["continental-shelf"],
    "分割": ["divided", "partitioned", "cut"],
    "相连": ["connected", "joined", "attached"],
    "断裂": ["fractured", "rifted", "split"],
    "连在一起": ["connected", "single-mass", "adjoined"],
    "分开": ["separated", "apart", "disjoined"],
    "环状": ["ring-shape", "circular", "encircling"],
    "带状": ["banded", "zonal", "belt"],
    "网状": ["network", "web", "mesh"],
    "楔形": ["wedge", "tapered"],
    "风成": ["aeolian", "wind-driven"],
    "冰成": ["glacial", "ice-driven"],
    "水成": ["fluvial", "water-driven"],
    "海成": ["marine", "ocean-driven"],
    "波状": ["wavy", "undulating-pattern"],
    "梯级": ["stepped", "terraced"],
    "羽状": ["feather", "pinnate-pattern"],
    "指状": ["digitate", "finger-pattern"],
    "扇状": ["fan-shaped", "deltaic"],
    "放射状": ["radial", "centrifugal"],
    "汇聚状": ["convergent", "centripetal"],
    "湖泊型": ["lacustrine", "lake-type"],
    "海洋型": ["marine-type", "ocean-dominated"],
    "山脉型": ["orogenic-type", "mountain-dominated"],
    "岛弧型": ["island-arc-type", "arc-setting"],
    "缝合带": ["suture-zone", "collision-belt"],
    "造山带": ["orogenic-belt", "mountain-belt", "fold-belt"],
    "俯冲带": ["subduction-zone", "convergent-margin"],
    "沉降带": ["subsidence-zone", "basin-forming"],
    "隆起带": ["uplift-zone", "dome-region"],
    "流域": ["watershed", "drainage-basin", "catchment"],
    "分水界": ["drainage-divide", "watershed-boundary"],
}


def _spatial_relation_aliases(text: str) -> list[str]:
    tokens: list[str] = []
    if re.search(r"(东|西)(侧|部)(有|伸出|延伸).*(半岛|陆地|大陆)", text):
        tokens.append("east-extension")
    if re.search(r"(西|西侧)(有|伸出|延伸).*(半岛|陆地|大陆)", text):
        tokens.append("west-extension")
    if re.search(r"(南|南侧)(有|伸出|延伸).*(半岛|陆地|大陆)", text):
        tokens.append("south-extension")
    if re.search(r"(北|北侧)(有|伸出|延伸).*(半岛|陆地|大陆)", text):
        tokens.append("north-extension")
    if re.search(r"(东|东部|东侧).*(山脉|山系|山)", text) or re.search(r"(山脉|山系|山).*(东|东部|东侧)", text):
        tokens.append("east-mountains")
    if re.search(r"(西|西部|西侧).*(山脉|山系|山)", text) or re.search(r"(山脉|山系|山).*(西|西部|西侧)", text):
        tokens.append("west-mountains")
    if re.search(r"(南|南部|南侧).*(山脉|山系|山)", text) or re.search(r"(山脉|山系|山).*(南|南部|南侧)", text):
        tokens.append("south-mountains")
    if re.search(r"(北|北部|北侧).*(山脉|山系|山)", text) or re.search(r"(山脉|山系|山).*(北|北部|北侧)", text):
        tokens.append("north-mountains")
    if re.search(r"(东|东部|东侧).*(海|海洋|内海)", text):
        tokens.append("east-sea")
    if re.search(r"(西|西部|西侧).*(海|海洋|内海)", text):
        tokens.append("west-sea")
    if re.search(r"中央.*(海|盆地|平原|高原)", text):
        tokens.append("central-feature")
    if re.search(r"(被|隔着|隔开).*(海|海洋|水)", text) or re.search(r"(海|海洋|水).*(隔开|分开|分割)", text):
        tokens.append("split-continent-by-sea")
    if re.search(r"(一块|单一|单个|只有一个|仅一个).*(大陆|陆地)", text):
        tokens.append("single-continent")
    if re.search(r"两(块|片|个).*(大陆|陆地)", text):
        tokens.append("dual-continent")
    if re.search(r"三(块|片|个).*(大陆|陆地)", text):
        tokens.append("three-continents")
    if re.search(r"(纵横|贯穿|沿).*(大陆|陆地|中部)", text):
        tokens.append("spine-ridge")
    if re.search(r"(纵向|南北).*(山脉|山系|走向)", text):
        tokens.append("ns-orogen")
    if re.search(r"(横向|东西).*(山脉|山系|走向)", text):
        tokens.append("ew-orogen")
    if re.search(r"(寒冷|冰冻|冻土|严寒|冰封|极地)", text):
        tokens.append("frozen")
        tokens.append("cold")
        tokens.append("polar")
    if re.search(r"(相接|邻接|毗连|接壤|相邻|紧挨)", text):
        tokens.append("touches")
        tokens.append("adjacent")
    if re.search(r"(横穿|贯穿|切割|穿越|跨越|穿过)", text):
        tokens.append("crosses")
        tokens.append("traverses")
    if re.search(r"(环绕|包围|围住|四面环|环绕着)", text):
        tokens.append("contains")
        tokens.append("encircles")
    if re.search(r"(内部|之中|里面|之内|内)", text) and re.search(r"(大陆|岛屿|陆)", text):
        tokens.append("within")
    if re.search(r"(重叠|叠置|部分覆盖|半覆盖)", text):
        tokens.append("overlaps")
    if re.search(r"(不相交|分离|隔离|远距离)", text):
        tokens.append("disjoint")
    if re.search(r"(沿着|顺沿|依傍|沿岸|沿边)", text):
        tokens.append("along")
    if re.search(r"(放射|发散|向外|离心)", text):
        tokens.append("radial-pattern")
    if re.search(r"(汇聚|向心|收拢|集中)", text):
        tokens.append("convergent-pattern")
    return list(set(tokens))


def _fallback_spatial(text: str) -> list[str]:
    tokens: list[str] = []
    if "东" in text and "侧" in text:
        tokens.append("eastwards")
    if "西" in text and "侧" in text:
        tokens.append("westwards")
    if "南" in text and "侧" in text:
        tokens.append("southwards")
    if "北" in text and "侧" in text:
        tokens.append("northwards")
    if "中" in text and "间" in text:
        tokens.append("between-two")
    if "山脉" in text:
        tokens.append("has-mountains")
    if "内海" in text:
        tokens.append("has-inland-sea")
    return tokens
