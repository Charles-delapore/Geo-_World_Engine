from __future__ import annotations

import json
import sys
import time
from typing import Any

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from app.rag.knowledge_base import TerrainKnowledgeBase
from app.rag.init_kb import init_builtin_knowledge_base
from app.core.semantic_mapper import (
    extract_spatial_features,
    extract_srg,
    extract_topo_relations,
    compute_plan_confidence,
    validate_plan,
)

TEST_PROMPTS: list[dict] = [
    {"prompt": "一块四面环海的大陆，北部有东西向山脉，南部温暖多雨", "expected_id": "single-island-north-mts"},
    {"prompt": "中间有内海，东西两岸被大陆包围，北缘有山脉", "expected_id": "east-west-inland-sea-north-mountains"},
    {"prompt": "一东一西两块大陆，中间是开阔海洋，没有内海", "expected_id": "west-east-open-ocean"},
    {"prompt": "两个连在一起的大陆", "expected_id": "single-island-continent"},
    {"prompt": "全球都是群岛，没有大块陆地", "expected_id": "archipelago-tropical"},
    {"prompt": "东方有一块大陆上面有高山，西方没有陆地全是海洋", "expected_id": "east-continent-mountains"},
    {"prompt": "中央有一片大内海，南北两侧都是大陆", "expected_id": "ns-inland-sea"},
    {"prompt": "大陆中间有一道南北向的大山脉，像大地之脊", "expected_id": "spine-mountain-continent"},
    {"prompt": "一个干燥的沙漠星球的超大陆上有峡谷和少量河流", "expected_id": "desert-planet-super"},
    {"prompt": "东西两块大陆中间被开阔海洋隔开，而且整体气候很寒冷有冻土", "expected_id": "ew-open-sea-cold"},
    {"prompt": "南方的独立大陆温和多雨，到处是原始森林和珍兽", "expected_id": "southland-temperate-mts"},
    {"prompt": "地图上总共三块大陆分居西、东、南三个方位", "expected_id": "three-continents-open"},
    {"prompt": "极度湿润的沼泽世界，没有高山，只有稠密河网和湿地", "expected_id": "swamp-wetland-continent"},
    {"prompt": "一块地势很高的大陆，像高原台地，边缘陡峭", "expected_id": "highland-plateau-continent"},
    {"prompt": "冰封的极地群岛，到处是冰川和白熊", "expected_id": "archipelago-polar"},
    {"prompt": "东部大陆有山系造就了雨影，让沙漠蔓延到西海岸", "expected_id": "monsoon-east-coast"},
    {"prompt": "寒冷苔原大陆上遍布冰缘地貌，到处是碎石和冻土丘", "expected_id": "tundra-boreal-continent"},
    {"prompt": "一片被深切峡谷切割的大陆，河流在V形谷中奔涌", "expected_id": "canyon-country-single"},
    {"prompt": "显著的地中海气候：夏季干热，冬季温和多雨，内海两侧各一块大陆", "expected_id": "mediterranean-temperate"},
    {"prompt": "大陆内部有一个巨大的裂谷，像被巨斧劈开的伤痕", "expected_id": "rift-valley-continent"},
    {"prompt": "喀斯特地貌的大陆，溶蚀洼地和天坑遍布", "expected_id": "karst-terrain-continent"},
    {"prompt": "北方是冻土针叶林带，南方是阔叶落叶林温带", "expected_id": "boreal-temperate-split"},
    {"prompt": "一块大陆朝东伸出长长的半岛，古来被称为'龙之首'", "expected_id": "single-island-east-penin-mts"},
    {"prompt": "一块超大陆占据地图绝大部分，中心是难以穿越的大沙漠", "expected_id": "megacontinent-central-desert"},
]

KNOWLEDGE_CASES: list[str] = [
    "什么是Geomorphon地形分类？",
    "内海和外海有什么区别？",
    "裂谷是如何形成的？",
    "喀斯特地貌的特征是什么？",
    "什么是TPI（地形位置指数）？",
    "地中海气候的主要特征是什么？",
    "构造抬升对地形有什么影响？",
]


def run_retrieval_tests(kb: TerrainKnowledgeBase) -> dict:
    results: dict[str, Any] = {"correct": 0, "total": 0, "failures": []}
    for case in TEST_PROMPTS:
        prompt = case["prompt"]
        expected = case["expected_id"]
        retrieved = kb.retrieve_with_distance(prompt, top_k=5)
        results["total"] += 1
        matched = False
        top1 = retrieved[0]["id"] if retrieved else "NONE"
        for r in retrieved:
            if r["id"] == expected:
                matched = True
                break
        if matched:
            results["correct"] += 1
        else:
            results["failures"].append({
                "prompt": prompt,
                "expected": expected,
                "got_top1": top1,
                "got_top5": [r["id"] for r in retrieved[:5]],
            })
    results["accuracy"] = round(results["correct"] / results["total"] * 100, 1) if results["total"] else 0
    return results


def run_semantic_tests() -> dict:
    results: dict[str, Any] = {"tests": {}}
    test_cases = [
        ("中间有内海，北缘有山脉", {"has_inland_sea": True, "has_mountains": True}),
        ("东西两块大陆中间被开阔海洋隔开", {"has_open_sea": True, "is_split_by_sea": True}),
        ("四面环海的大陆", {"is_single_continent": True, "has_open_sea": False, "has_inland_sea": False}),
        ("热带雨林大陆，炎热多雨", {"climate_zone": "tropical"}),
        ("极地冰冻苔原", {"climate_zone": "polar"}),
        ("高山峡谷深切V形谷", {"terrain_style": "canyon"}),
        ("溶蚀洼地和天坑", {"terrain_style": "karst"}),
        ("冻土针叶林带", {"terrain_style": "tundra"}),
        ("地中海气候夏季干热", {"climate_zone": "mediterranean"}),
        ("裂谷地堑大陆被拉伸", {"terrain_style": "rift"}),
        ("两大陆接壤相邻", {"topo": ["touches"]}),
        ("山脉纵贯南北大陆", {"topo": ["crosses"]}),
        ("沿海南部平原", {"topo": ["along"]}),
    ]
    for text, expected_checks in test_cases:
        features = extract_spatial_features(text)
        topo = extract_topo_relations(text)
        srg = extract_srg(text)
        all_ok = True
        mismatches = []
        for key, val in expected_checks.items():
            if key == "topo":
                for pred in val:
                    if pred not in topo:
                        all_ok = False
                        mismatches.append(f"topo_missing_{pred}")
            else:
                actual = features.get(key)
                if actual != val:
                    all_ok = False
                    mismatches.append(f"{key}: expected {val}, got {actual}")
        results["tests"][text] = {"ok": all_ok}
        if mismatches:
            results["tests"][text]["mismatches"] = mismatches
            results["tests"][text]["features"] = features
            results["tests"][text]["topo"] = topo
    return results


def run_knowledge_tests(kb: TerrainKnowledgeBase) -> dict:
    results = {}
    for query in KNOWLEDGE_CASES:
        retrieved = kb.retrieve_with_distance(query, top_k=3)
        results[query] = [r["name"] for r in retrieved]
    return results


def print_report(retrieval: dict, semantic: dict, knowledge: dict):
    print("\n" + "=" * 70)
    print("RAG 系统综合测试报告")
    print("=" * 70)

    print(f"\n📊 检索准确率测试:")
    print(f"   总测试: {retrieval['total']} | 正确: {retrieval['correct']} | 准确率: {retrieval['accuracy']}%")
    if retrieval["failures"]:
        print(f"   失败案例 ({len(retrieval['failures'])}):")
        for f in retrieval["failures"]:
            print(f"     ❌ '{f['prompt'][:50]}...' → 期望 {f['expected']}, 得到 {f['got_top1']}")

    print(f"\n🧠 语义分析测试:")
    total = len(semantic["tests"])
    ok = sum(1 for v in semantic["tests"].values() if v.get("ok"))
    print(f"   总测试: {total} | 通过: {ok} | 通过率: {round(ok/total*100,1)}%")
    for text, result in semantic["tests"].items():
        status = "✅" if result.get("ok") else "❌"
        print(f"   {status} '{text[:60]}'")
        if not result.get("ok"):
            for m in result.get("mismatches", []):
                print(f"       → {m}")

    print(f"\n📚 知识检索测试:")
    for query, recipes in knowledge.items():
        print(f"   查询: '{query}' → 匹配: {recipes[:2]}")

    print("\n" + "=" * 70)


def main():
    print("初始化知识库...")
    init_builtin_knowledge_base(force=True)
    kb = TerrainKnowledgeBase()

    print(f"已加载 {len(kb._recipes)} 个食谱")

    print("\n运行检索准确率测试...")
    retrieval = run_retrieval_tests(kb)

    print("运行语义分析测试...")
    semantic = run_semantic_tests()

    print("运行知识检索测试...")
    knowledge = run_knowledge_tests(kb)

    print_report(retrieval, semantic, knowledge)

    return retrieval, semantic, knowledge


if __name__ == "__main__":
    main()
