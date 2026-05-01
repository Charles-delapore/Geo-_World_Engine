from __future__ import annotations

from difflib import SequenceMatcher

FAILURE_CASES: list[dict] = [
    {
        "prompt": "一块四面环海的大陆，北部有东西向山脉，南部温暖多雨",
        "expected_recipe_id": "single-island-north-mts",
        "fail_description": "检索为南北双大陆而非单岛型。",
        "corrective_aliases": ["single-island", "north-mountains", "south-warm", "四面环海"],
        "boost_recipe_ids": ["single-island-north-mts"],
    },
    {
        "prompt": "中间有内海，东西两岸被大陆包围，北缘有山脉",
        "expected_recipe_id": "east-west-inland-sea-north-mountains",
        "fail_description": "未能准确识别内海布局中的北缘山脉模式。",
        "corrective_aliases": ["inland-sea", "north-mountains", "mediterranean", "north-rim", "中间有内海", "北缘有山脉"],
        "boost_recipe_ids": ["east-west-inland-sea-north-mountains", "north-rim-mountains-inland"],
    },
    {
        "prompt": "一东一西两块大陆，中间是开阔海洋，没有内海",
        "expected_recipe_id": "west-east-open-ocean",
        "fail_description": "对'开阔海洋'和'内海'的区分不够显著。",
        "corrective_aliases": ["open-ocean", "split-east-west", "被开阔海洋隔开"],
        "boost_recipe_ids": ["west-east-open-ocean"],
        "penalize_recipe_ids": ["west-east-inland-sea"],
    },
    {
        "prompt": "两个连在一起的大陆",
        "expected_recipe_id": "single-island-continent",
        "fail_description": "'连在一起'应理解为单岛大陆但被曲解为双大陆。",
        "corrective_aliases": ["single-island", "connected-landmass"],
        "boost_recipe_ids": ["single-island-continent", "supercontinent-arid-interior"],
        "penalize_recipe_ids": ["west-east-open-ocean", "north-south-continental-split"],
    },
    {
        "prompt": "全球都是群岛，没有大块陆地",
        "expected_recipe_id": "archipelago-tropical",
        "fail_description": "全群岛模式未得到优先匹配，有时会误检索为单岛。",
        "corrective_aliases": ["archipelago", "all-islands", "ocean-world", "群岛", "没有大块陆地"],
        "boost_recipe_ids": ["archipelago-tropical", "archipelago-global-ring"],
        "penalize_recipe_ids": ["single-island-continent"],
    },
    {
        "prompt": "东方有一块大陆上面有高山，西方没有陆地全是海洋",
        "expected_recipe_id": "east-continent-mountains",
        "fail_description": "东大陆有山而西侧无陆地的不对称布局匹配困难。",
        "corrective_aliases": ["east-continent", "east-mountains", "单一大陆", "东方有一块大陆"],
        "boost_recipe_ids": ["east-continent-mountains"],
    },
    {
        "prompt": "中央有一片大内海，南北两侧都是大陆",
        "expected_recipe_id": "ns-inland-sea",
        "fail_description": "中央内海未正确区分开海与内海。",
        "corrective_aliases": ["inland-sea", "central-basin", "南北大陆", "有内海"],
        "boost_recipe_ids": ["ns-inland-sea"],
    },
    {
        "prompt": "大陆中间有一道南北向的大山脉，像大地之脊",
        "expected_recipe_id": "spine-mountain-continent",
        "fail_description": "'大地之脊'纵向主山脉未被正确抓取。",
        "corrective_aliases": ["spine-ridge", "north-south-mountains", "大陆脊梁", "山脉纵贯", "像大地之脊"],
        "boost_recipe_ids": ["spine-mountain-continent"],
    },
    {
        "prompt": "一个干燥的沙漠星球的超大陆上有峡谷和少量河流",
        "expected_recipe_id": "desert-planet-super",
        "fail_description": "干旱超大陆检索偏弱。",
        "corrective_aliases": ["desert-planet", "arid-supercontinent", "干燥", "沙漠", "干旱"],
        "boost_recipe_ids": ["desert-planet-super", "megacontinent-central-desert"],
    },
    {
        "prompt": "东西两块大陆中间被开阔海洋隔开，而且整体气候很寒冷有冻土",
        "expected_recipe_id": "ew-open-sea-cold",
        "fail_description": "'中间被开阔海洋隔开'和'寒冷'的组合需要更高精度的匹配。",
        "corrective_aliases": ["split-east-west", "open-ocean", "frozen", "cold", "被开阔海洋隔开", "寒冷", "冻土"],
        "boost_recipe_ids": ["ew-open-sea-cold"],
    },
    {
        "prompt": "一块大陆朝东伸出长长的半岛，古来被称为'龙之首'",
        "expected_recipe_id": "single-island-east-penin-mts",
        "fail_description": "东伸半岛带有比喻的文本难以被准确的问句解析。",
        "corrective_aliases": ["east-peninsula", "东伸出半岛", "东半岛"],
        "boost_recipe_ids": ["single-island-east-penin-mts", "single-island-east-peninsula"],
    },
    {
        "prompt": "南方的独立大陆温和多雨，到处是原始森林和珍兽",
        "expected_recipe_id": "southland-temperate-mts",
        "fail_description": "南方独立大陆和河流森林特征需要较多匹配规则。",
        "corrective_aliases": ["south-continent", "forest", "temperate", "森林", "多雨", "南方大陆"],
        "boost_recipe_ids": ["southland-temperate-mts"],
    },
    {
        "prompt": "地图上总共三块大陆分居西、东、南三个方位",
        "expected_recipe_id": "three-continents-open",
        "fail_description": "三块大陆分散模式未优先响应。",
        "corrective_aliases": ["multi-continent", "三块大陆", "分居三个方位"],
        "boost_recipe_ids": ["three-continents-open"],
    },
    {
        "prompt": "极度湿润的沼泽世界，没有高山，只有稠密河网和湿地",
        "expected_recipe_id": "swamp-wetland-continent",
        "fail_description": "沼泽湿地世界没有被RAG正确表征。",
        "corrective_aliases": ["swamp", "wetland", "no-mountains", "沼泽", "没有高山", "湿地"],
        "boost_recipe_ids": ["swamp-wetland-continent"],
    },
    {
        "prompt": "一块地势很高的大陆，像高原台地，边缘陡峭",
        "expected_recipe_id": "highland-plateau-continent",
        "fail_description": "'高原台地'和边缘陡峭未映射到匹配的plateau标签。",
        "corrective_aliases": ["plateau", "highland", "高原", "台地", "像高原台地", "边缘陡峭"],
        "boost_recipe_ids": ["highland-plateau-continent"],
    },
    {
        "prompt": "一块超大陆占据地图绝大部分，中心是难以穿越的大沙漠",
        "expected_recipe_id": "megacontinent-central-desert",
        "fail_description": "中心沙漠使超大陆的中央盆地类检索有杂音。",
        "corrective_aliases": ["supercontinent", "central-desert", "地图绝大部分", "占据大部分", "中心是沙漠"],
        "boost_recipe_ids": ["megacontinent-central-desert"],
    },
    {
        "prompt": "有四块大陆像四大天王镇守东西南北",
        "expected_recipe_id": "four-continents-scatter",
        "fail_description": "比喻修辞削弱四方大陆模式的检索。",
        "corrective_aliases": ["multi-continent", "四方大陆", "四块大陆", "东西南北"],
        "boost_recipe_ids": ["four-continents-scatter"],
    },
    {
        "prompt": "宽宽的海洋两边都有大陆，南边的大陆边缘山很高",
        "expected_recipe_id": "ew-open-sea-south-mts",
        "fail_description": "'宽宽的海洋两边都有大陆'+南边山脉的组合。",
        "corrective_aliases": ["split-east-west", "open-ocean", "south-mountains", "南部有山脉", "中间是海洋", "南边的大陆"],
        "boost_recipe_ids": ["ew-open-sea-south-mts"],
    },
    {
        "prompt": "冰封的极地群岛，到处是冰川和白熊",
        "expected_recipe_id": "archipelago-polar",
        "fail_description": "极地群岛需要明确标注冻结标签而非普通群岛。",
        "corrective_aliases": ["archipelago", "polar", "frozen", "极地", "冰封", "冰川"],
        "boost_recipe_ids": ["archipelago-polar"],
    },
    {
        "prompt": "东部大陆有山系造就了雨影，让沙漠蔓延到西海岸",
        "expected_recipe_id": "monsoon-east-coast",
        "fail_description": "雨影/季风系统未参与嵌入匹配。",
        "corrective_aliases": ["monsoon", "east-mountains", "rain-shadow", "季风", "雨影", "东部大陆有山"],
        "boost_recipe_ids": ["monsoon-east-coast"],
    },
    {
        "prompt": "一片被群山环绕的高原，内部平坦但边缘陡峭如悬崖",
        "expected_recipe_id": "highland-plateau-continent",
        "fail_description": "高原+悬崖组合未能正确关联到plateau+cliff双重特征。",
        "corrective_aliases": ["plateau", "cliff", "escarpment", "高原", "悬崖", "边缘陡峭"],
        "boost_recipe_ids": ["highland-plateau-continent"],
    },
    {
        "prompt": "寒冷苔原大陆上遍布冰缘地貌，到处是碎石和冻土丘",
        "expected_recipe_id": "tundra-boreal-continent",
        "fail_description": "苔原+冰缘地貌的检索需要冻结标签与苔原标签联动。",
        "corrective_aliases": ["tundra", "frozen", "periglacial", "苔原", "冻土", "冰缘"],
        "boost_recipe_ids": ["tundra-boreal-continent"],
    },
    {
        "prompt": "一片被深切峡谷切割的大陆，河流在V形谷中奔涌",
        "expected_recipe_id": "canyon-country-single",
        "fail_description": "深切峡谷+V形谷未被正确映射到canyon特征。",
        "corrective_aliases": ["canyon", "gorge", "ravine", "峡谷", "深切", "V形谷"],
        "boost_recipe_ids": ["canyon-country-single"],
    },
    {
        "prompt": "显著的地中海气候：夏季干热，冬季温和多雨，内海两侧各一块大陆",
        "expected_recipe_id": "mediterranean-temperate",
        "fail_description": "地中海气候未被识别为独立的climate_zone。",
        "corrective_aliases": ["mediterranean", "temperate", "inland-sea", "地中海气候", "夏季干热", "冬季温和多雨"],
        "boost_recipe_ids": ["mediterranean-temperate"],
    },
    {
        "prompt": "大陆内部有一个巨大的裂谷，像被巨斧劈开的伤痕",
        "expected_recipe_id": "rift-valley-continent",
        "fail_description": "裂谷描述的比喻干扰了rift特征的识别。",
        "corrective_aliases": ["rift-valley", "graben", "裂谷", "劈开", "伤痕"],
        "boost_recipe_ids": ["rift-valley-continent"],
    },
    {
        "prompt": "一块大陆被一条南北走向的山脊纵贯，像鱼骨一样分出无数侧岭",
        "expected_recipe_id": "spine-mountain-continent",
        "fail_description": "'鱼骨'比喻干扰了脊线/支脊的匹配。",
        "corrective_aliases": ["spine-ridge", "north-south-mountains", "纵贯", "脊线", "支脊", "鱼骨"],
        "boost_recipe_ids": ["spine-mountain-continent"],
    },
    {
        "prompt": "喀斯特地貌的大陆，溶蚀洼地和天坑遍布",
        "expected_recipe_id": "karst-terrain-continent",
        "fail_description": "喀斯特/溶蚀等专业地貌术语未进入检索词库。",
        "corrective_aliases": ["karst", "dissolution", "喀斯特", "溶蚀", "天坑", "洼地"],
        "boost_recipe_ids": ["karst-terrain-continent"],
    },
    {
        "prompt": "北方是冻土针叶林带，南方是阔叶落叶林温带",
        "expected_recipe_id": "boreal-temperate-split",
        "fail_description": "针叶林/阔叶林/泰加林等植被带术语缺失。",
        "corrective_aliases": ["boreal", "taiga", "temperate", "针叶林", "阔叶林", "北方针叶林"],
        "boost_recipe_ids": ["boreal-temperate-split", "tundra-boreal-continent"],
    },
]


class FailureCaseDB:
    def __init__(self) -> None:
        self._cases: list[dict] = list(FAILURE_CASES)

    def count(self) -> int:
        return len(self._cases)

    def find_by_prompt(self, prompt: str, threshold: float = 0.55) -> list[dict]:
        matched: list[dict] = []
        for case in self._cases:
            case_prompt = case.get("prompt", "")
            ratio = SequenceMatcher(None, prompt.lower(), case_prompt.lower()).ratio()
            if ratio >= threshold:
                result = dict(case)
                result["similarity"] = round(ratio, 3)
                matched.append(result)
        matched.sort(key=lambda c: c.get("similarity", 0), reverse=True)
        return matched

