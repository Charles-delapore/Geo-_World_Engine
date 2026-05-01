from __future__ import annotations

GEOMORPHON_TYPES: dict[str, dict] = {
    "flat": {
        "cn": ["平地", "平坦", "平地地形"],
        "aliases": ["flat", "level", "plain", "plateau-top"],
        "description": "Locally flat area with no significant slope in any direction",
        "literature": "jasiewicz_stepinski_2013_geomorphons",
    },
    "peak": {
        "cn": ["山峰", "顶点", "峰顶", "山巅"],
        "aliases": ["peak", "summit", "mountain-top", "pinnacle"],
        "description": "Local maximum, higher than all surrounding cells",
        "literature": "jasiewicz_stepinski_2013_geomorphons",
    },
    "ridge": {
        "cn": ["山脊", "岭", "脊线", "分水岭"],
        "aliases": ["ridge", "ridgeline", "divide", "watershed-boundary"],
        "description": "Elongated high region, convex in one direction, flat or convex in orthogonal",
        "literature": "jasiewicz_stepinski_2013_geomorphons",
    },
    "shoulder": {
        "cn": ["坡肩", "山肩", "上部坡"],
        "aliases": ["shoulder", "upper-slope", "convex-break"],
        "description": "Transition zone between ridge and slope, convex profile curvature",
        "literature": "jasiewicz_stepinski_2013_geomorphons",
    },
    "spur": {
        "cn": ["山嘴", "支脊", "横岭"],
        "aliases": ["spur", "lateral-ridge", "branch-ridge"],
        "description": "Convex in orthogonal direction, flat or convex along profile",
        "literature": "jasiewicz_stepinski_2013_geomorphons",
    },
    "slope": {
        "cn": ["坡地", "山坡", "斜面"],
        "aliases": ["slope", "hillside", "inclined-surface"],
        "description": "Steady inclined surface, neither convex nor concave",
        "literature": "jasiewicz_stepinski_2013_geomorphons",
    },
    "hollow": {
        "cn": ["洼地", "凹地", "汇水凹地"],
        "aliases": ["hollow", "concave", "headwater-basin", "amphitheater"],
        "description": "Concave in orthogonal direction, flat or concave along profile",
        "literature": "jasiewicz_stepinski_2013_geomorphons",
    },
    "footslope": {
        "cn": ["坡脚", "山麓", "下部坡"],
        "aliases": ["footslope", "lower-slope", "concave-break", "pediment"],
        "description": "Transition zone between slope and valley, concave profile curvature",
        "literature": "jasiewicz_stepinski_2013_geomorphons",
    },
    "valley": {
        "cn": ["山谷", "河谷", "沟谷"],
        "aliases": ["valley", "ravine", "gully", "canyon", "gorge"],
        "description": "Elongated low region, concave in one direction, flat or concave in orthogonal",
        "literature": "jasiewicz_stepinski_2013_geomorphons",
    },
    "pit": {
        "cn": ["坑洼", "洼坑", "陷穴", "天坑"],
        "aliases": ["pit", "depression", "sinkhole", "basin-bottom"],
        "description": "Local minimum, lower than all surrounding cells",
        "literature": "jasiewicz_stepinski_2013_geomorphons",
    },
}


TERRAIN_METRICS: dict[str, dict] = {
    "slope": {
        "cn": ["坡度", "倾斜度", "坡降"],
        "aliases": ["slope", "gradient", "inclination", "steepness"],
        "unit": "degrees or percent",
        "description": "Rate of change of elevation - the first derivative of the surface",
    },
    "aspect": {
        "cn": ["坡向", "朝向", "方位角"],
        "aliases": ["aspect", "orientation", "exposure", "azimuth"],
        "unit": "degrees (0-360)",
        "description": "Direction of maximum slope - which way the surface faces",
    },
    "profile_curvature": {
        "cn": ["剖面曲率", "纵向曲率"],
        "aliases": ["profile-curvature", "along-slope-curvature"],
        "unit": "1/m",
        "description": "Curvature in the direction of maximum slope - acceleration/deceleration of flow",
    },
    "plan_curvature": {
        "cn": ["平面曲率", "横向曲率", "等高线曲率"],
        "aliases": ["plan-curvature", "contour-curvature", "across-slope-curvature"],
        "unit": "1/m",
        "description": "Curvature perpendicular to slope direction - flow convergence/divergence",
    },
    "tpi": {
        "cn": ["地形位置指数", "TPI"],
        "aliases": ["tpi", "topographic-position-index"],
        "unit": "meters",
        "description": "Difference between cell elevation and mean neighborhood elevation",
    },
    "tri": {
        "cn": ["地形粗糙度指数", "TRI"],
        "aliases": ["tri", "terrain-ruggedness-index", "roughness"],
        "unit": "meters",
        "description": "Mean absolute difference between cell and its neighbors",
    },
    "flow_accumulation": {
        "cn": ["汇流累积量", "集水面积", "上游集水区"],
        "aliases": ["flow-accumulation", "contributing-area", "upslope-area", "catchment-area"],
        "unit": "number of cells or m2",
        "description": "Number of upslope cells draining into each cell",
    },
    "drainage_area": {
        "cn": ["流域面积", "排水面积"],
        "aliases": ["drainage-area", "watershed-area", "basin-area"],
        "unit": "m2 or km2",
        "description": "Total contributing area including the cell itself",
    },
    "wetness_index": {
        "cn": ["湿润指数", "地形湿度指数", "TWI"],
        "aliases": ["twi", "topographic-wetness-index", "cti", "compound-topographic-index"],
        "unit": "unitless",
        "description": "ln(a/tan(b)) where a is contributing area and b is slope",
    },
    "sky_view_factor": {
        "cn": ["天空可视因子", "开阔度"],
        "aliases": ["svf", "sky-view-factor", "openness"],
        "unit": "0-1",
        "description": "Proportion of visible sky hemisphere from a given point",
    },
    "accessibility": {
        "cn": ["可达性", "可通行性"],
        "aliases": ["accessibility", "traversability", "passability"],
        "unit": "varies",
        "description": "Ease of movement across terrain - used in game world design (Argudo 2025)",
    },
    "landform_entropy": {
        "cn": ["地形熵", "景观多样性"],
        "aliases": ["landform-entropy", "geomorphological-diversity"],
        "unit": "bits",
        "description": "Shannon entropy of geomorphon classes in a window",
    },
}


GEOMORPHIC_UNITS_FLUVIAL: dict[str, dict] = {
    "channel": {
        "cn": ["河道", "河槽", "主河槽"],
        "aliases": ["channel", "main-channel", "thalweg", "active-channel"],
        "description": "Primary water-conveying corridor, often the deepest part of the valley bottom",
        "source": "zhang_fryirs_2024_esp",
    },
    "point_bar": {
        "cn": ["点坝", "凸岸沙坝"],
        "aliases": ["point-bar", "meander-bar", "inner-bank-deposit"],
        "description": "Sediment deposit on the inside of a meander bend",
        "source": "zhang_fryirs_2024_esp",
    },
    "lateral_bar": {
        "cn": ["侧坝", "边滩", "侧向沙坝"],
        "aliases": ["lateral-bar", "side-bar", "bank-attached-bar"],
        "description": "Sediment bar attached to one bank in low-sinuosity reaches",
        "source": "zhang_fryirs_2024_esp",
    },
    "mid_channel_bar": {
        "cn": ["江心洲", "河中沙洲"],
        "aliases": ["mid-channel-bar", "braid-bar", "island-bar"],
        "description": "Sediment accumulation in the middle of the channel",
        "source": "zhang_fryirs_2024_esp",
    },
    "riffle": {
        "cn": ["浅滩", "湍滩"],
        "aliases": ["riffle", "shoal", "shallow-rapid"],
        "description": "Shallow, fast-flowing reach with coarse bed material",
        "source": "zhang_fryirs_2024_esp",
    },
    "pool": {
        "cn": ["深潭", "水潭", "渊"],
        "aliases": ["pool", "deep-water", "scour-hole", "plunge-pool"],
        "description": "Deep, slow-moving water zone often at meander outer bends",
        "source": "zhang_fryirs_2024_esp",
    },
    "floodplain": {
        "cn": ["河漫滩", "洪泛平原", "泛滥平原"],
        "aliases": ["floodplain", "alluvial-plain", "overbank-zone"],
        "description": "Flat area adjacent to channel, inundated during high flow events",
        "source": "zhang_fryirs_2024_esp",
    },
    "terrace": {
        "cn": ["阶地", "河岸阶地"],
        "aliases": ["terrace", "river-terrace", "abandoned-floodplain"],
        "description": "Former floodplain elevated above current channel level",
        "source": "zhang_fryirs_2024_esp",
    },
    "levee": {
        "cn": ["天然堤", "自然堤"],
        "aliases": ["levee", "natural-levee", "bank-ridge"],
        "description": "Raised ridge along channel banks formed by overbank deposition",
        "source": "zhang_fryirs_2024_esp",
    },
    "backswamp": {
        "cn": ["后沼", "漫滩沼泽"],
        "aliases": ["backswamp", "floodbasin", "backwater-swamp"],
        "description": "Low-lying floodplain area behind levees with poor drainage",
        "source": "zhang_fryirs_2024_esp",
    },
    "crevasse_splay": {
        "cn": ["决口扇", "破口扇"],
        "aliases": ["crevasse-splay", "breach-fan", "overbank-splay"],
        "description": "Fan-shaped deposit where flow breaks through a levee",
        "source": "zhang_fryirs_2024_esp",
    },
}


PHYSICAL_PROCESSES: dict[str, dict] = {
    "tectonic_uplift": {
        "cn": ["构造抬升", "地壳隆起", "造山运动"],
        "aliases": ["tectonic-uplift", "orogeny", "crustal-deformation", "mountain-building"],
        "description": "Vertical displacement of Earth's crust creating positive topography",
    },
    "stream_power_erosion": {
        "cn": ["河流动力侵蚀", "流水侵蚀", "水力侵蚀"],
        "aliases": ["stream-power-erosion", "fluvial-incision", "river-erosion"],
        "description": "Erosion rate proportional to drainage area and slope - E = K * A^m * S^n",
        "source": "schott_2023_large_scale_erosion",
    },
    "thermal_erosion": {
        "cn": ["热力侵蚀", "寒冻风化", "冻融侵蚀"],
        "aliases": ["thermal-erosion", "frost-weathering", "freeze-thaw"],
        "description": "Physical weathering from repeated freezing and thawing cycles",
    },
    "sediment_deposition": {
        "cn": ["沉积作用", "泥沙淤积", "堆积"],
        "aliases": ["deposition", "sedimentation", "aggradation", "accumulation"],
        "description": "Material dropped when transport capacity decreases below sediment load",
    },
    "sediment_transport": {
        "cn": ["泥沙输运", "沉积物搬运"],
        "aliases": ["sediment-transport", "bedload", "suspended-load"],
        "description": "Movement of solid particles by water, wind, or ice",
    },
    "mass_wasting": {
        "cn": ["块体运动", "滑坡", "崩塌"],
        "aliases": ["mass-wasting", "landslide", "slope-failure", "debris-flow"],
        "description": "Downslope movement of rock, soil, and debris under gravity",
    },
    "karst_dissolution": {
        "cn": ["喀斯特溶蚀", "岩溶作用"],
        "aliases": ["karst", "dissolution", "chemical-weathering", "carbonate-erosion"],
        "description": "Dissolution of soluble rocks (limestone, dolomite) creating caves and sinkholes",
    },
    "aeolian_processes": {
        "cn": ["风成过程", "风蚀", "风积"],
        "aliases": ["aeolian", "wind-transport", "dune-formation", "deflation"],
        "description": "Wind-driven erosion, transport, and deposition of sediments",
    },
    "glacial_erosion": {
        "cn": ["冰川侵蚀", "冰蚀"],
        "aliases": ["glacial-erosion", "ice-scouring", "plucking", "abrasion"],
        "description": "Erosion caused by moving ice masses scouring the bedrock",
        "source": "lipka_2025_geomorphons_gcl",
    },
    "coastal_processes": {
        "cn": ["海岸过程", "海蚀", "海积"],
        "aliases": ["coastal-processes", "wave-erosion", "longshore-drift", "coastal-deposition"],
        "description": "Wave and tidal actions shaping shorelines",
        "source": "abril_2024_coastlines_sle",
    },
    "periglacial": {
        "cn": ["冰缘过程", "冻土地貌"],
        "aliases": ["periglacial", "frost-action", "solifluction", "patterned-ground"],
        "description": "Geomorphic processes in cold, non-glacial environments",
    },
}


CLIMATE_ZONES: dict[str, dict] = {
    "tropical_rainforest": {
        "cn": ["热带雨林", "热带多雨"],
        "aliases": ["tropical", "rainforest", "equatorial", "af-climate"],
    },
    "tropical_monsoon": {
        "cn": ["热带季风", "季风雨林"],
        "aliases": ["monsoon", "tropical-wet-dry", "am-climate"],
    },
    "tropical_savanna": {
        "cn": ["热带草原", "热带稀树草原"],
        "aliases": ["savanna", "tropical-grassland", "aw-climate"],
    },
    "arid_desert": {
        "cn": ["干旱沙漠", "荒漠"],
        "aliases": ["arid", "desert", "hot-desert", "bwh-climate"],
    },
    "semi_arid_steppe": {
        "cn": ["半干旱草原", "干草原"],
        "aliases": ["semi-arid", "steppe", "dry-grassland", "bsh-climate"],
    },
    "mediterranean": {
        "cn": ["地中海气候", "亚热带夏干"],
        "aliases": ["mediterranean", "dry-summer", "cs-climate"],
    },
    "humid_subtropical": {
        "cn": ["亚热带湿润", "亚热带季风湿润"],
        "aliases": ["subtropical", "humid-warm", "cfa-climate"],
    },
    "temperate_oceanic": {
        "cn": ["温带海洋", "海洋性温带"],
        "aliases": ["temperate", "oceanic", "mild", "cfb-climate"],
    },
    "temperate_continental": {
        "cn": ["温带大陆", "大陆性温带"],
        "aliases": ["continental", "seasonal", "df-climate"],
    },
    "boreal": {
        "cn": ["寒温带", "亚寒带", "副极地"],
        "aliases": ["boreal", "subarctic", "taiga", "dfc-climate"],
    },
    "tundra": {
        "cn": ["苔原", "冻原", "寒漠"],
        "aliases": ["tundra", "permafrost", "arctic-plain", "et-climate"],
    },
    "ice_cap": {
        "cn": ["冰盖", "冰原", "永久冰"],
        "aliases": ["ice-cap", "polar-ice", "glacial", "ef-climate"],
    },
    "highland": {
        "cn": ["高原气候", "山地气候", "高海拔"],
        "aliases": ["highland", "alpine", "mountain-climate", "h-climate"],
    },
}


BIOME_TYPES: dict[str, dict] = {
    "tropical_forest": {
        "cn": ["热带森林", "雨林"],
        "aliases": ["rainforest", "jungle", "tropical-forest"],
    },
    "temperate_forest": {
        "cn": ["温带森林", "落叶阔叶林"],
        "aliases": ["temperate-forest", "deciduous", "broadleaf"],
    },
    "coniferous_forest": {
        "cn": ["针叶林", "泰加林", "北方针叶林"],
        "aliases": ["coniferous", "evergreen", "pine-forest", "taiga"],
    },
    "grassland": {
        "cn": ["草原", "草地"],
        "aliases": ["grassland", "prairie", "steppe", "pampa"],
    },
    "desert_scrub": {
        "cn": ["荒漠灌丛", "旱生灌丛"],
        "aliases": ["desert-scrub", "xeric-shrubland", "dryland"],
    },
    "tundra_vegetation": {
        "cn": ["苔原植被", "冻原植被"],
        "aliases": ["tundra-vegetation", "arctic-moss", "lichen"],
    },
    "wetland": {
        "cn": ["湿地", "沼泽", "泥炭地"],
        "aliases": ["wetland", "marsh", "swamp", "bog", "fen"],
    },
    "mangrove": {
        "cn": ["红树林", "海岸红树林"],
        "aliases": ["mangrove", "coastal-swamp", "tidal-forest"],
    },
    "mediterranean_scrub": {
        "cn": ["地中海灌丛", "硬叶灌丛"],
        "aliases": ["mediterranean-scrub", "chaparral", "maquis", "sclerophyll"],
    },
    "alpine_meadow": {
        "cn": ["高山草甸", "高山草原"],
        "aliases": ["alpine-meadow", "mountain-grassland", "fellfield"],
    },
}


TOPOGRAPHIC_FEATURES: dict[str, dict] = {
    "mountain_range": {
        "cn": ["山脉", "山系", "造山带"],
        "aliases": ["mountain-range", "orogen", "cordillera"],
    },
    "plateau": {
        "cn": ["高原", "台地"],
        "aliases": ["plateau", "tableland", "high-plain"],
    },
    "basin": {
        "cn": ["盆地", "洼地"],
        "aliases": ["basin", "depression", "inland-basin"],
    },
    "canyon": {
        "cn": ["峡谷", "河谷", "沟壑"],
        "aliases": ["canyon", "gorge", "ravine", "deep-valley"],
    },
    "plain": {
        "cn": ["平原", "低地"],
        "aliases": ["plain", "flatland", "lowland"],
    },
    "peninsula": {
        "cn": ["半岛"],
        "aliases": ["peninsula", "cape", "promontory", "headland"],
    },
    "archipelago": {
        "cn": ["群岛", "岛链"],
        "aliases": ["archipelago", "island-chain", "island-arc"],
    },
    "isthmus": {
        "cn": ["地峡", "陆桥"],
        "aliases": ["isthmus", "land-bridge", "neck"],
    },
    "strait": {
        "cn": ["海峡"],
        "aliases": ["strait", "channel", "narrows", "passage"],
    },
    "fjord": {
        "cn": ["峡湾", "冰蚀谷湾"],
        "aliases": ["fjord", "glacial-valley", "drowned-valley"],
    },
    "delta": {
        "cn": ["三角洲", "河口三角洲"],
        "aliases": ["delta", "river-mouth", "alluvial-fan"],
    },
    "estuary": {
        "cn": ["河口", "河口湾", "三角湾"],
        "aliases": ["estuary", "river-mouth", "tidal-mouth"],
    },
    "atoll": {
        "cn": ["环礁", "珊瑚环礁"],
        "aliases": ["atoll", "coral-ring", "lagoon-island"],
    },
    "lagoon": {
        "cn": ["泻湖", "潟湖", "环礁湖"],
        "aliases": ["lagoon", "sheltered-water", "atoll-lagoon"],
    },
    "cliff": {
        "cn": ["悬崖", "海崖", "陡崖"],
        "aliases": ["cliff", "escarpment", "bluff", "steep-face"],
    },
    "valley_floor": {
        "cn": ["谷底", "河谷底"],
        "aliases": ["valley-floor", "valley-bottom", "channel-floor"],
    },
    "interfluve": {
        "cn": ["河间地", "分水高地"],
        "aliases": ["interfluve", "divide-area", "between-streams"],
    },
    "continental_shelf": {
        "cn": ["大陆架", "陆棚"],
        "aliases": ["continental-shelf", "shelf-sea", "neritic-zone"],
    },
    "abyssal_plain": {
        "cn": ["深海平原", "深海底平原"],
        "aliases": ["abyssal-plain", "deep-ocean-floor", "oceanic-basin"],
    },
    "rift_valley": {
        "cn": ["裂谷", "地堑"],
        "aliases": ["rift-valley", "graben", "extensional-basin"],
    },
    "volcanic_arc": {
        "cn": ["火山弧", "岛弧"],
        "aliases": ["volcanic-arc", "island-arc", "arc-system"],
    },
    "hotspot_chain": {
        "cn": ["热点火山链", "火山链"],
        "aliases": ["hotspot-chain", "seamount-chain", "volcanic-trail"],
    },
    "impact_crater": {
        "cn": ["陨石坑", "撞击坑"],
        "aliases": ["impact-crater", "meteor-crater", "astrobleme"],
    },
    "caldera": {
        "cn": ["破火山口", "火山洼地"],
        "aliases": ["caldera", "volcanic-collapse", "crater-lake"],
    },
}


SPATIAL_RELATIONS: dict[str, dict] = {
    "disjoint": {
        "cn": ["不相交", "分离", "隔离"],
        "aliases": ["disjoint", "separated", "isolated"],
    },
    "touches": {
        "cn": ["相接", "邻接", "毗连", "接壤"],
        "aliases": ["touches", "adjacent", "bordering", "neighbor"],
    },
    "crosses": {
        "cn": ["相交", "横穿", "贯穿"],
        "aliases": ["crosses", "intersects", "cuts-across", "traverses"],
    },
    "overlaps": {
        "cn": ["重叠", "叠置", "部分包含"],
        "aliases": ["overlaps", "partially-covers", "overlaid"],
    },
    "within": {
        "cn": ["内部", "包含", "围合"],
        "aliases": ["within", "inside", "contained", "enclosed"],
    },
    "contains": {
        "cn": ["包含", "包围", "包括"],
        "aliases": ["contains", "encloses", "surrounds", "encircles"],
    },
    "along": {
        "cn": ["沿着", "顺沿", "依傍"],
        "aliases": ["along", "following", "parallel-to", "flanking"],
    },
    "between": {
        "cn": ["之间", "中间", "间隔"],
        "aliases": ["between", "interposed", "sandwiched", "amid"],
    },
}


TERRAIN_GENERATION_METHODS: dict[str, dict] = {
    "procedural_noise": {
        "cn": ["过程噪声", "Perlin噪声", "Simplex噪声"],
        "aliases": ["procedural-noise", "perlin-noise", "simplex-noise", "fbm"],
        "description": "Mathematical noise functions for creating fractal-like surfaces",
        "literature": ["thorimbert_2018_polynomial", "goslin_terrain_diffusion"],
    },
    "physics_based_erosion": {
        "cn": ["物理侵蚀", "水力侵蚀", "热力侵蚀"],
        "aliases": ["physics-erosion", "stream-power", "thermal-erosion", "fluvial"],
        "description": "Simulating physical processes that shape landscapes",
        "literature": ["schott_2023_large_scale", "schott_2024_amplification"],
    },
    "example_based": {
        "cn": ["基于样例", "地形合成", "纹理合成"],
        "aliases": ["example-based", "texture-synthesis", "patch-based"],
        "description": "Learning terrain patterns from real DEMs and reproducing them",
        "literature": ["scott_2022_evaluating_realism", "jain_2024_infinite_lod"],
    },
    "diffusion_model": {
        "cn": ["扩散模型", "生成扩散", "去噪扩散"],
        "aliases": ["diffusion", "latent-diffusion", "denoising-diffusion", "ddpm"],
        "description": "Deep learning models that learn to reverse a noise process",
        "literature": ["mesa_borne_pons", "terra_fusion_higo", "geodiffussr_inui"],
    },
    "gan_based": {
        "cn": ["GAN生成", "对抗网络", "生成对抗"],
        "aliases": ["gan", "generative-adversarial", "dcgan"],
        "description": "Generator-discriminator architecture for terrain heightmap synthesis",
    },
    "sketch_based": {
        "cn": ["草图引导", "手绘地形", "草图绘制"],
        "aliases": ["sketch-based", "user-guided", "drawn-terrain"],
        "description": "User provides sketches (ridges, valleys, rivers) to guide generation",
        "literature": ["tdn_hu_2024", "terra_fusion_higo_2025"],
    },
    "llm_agent": {
        "cn": ["大语言模型", "LLM代理", "语言驱动"],
        "aliases": ["llm-agent", "language-driven", "text-to-terrain", "nl2terrain"],
        "description": "Using LLMs to interpret text and configure terrain parameters",
        "literature": ["zhang_2024_words_to_worlds", "lim_2024_zero_shot_3d_map"],
    },
    "hydrology_based": {
        "cn": ["水文引导", "河流网络", "流域地形"],
        "aliases": ["hydrology-based", "river-network", "watershed-terrain"],
        "description": "Using river networks and drainage patterns to define terrain",
        "literature": ["geneveaux_2013_hydrology"],
    },
    "landscape_evolution": {
        "cn": ["景观演化", "地形演化", "地貌模拟"],
        "aliases": ["lem", "landscape-evolution-model", "long-term-simulation"],
        "description": "Simulating long-term landscape development under climate and tectonics",
        "literature": ["salles_2016_badlands", "wallin_2024_modular_lem"],
    },
    "tile_based_infinite": {
        "cn": ["无限瓦片", "瓦片拼接", "无限地形"],
        "aliases": ["infinite-tiles", "seamless-world", "procedural-world"],
        "description": "Generating seamless infinite landscapes through tiling",
        "literature": ["sharma_2024_earthgen"],
    },
}


def get_all_geo_terms() -> list[str]:
    terms: list[str] = []
    for cat in [
        GEOMORPHON_TYPES, TERRAIN_METRICS, GEOMORPHIC_UNITS_FLUVIAL,
        PHYSICAL_PROCESSES, CLIMATE_ZONES, BIOME_TYPES, TOPOGRAPHIC_FEATURES,
        SPATIAL_RELATIONS, TERRAIN_GENERATION_METHODS,
    ]:
        for info in cat.values():
            terms.extend(info.get("cn", []))
            terms.extend(info.get("aliases", []))
    return list(set(terms))


def get_geomorphon_compound_words() -> list[str]:
    words: list[str] = []
    for info in GEOMORPHON_TYPES.values():
        for cn_term in info.get("cn", []):
            words.append(cn_term)
    words.extend([
        "山脊线", "分水岭", "坡度陡", "坡度缓", "坡向朝", "朝阳坡",
        "阴坡", "阳坡", "背风坡", "迎风坡", "高山坡", "缓坡",
        "陡坡", "坡面", "坡积", "坡麓", "山麓带", "山前",
        "水岭", "谷线", "脊线", "坡地", "丘陵地", "阶地",
        "洪积扇", "冲积扇", "冲积平原", "河流阶地", "海蚀阶地",
        "溶蚀洼地", "喀斯特谷", "冰蚀谷", "U形谷", "V形谷",
        "断层崖", "断层谷", "褶皱山", "断块山",
    ])
    return words


def get_synthesis_compound_words() -> list[str]:
    return [
        "物理侵蚀", "水力侵蚀", "热力侵蚀", "化学风化", "物理风化",
        "构造抬升", "造山运动", "板块碰撞", "俯冲带", "地壳隆升",
        "扩散模型", "隐空间", "去噪过程", "条件生成", "无条件生成",
        "高程图", "纹理图", "表面纹理", "地形几何", "地形结构",
        "多层次", "跨尺度", "自相似", "分形", "分维",
    ]
