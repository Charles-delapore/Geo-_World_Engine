from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import label, distance_transform_edt, sum as ndimage_sum

logger = logging.getLogger(__name__)

_HAS_SHAPELY = False
try:
    from shapely.geometry import shape, MultiPolygon, Point
    from shapely.ops import unary_union
    import geopandas as gpd
    _HAS_SHAPELY = True
except ImportError:
    pass

_HAS_SKIMAGE = False
try:
    from skimage.morphology import skeletonize as sk_skeletonize
    from skimage.measure import regionprops as sk_regionprops, label as sk_label
    _HAS_SKIMAGE = True
except ImportError:
    pass


def count_components(mask: np.ndarray, min_cells: int = 1) -> int:
    binary = mask.astype(np.int8) if mask.dtype != np.int8 else mask
    _, num = label(binary)
    if min_cells <= 1:
        return int(num)
    labels_arr, _ = label(binary)
    kept = 0
    for cid in range(1, num + 1):
        if int(np.sum(labels_arr == cid)) >= min_cells:
            kept += 1
    return kept


def largest_component_ratio(mask: np.ndarray) -> float:
    labels_arr, num = label(mask.astype(np.int8))
    if num == 0:
        return 0.0
    sizes = [int(np.sum(labels_arr == cid)) for cid in range(1, num + 1)]
    total = int(np.sum(mask))
    if total == 0:
        return 0.0
    return float(max(sizes)) / float(total)


def bbox_aspect_ratio(mask: np.ndarray) -> float:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return 1.0
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    height = float(rmax - rmin + 1)
    width = float(cmax - cmin + 1)
    if width < 1e-6:
        return 1.0
    return height / width


def principal_axis_angle(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return 0.0
    coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]
    angle = np.degrees(np.arctan2(principal[1], principal[0]))
    return float(angle)


def solidity(mask: np.ndarray) -> float:
    from scipy.ndimage import binary_fill_holes
    filled = binary_fill_holes(mask.astype(bool))
    filled_area = float(np.sum(filled))
    if filled_area < 1e-6:
        return 1.0
    return float(np.sum(mask)) / filled_area


def compactness(mask: np.ndarray) -> float:
    area = float(np.sum(mask))
    if area < 1e-6:
        return 1.0
    perimeter = _perimeter_length(mask)
    if perimeter < 1e-6:
        return 1.0
    return float(4.0 * np.pi * area / (perimeter ** 2))


def coastline_length(mask: np.ndarray) -> float:
    return _perimeter_length(mask)


def coast_roughness(mask: np.ndarray) -> float:
    area = float(np.sum(mask))
    if area < 1e-6:
        return 0.0
    perimeter = _perimeter_length(mask)
    if perimeter < 1e-6:
        return 0.0
    circle_perimeter = 2.0 * np.sqrt(np.pi * area)
    return float(perimeter / max(circle_perimeter, 1e-6))


def cross_cut_score(water_mask: np.ndarray) -> float:
    h, w = water_mask.shape
    if h < 4 or w < 4:
        return 0.0
    center_h_start = int(h * 0.35)
    center_h_end = int(h * 0.65)
    center_w_start = int(w * 0.35)
    center_w_end = int(w * 0.65)
    vertical_band = water_mask[:, center_w_start:center_w_end]
    horizontal_band = water_mask[center_h_start:center_h_end, :]
    vertical_ratio = float(np.mean(vertical_band))
    horizontal_ratio = float(np.mean(horizontal_band))
    return float(min(vertical_ratio, horizontal_ratio) * (vertical_ratio + horizontal_ratio))


def enclosure_score(water_mask: np.ndarray, land_mask: np.ndarray) -> float:
    if not np.any(water_mask) or not np.any(land_mask):
        return 0.0
    dist_from_water = distance_transform_edt(~water_mask)
    dist_from_land = distance_transform_edt(~land_mask)
    boundary = (dist_from_water > 0) & (dist_from_water <= 3) & land_mask
    if not np.any(boundary):
        return 0.0
    surrounding_land = float(np.sum(land_mask & (dist_from_water <= 8)))
    surrounding_total = float(np.sum(dist_from_water <= 8))
    if surrounding_total < 1e-6:
        return 0.0
    return float(surrounding_land / surrounding_total)


def component_labels(mask: np.ndarray, min_cells: int = 1) -> np.ndarray:
    labels_arr, num = label(mask.astype(np.int8))
    if min_cells <= 1:
        return labels_arr
    for cid in range(1, num + 1):
        if int(np.sum(labels_arr == cid)) < min_cells:
            labels_arr[labels_arr == cid] = 0
    return labels_arr


def mask_to_geojson(mask: np.ndarray) -> list[dict]:
    if not _HAS_SHAPELY:
        return []
    import json
    from rasterio.features import shapes as rio_shapes

    _HAS_RASTERIO = False
    try:
        from rasterio.features import shapes as _rio_shapes
        _HAS_RASTERIO = True
    except ImportError:
        pass

    if not _HAS_RASTERIO:
        return []

    results = []
    for geom, value in _rio_shapes(mask.astype(np.uint8), mask=mask > 0):
        if value > 0:
            results.append(geom)
    return results


def shapely_skeleton_length(mask: np.ndarray) -> float:
    if not _HAS_SKIMAGE:
        return skeleton_length_numpy(mask)
    binary = mask.astype(bool)
    skel = sk_skeletonize(binary)
    return float(np.sum(skel))


def skeleton_length_numpy(mask: np.ndarray) -> float:
    from scipy.ndimage import binary_erosion, generate_binary_structure
    struct = generate_binary_structure(2, 2)
    binary = mask.astype(bool)
    skel = binary.copy()
    prev = np.zeros_like(skel)
    while np.any(skel) and not np.array_equal(skel, prev):
        prev = skel.copy()
        eroded = binary_erosion(skel, structure=struct)
        skel = skel & ~eroded
        skel = binary_erosion(skel, structure=generate_binary_structure(2, 1))
        if not np.any(skel):
            break
    return float(np.sum(prev))


def skeleton_length(mask: np.ndarray) -> float:
    if _HAS_SKIMAGE:
        return shapely_skeleton_length(mask)
    return skeleton_length_numpy(mask)


def shapely_convex_hull_area(mask: np.ndarray) -> float:
    if not _HAS_SHAPELY:
        return 0.0
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return 0.0
    try:
        from shapely.geometry import MultiPoint
        points = MultiPoint(list(zip(xs.astype(float), ys.astype(float))))
        hull = points.convex_hull
        return float(hull.area)
    except Exception:
        return 0.0


def shapely_component_properties(mask: np.ndarray, min_cells: int = 20) -> list[dict]:
    if not _HAS_SKIMAGE:
        return []
    labeled = sk_label(mask > 0)
    props = sk_regionprops(labeled)
    results = []
    for prop in props:
        if prop.area < min_cells:
            continue
        results.append({
            "area": int(prop.area),
            "bbox": prop.bbox,
            "centroid": prop.centroid,
            "orientation": float(prop.orientation),
            "axis_major_length": float(prop.axis_major_length if hasattr(prop, 'axis_major_length') else prop.major_axis_length),
            "axis_minor_length": float(prop.axis_minor_length if hasattr(prop, 'axis_minor_length') else prop.minor_axis_length),
            "eccentricity": float(prop.eccentricity),
            "solidity": float(prop.solidity),
        })
    return results


def compute_metric_report(
    elevation: np.ndarray,
    topology_intent: dict | None = None,
    include_terrain_analysis: bool = True,
) -> dict:
    land_mask = elevation > 0.0
    water_mask = elevation <= 0.0
    report: dict = {
        "land_components": count_components(land_mask, min_cells=20),
        "water_components": count_components(water_mask, min_cells=20),
        "largest_land_ratio": largest_component_ratio(land_mask),
        "land_area_ratio": float(np.mean(land_mask)),
        "bbox_aspect_ratio": bbox_aspect_ratio(land_mask),
        "principal_axis_angle": principal_axis_angle(land_mask),
        "solidity": solidity(land_mask),
        "compactness": compactness(land_mask),
        "coast_roughness": coast_roughness(land_mask),
        "cross_cut_score": cross_cut_score(water_mask),
        "enclosure_score": enclosure_score(water_mask, land_mask),
        "skeleton_length": skeleton_length(land_mask),
    }

    if _HAS_SKIMAGE:
        component_props = shapely_component_properties(land_mask, min_cells=20)
        if component_props:
            dominant = max(component_props, key=lambda p: p["area"])
            report["dominant_component"] = dominant
            report["component_details"] = component_props

    if _HAS_SHAPELY:
        hull_area = shapely_convex_hull_area(land_mask)
        mask_area = float(np.sum(land_mask))
        if hull_area > 0:
            report["convex_hull_solidity"] = mask_area / hull_area

    if include_terrain_analysis:
        try:
            from app.core.terrain_analysis import (
                compute_geomorphons,
                compute_ptrm,
                compute_coastline_fractal_dimension,
                compute_coastline_roughness_exponent,
                compute_multi_scale_tpi,
                compute_gradient_magnitude,
            )
            geom = compute_geomorphons(elevation, lookup_distance=3, flat_threshold=0.5)
            report["ptrm_score"] = compute_ptrm(geom)
            if np.any(land_mask):
                report["coastline_fractal_dim"] = compute_coastline_fractal_dimension(land_mask)
                report["coastline_roughness_exponent"] = compute_coastline_roughness_exponent(land_mask)
            mstpi = compute_multi_scale_tpi(elevation, scales=[3, 9, 27])
            report["tpi_variance_fine"] = float(np.var(mstpi["tpi_3"]))
            report["tpi_variance_coarse"] = float(np.var(mstpi["tpi_27"]))
            grad_mag = compute_gradient_magnitude(elevation)
            report["mean_gradient"] = float(np.mean(grad_mag[land_mask])) if np.any(land_mask) else 0.0
            report["max_gradient"] = float(np.max(grad_mag[land_mask])) if np.any(land_mask) else 0.0
        except Exception as exc:
            logger.warning("terrain_analysis metrics failed: %s", exc)

    if topology_intent:
        kind = str(topology_intent.get("kind", "")).lower()
        report["topology_intent_kind"] = kind
        if kind == "single_island":
            report["single_island_pass"] = report["land_components"] == 1
        elif kind == "two_continents_with_rift_sea":
            report["two_continents_pass"] = report["land_components"] == 2
        elif kind == "central_enclosed_inland_sea":
            report["inland_sea_enclosure_pass"] = report["enclosure_score"] > 0.5
    return report


def _perimeter_length(mask: np.ndarray) -> float:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    edges = 0
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(padded, dy, axis=0), dx, axis=1)
        edges += int(np.sum((padded == 1) & (shifted == 0)))
    return float(edges)
