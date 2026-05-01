from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    uniform_filter,
    sobel,
    label,
    binary_dilation,
    generate_binary_structure,
)


def sphere_naturalize_coastline(
    elevation: np.ndarray,
    boundary_irregularity: float = 0.5,
    coast_complexity: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    h, w = elevation.shape
    land_mask = elevation > 0.0
    water_mask = elevation <= 0.0
    if not np.any(land_mask) or not np.any(water_mask):
        return elevation

    dist_land = distance_transform_edt(land_mask)
    dist_water = distance_transform_edt(water_mask)
    boundary_width = max(3, int(6 + coast_complexity * 8))
    boundary_band = (dist_land < boundary_width) & (dist_water < boundary_width)

    if not np.any(boundary_band):
        return elevation

    from app.core.sphere_terrain import erp_grid_to_sphere, sphere_fbm

    sx, sy, sz = erp_grid_to_sphere(w, h)

    low_freq = sphere_fbm(sx, sy, sz, scale=120.0, octaves=3, persistence=0.56, lacunarity=2.0, seed=seed + 1201)
    mid_freq = sphere_fbm(sx, sy, sz, scale=52.0, octaves=4, persistence=0.55, lacunarity=2.1, seed=seed + 1207)
    high_freq = sphere_fbm(sx, sy, sz, scale=24.0, octaves=3, persistence=0.5, lacunarity=2.2, seed=seed + 1213)

    perturbation = (
        (low_freq * 2.0 - 1.0) * 0.12 * boundary_irregularity
        + (mid_freq * 2.0 - 1.0) * 0.08 * boundary_irregularity * coast_complexity
        + (high_freq * 2.0 - 1.0) * 0.03 * coast_complexity
    )

    falloff = np.clip(1.0 - np.minimum(dist_land, dist_water) / max(boundary_width, 1), 0.0, 1.0)
    shaped = elevation.astype(np.float32).copy()
    shaped[boundary_band] += (perturbation * falloff)[boundary_band].astype(np.float32)

    land_core = dist_land >= boundary_width
    water_core = dist_water >= boundary_width
    shaped[land_core] = np.maximum(shaped[land_core], elevation[land_core])
    shaped[water_core] = np.minimum(shaped[water_core], elevation[water_core])

    return gaussian_filter(shaped, sigma=0.8).astype(np.float32)


def sphere_enforce_fractal_dimension(
    elevation: np.ndarray,
    target_fd: float = 1.15,
    tolerance: float = 0.10,
    max_iterations: int = 2,
    coast_complexity: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    from app.core.coast_naturalizer import _measure_coast_fractal_dimension

    result = elevation.copy()
    for _ in range(max_iterations):
        try:
            current_fd = _measure_coast_fractal_dimension(result)
        except Exception:
            break

        if abs(current_fd - target_fd) < tolerance:
            break

        if current_fd < target_fd:
            result = sphere_naturalize_coastline(
                result,
                boundary_irregularity=0.3 + coast_complexity * 0.4,
                coast_complexity=coast_complexity * 1.2,
                seed=seed + 1300,
            )
        else:
            result = gaussian_filter(result, sigma=1.2).astype(np.float32)

    return result


def sphere_curvature_guided_erosion(
    elevation: np.ndarray,
    iterations: int = 2,
    erosion_rate: float = 0.012,
    ridge_protection: float = 0.75,
) -> np.ndarray:
    result = elevation.astype(np.float32).copy()
    h, w = result.shape

    for _ in range(iterations):
        gy, gx = np.gradient(result)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2 + 1e-10)

        gyy, gyx = np.gradient(gy)
        gxy, gxx = np.gradient(gx)
        curvature = gxx + gyy

        is_valley = curvature < 0
        is_ridge = curvature > 0

        erosion = np.zeros_like(result)
        erosion[is_valley] = erosion_rate * (1.0 - ridge_protection) * grad_mag[is_valley]
        erosion[is_ridge] = -erosion_rate * 0.1 * grad_mag[is_ridge]

        land = result > 0
        result[land] -= erosion[land]

    return np.clip(result, -1.0, 1.0).astype(np.float32)


def sphere_multi_scale_erosion(
    elevation: np.ndarray,
    scales: list[int] | None = None,
    base_rate: float = 0.006,
) -> np.ndarray:
    if scales is None:
        scales = [1, 3]

    result = elevation.astype(np.float32).copy()

    for scale in scales:
        smoothed = uniform_filter(result, size=scale * 2 + 1)
        diff = result - smoothed
        erosion = base_rate * scale * np.clip(diff, 0, None)
        land = result > 0
        result[land] -= erosion[land]

    return np.clip(result, -1.0, 1.0).astype(np.float32)


def sphere_compute_multi_scale_tpi(
    elevation: np.ndarray,
    scales: list[int] | None = None,
    curvature_weight: float = 0.3,
) -> dict[str, np.ndarray]:
    if scales is None:
        scales = [3, 9]

    result = {}
    for s in scales:
        mean_elev = uniform_filter(elevation, size=s * 2 + 1)
        tpi = elevation - mean_elev

        if curvature_weight > 0:
            gy, gx = np.gradient(elevation)
            gxx = np.gradient(gx, axis=1)
            gyy = np.gradient(gy, axis=0)
            gxy = np.gradient(gx, axis=0)

            denom = (gx ** 2 + gy ** 2) ** 1.5 + 1e-10
            profile_curv = -(gxx * gx ** 2 + 2 * gxy * gx * gy + gyy * gy ** 2) / denom

            plan_denom = (gx ** 2 + gy ** 2) ** 1.5 + 1e-10
            plan_curv = -(gxx * gy ** 2 - 2 * gxy * gx * gy + gyy * gx ** 2) / plan_denom

            curvature_enhancement = 1.0 + curvature_weight * (np.abs(profile_curv) + np.abs(plan_curv))
            curvature_enhancement = np.clip(curvature_enhancement, 0.5, 3.0)
            tpi = tpi * curvature_enhancement

        result[f"tpi_{s}"] = tpi.astype(np.float32)

    return result


def sphere_tpi_feedback(
    elevation: np.ndarray,
    ruggedness: float = 0.55,
    scales: list[int] | None = None,
) -> np.ndarray:
    mstpi = sphere_compute_multi_scale_tpi(elevation, scales=scales or [3, 9])
    keys = list(mstpi.keys())
    tpi_fine = mstpi[keys[0]]
    tpi_coarse = mstpi[keys[1]]
    tpi_combined = tpi_fine * 0.6 + tpi_coarse * 0.4
    tpi_norm = tpi_combined / (np.std(tpi_combined) + 1e-10)

    ridge_mask = tpi_norm > 2.0
    valley_mask = tpi_norm < -2.0
    land = elevation > 0

    result = elevation.copy()
    result[ridge_mask & land] *= 1.0 + 0.01 * ruggedness
    result[valley_mask & land] *= 1.0 - 0.005 * ruggedness

    return np.clip(result, -1.0, 1.0).astype(np.float32)


def sphere_topology_guard(
    elevation: np.ndarray,
    topology_intent: dict | None = None,
    max_repair_strength: float = 0.3,
) -> tuple[np.ndarray, dict]:
    from app.core.geometry_metrics import count_components, component_labels

    result = elevation.copy()
    repairs = []
    land_mask = result > 0

    if not np.any(land_mask):
        return result, {"repairs": repairs, "passed": False}

    n_components = 0
    labels = component_labels(land_mask)
    n_components = int(labels.max())
    kind = str((topology_intent or {}).get("kind", "")).strip().lower()

    if kind == "single_island" and n_components > 1:
        component_sizes = np.bincount(labels.ravel())
        component_sizes[0] = 0
        largest = np.argmax(component_sizes)
        small_mask = (labels != 0) & (labels != largest)
        if np.any(small_mask):
            result[small_mask] = -0.1 * max_repair_strength
            repairs.append(f"removed_{np.sum(small_mask)}_small_island_pixels")

    elif kind == "two_continents_with_rift_sea" and n_components < 2:
        h, w = result.shape
        mid = w // 2
        left_land = np.sum(result[:, :mid] > 0)
        right_land = np.sum(result[:, mid:] > 0)
        if left_land > 0 and right_land > 0:
            rift_width = max(2, int(w * 0.04))
            rift_start = mid - rift_width // 2
            rift_end = rift_start + rift_width
            rift_mask = np.zeros_like(land_mask)
            rift_mask[:, rift_start:rift_end] = True
            rift_land = rift_mask & (result > 0)
            if np.any(rift_land):
                result[rift_land] = -0.05 * max_repair_strength
                repairs.append("created_rift_sea")

    return result, {"repairs": repairs, "passed": len(repairs) == 0}


def sphere_relax_periodic_longitude(
    values: np.ndarray,
    band_width: int | None = None,
    strength: float = 0.35,
) -> np.ndarray:
    source = values.astype(np.float32)
    width = source.shape[1]
    band = min(max(2, int(band_width or width // 96)), max(2, width // 8))
    if band * 2 >= width:
        return source

    padded = np.concatenate([source[:, -band:], source, source[:, :band]], axis=1)
    periodic = gaussian_filter(padded, sigma=(0.0, 1.0)).astype(np.float32)[:, band:-band]

    weights = np.zeros_like(source, dtype=np.float32)
    ramp = np.linspace(1.0, 0.0, band, dtype=np.float32)
    weights[:, :band] = ramp
    weights[:, -band:] = ramp[::-1]

    relaxed = source * (1.0 - weights * strength) + periodic * (weights * strength)
    seam = (relaxed[:, 0] + relaxed[:, -1]) * 0.5
    relaxed[:, 0] = seam
    relaxed[:, -1] = seam
    return relaxed.astype(np.float32)


def sphere_fractal_coastline(
    elevation: np.ndarray,
    max_depth: int = 4,
    base_amplitude: float = 1.5,
    amplitude_decay: float = 0.9,
    min_edge: float = 1.0,
    smooth_threshold: float = 0.25,
    roughness_contrast: float = 1.5,
    profile_harmonics: int = 4,
    seed: int = 0,
) -> np.ndarray:
    h, w = elevation.shape
    result = elevation.astype(np.float32).copy()

    land_mask = result > 0.0
    if not np.any(land_mask) or not np.any(~land_mask):
        return result

    rng = np.random.default_rng(seed)

    dist_land = distance_transform_edt(land_mask)
    dist_water = distance_transform_edt(~land_mask)
    boundary_band = (dist_land < 8) & (dist_water < 8)

    if not np.any(boundary_band):
        return result

    gy, gx = np.gradient(result)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2 + 1e-10)
    grad_mag_norm = grad_mag / (grad_mag.max() + 1e-10)

    roughness = np.zeros((h, w), dtype=np.float32)
    for k in range(1, profile_harmonics + 1):
        freq = 2.0 * np.pi * k / w
        phase = rng.uniform(0, 2 * np.pi)
        col_idx = np.arange(w, dtype=np.float32)
        row_idx = np.arange(h, dtype=np.float32)
        col_wave = np.cos(freq * col_idx + phase)
        row_wave = np.cos(freq * row_idx * 0.5 + phase * 0.7)
        roughness += np.outer(row_wave, col_wave) / k
    roughness = (roughness - roughness.min()) / (roughness.max() - roughness.min() + 1e-10)
    roughness = np.clip(roughness, 0, 1) ** roughness_contrast

    falloff = np.clip(1.0 - np.minimum(dist_land, dist_water) / 8.0, 0.0, 1.0)

    perturbation = np.zeros((h, w), dtype=np.float32)
    amplitude = base_amplitude

    for depth in range(max_depth):
        scale = max(1, int(w / (2 ** (depth + 3))))
        noise = rng.normal(0, amplitude, (h, w)).astype(np.float32)
        noise = gaussian_filter(noise, sigma=scale)

        smooth_mask = grad_mag_norm < smooth_threshold
        noise[smooth_mask] *= 0.1

        noise *= roughness
        perturbation += noise
        amplitude *= amplitude_decay

    perturbation *= falloff * 0.02
    result[boundary_band] += perturbation[boundary_band]

    land_core = dist_land >= 8
    water_core = dist_water >= 8
    result[land_core] = np.maximum(result[land_core], elevation[land_core])
    result[water_core] = np.minimum(result[water_core], elevation[water_core])

    return gaussian_filter(result, sigma=0.6).astype(np.float32)


def run_sphere_postprocess(
    elevation: np.ndarray,
    profile: dict | None = None,
    topology_intent: dict | None = None,
    seed: int = 0,
    uplift: np.ndarray | None = None,
    hardness: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    profile = profile or {}
    topology_intent = topology_intent or {}

    coast_complexity = float(profile.get("coast_complexity", 0.5))
    ruggedness = float(profile.get("ruggedness", 0.55))
    boundary_irregularity = float(topology_intent.get("boundary_irregularity", 0.5))

    result = elevation.copy()
    guard_result = {"repairs": [], "passed": True}

    result = sphere_naturalize_coastline(
        result,
        boundary_irregularity=boundary_irregularity,
        coast_complexity=coast_complexity,
        seed=seed,
    )

    if coast_complexity > 0.3:
        try:
            result = sphere_fractal_coastline(
                result,
                max_depth=4,
                base_amplitude=1.0 + coast_complexity,
                amplitude_decay=0.9,
                smooth_threshold=0.25,
                roughness_contrast=1.0 + coast_complexity,
                profile_harmonics=4,
                seed=seed + 1400,
            )
        except Exception:
            pass

    try:
        target_fd = 1.10 + coast_complexity * 0.12
        result = sphere_enforce_fractal_dimension(
            result,
            target_fd=target_fd,
            tolerance=0.10,
            max_iterations=1,
            coast_complexity=coast_complexity,
            seed=seed,
        )
    except Exception:
        pass

    result, guard_result = sphere_topology_guard(
        result,
        topology_intent=topology_intent,
        max_repair_strength=0.3,
    )

    if ruggedness > 0.3:
        try:
            from app.core.sphere_stream_power import (
                stream_power_erosion,
                thermal_erosion,
                sediment_deposition,
                multi_scale_breaching,
                diffusion_retargeting,
                _compute_d8_flow,
                _accumulate_drainage_area_fast,
            )

            erosion_strength = min(ruggedness * 0.6, 0.35)

            if uplift is None:
                try:
                    from app.core.uplift_field import generate_uplift_field
                    h, w = result.shape
                    uplift = generate_uplift_field(
                        erp_height=h,
                        erp_width=w,
                        seed=seed,
                        continent_mask=(result > 0).astype(np.float32),
                        ruggedness=ruggedness,
                    )
                except Exception:
                    uplift = None

            if hardness is None:
                try:
                    from app.core.uplift_field import generate_hardness_map
                    hardness = generate_hardness_map(
                        result,
                        ruggedness=ruggedness,
                        seed=seed,
                    )
                except Exception:
                    hardness = None

            result = stream_power_erosion(
                result,
                K=2e-4 * erosion_strength,
                m=0.5,
                n=1.0,
                dt=2000.0,
                iterations=max(5, int(20 * erosion_strength)),
                uplift=uplift,
                hardness=hardness,
                wrap_longitude=True,
            )

            result = thermal_erosion(
                result,
                talus_angle=0.7,
                erosion_rate=0.008 * erosion_strength,
                iterations=5,
                wrap_longitude=True,
            )

            if ruggedness > 0.5:
                flow_dir_input = result
                if True:
                    padded = np.concatenate([result[:, -1:], result, result[:, :1]], axis=1)
                    flow_dir_input = padded
                flow_dir = _compute_d8_flow(flow_dir_input)
                if True:
                    flow_dir = flow_dir[:, 1:-1]
                drainage_area = _accumulate_drainage_area_fast(result, flow_dir)

                result = sediment_deposition(
                    result,
                    flow_dir=flow_dir,
                    drainage_area=drainage_area,
                    deposition_rate=0.002 * erosion_strength,
                    wrap_longitude=True,
                )

            if ruggedness > 0.4:
                original_elev = elevation.copy() if elevation.shape == result.shape else None
                if original_elev is not None:
                    result = diffusion_retargeting(
                        result,
                        original_elev,
                        strength=0.3 * erosion_strength,
                        iterations=10,
                    )

            result = multi_scale_breaching(result, wrap_longitude=True)

        except Exception:
            result = sphere_curvature_guided_erosion(
                result,
                iterations=2,
                erosion_rate=0.012 * min(ruggedness * 0.6, 0.35),
                ridge_protection=0.75,
            )
            result = sphere_multi_scale_erosion(
                result,
                scales=[1, 3],
                base_rate=0.006 * min(ruggedness * 0.6, 0.35),
            )

    result = sphere_tpi_feedback(result, ruggedness=ruggedness)

    h, w = result.shape
    result = sphere_relax_periodic_longitude(result, band_width=max(4, w // 96), strength=0.35)

    return np.clip(result, -1.0, 1.0).astype(np.float32), guard_result
