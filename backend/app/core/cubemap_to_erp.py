from __future__ import annotations

import numpy as np

from app.core.cubemap_terrain import NUM_FACES, _face_to_sphere


def _sphere_to_face(sx: np.ndarray, sy: np.ndarray, sz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    abs_x = np.abs(sx)
    abs_y = np.abs(sy)
    abs_z = np.abs(sz)

    face_id = np.zeros(sx.shape, dtype=np.int32)
    u = np.zeros(sx.shape, dtype=np.float32)
    v = np.zeros(sx.shape, dtype=np.float32)

    mask_px = (abs_x >= abs_y) & (abs_x >= abs_z) & (sx > 0)
    mask_nx = (abs_x >= abs_y) & (abs_x >= abs_z) & (sx <= 0)
    mask_py = (abs_y >= abs_x) & (abs_y >= abs_z) & (sy > 0)
    mask_ny = (abs_y >= abs_x) & (abs_y >= abs_z) & (sy <= 0)
    mask_pz = (abs_z >= abs_x) & (abs_z >= abs_y) & (sz > 0)
    mask_nz = (abs_z >= abs_x) & (abs_z >= abs_y) & (sz <= 0)

    face_id[mask_px] = 0
    u[mask_px] = -sz[mask_px] / np.maximum(abs_x[mask_px], 1e-12)
    v[mask_px] = sy[mask_px] / np.maximum(abs_x[mask_px], 1e-12)

    face_id[mask_nx] = 1
    u[mask_nx] = sz[mask_nx] / np.maximum(abs_x[mask_nx], 1e-12)
    v[mask_nx] = sy[mask_nx] / np.maximum(abs_x[mask_nx], 1e-12)

    face_id[mask_py] = 2
    u[mask_py] = sx[mask_py] / np.maximum(abs_y[mask_py], 1e-12)
    v[mask_py] = -sz[mask_py] / np.maximum(abs_y[mask_py], 1e-12)

    face_id[mask_ny] = 3
    u[mask_ny] = sx[mask_ny] / np.maximum(abs_y[mask_ny], 1e-12)
    v[mask_ny] = sz[mask_ny] / np.maximum(abs_y[mask_ny], 1e-12)

    face_id[mask_pz] = 4
    u[mask_pz] = sx[mask_pz] / np.maximum(abs_z[mask_pz], 1e-12)
    v[mask_pz] = sy[mask_pz] / np.maximum(abs_z[mask_pz], 1e-12)

    face_id[mask_nz] = 5
    u[mask_nz] = -sx[mask_nz] / np.maximum(abs_z[mask_nz], 1e-12)
    v[mask_nz] = sy[mask_nz] / np.maximum(abs_z[mask_nz], 1e-12)

    return face_id, u, v


def _bilinear_sample_face(
    face_data: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    h, w = face_data.shape[:2]
    fx = (u + 1.0) * 0.5 * (w - 1)
    fy = (v + 1.0) * 0.5 * (h - 1)

    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = (fx - x0).astype(np.float32)
    wy = (fy - y0).astype(np.float32)

    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    if face_data.ndim == 2:
        top = face_data[y0, x0] * (1.0 - wx) + face_data[y0, x1] * wx
        bottom = face_data[y1, x0] * (1.0 - wx) + face_data[y1, x1] * wx
        return (top * (1.0 - wy) + bottom * wy).astype(np.float32)
    else:
        top = face_data[y0, x0] * (1.0 - wx[..., None]) + face_data[y0, x1] * wx[..., None]
        bottom = face_data[y1, x0] * (1.0 - wx[..., None]) + face_data[y1, x1] * wx[..., None]
        return (top * (1.0 - wy[..., None]) + bottom * wy[..., None]).astype(np.float32)


def cubemap_to_erp(
    cube_faces: list[np.ndarray],
    erp_width: int = 2048,
    erp_height: int = 1024,
) -> np.ndarray:
    lon = np.linspace(0, 2 * np.pi, erp_width, endpoint=False, dtype=np.float64)
    lat = np.linspace(np.pi / 2, -np.pi / 2, erp_height, dtype=np.float64)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    sx = (np.cos(lat_grid) * np.cos(lon_grid)).astype(np.float32)
    sy = (np.cos(lat_grid) * np.sin(lon_grid)).astype(np.float32)
    sz = (np.sin(lat_grid)).astype(np.float32)

    face_id, u, v = _sphere_to_face(sx, sy, sz)

    result = np.zeros((erp_height, erp_width), dtype=np.float32)

    for fid in range(NUM_FACES):
        mask = face_id == fid
        if not np.any(mask):
            continue
        u_masked = u[mask]
        v_masked = v[mask]
        sampled = _bilinear_sample_face(cube_faces[fid], u_masked, v_masked)
        result[mask] = sampled

    return result


def cubemap_rgb_to_erp(
    cube_faces_rgb: list[np.ndarray],
    erp_width: int = 2048,
    erp_height: int = 1024,
) -> np.ndarray:
    lon = np.linspace(0, 2 * np.pi, erp_width, endpoint=False, dtype=np.float64)
    lat = np.linspace(np.pi / 2, -np.pi / 2, erp_height, dtype=np.float64)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    sx = (np.cos(lat_grid) * np.cos(lon_grid)).astype(np.float32)
    sy = (np.cos(lat_grid) * np.sin(lon_grid)).astype(np.float32)
    sz = (np.sin(lat_grid)).astype(np.float32)

    face_id, u, v = _sphere_to_face(sx, sy, sz)

    channels = cube_faces_rgb[0].shape[2] if cube_faces_rgb[0].ndim == 3 else 1
    result = np.zeros((erp_height, erp_width, channels), dtype=np.float32)

    for fid in range(NUM_FACES):
        mask = face_id == fid
        if not np.any(mask):
            continue
        u_masked = u[mask]
        v_masked = v[mask]
        sampled = _bilinear_sample_face(cube_faces_rgb[fid], u_masked, v_masked)
        if result.ndim == 3 and sampled.ndim == 1:
            result[mask] = sampled[..., None]
        else:
            result[mask] = sampled

    if channels == 1:
        return result[:, :, 0]
    return result


def erp_to_cubemap(
    erp_data: np.ndarray,
    face_size: int = 512,
) -> list[np.ndarray]:
    from app.core.cubemap_terrain import _face_to_sphere

    is_rgb = erp_data.ndim == 3
    cube_faces: list[np.ndarray] = []

    for fid in range(NUM_FACES):
        u_coords = np.linspace(-1.0, 1.0, face_size, dtype=np.float32)
        v_coords = np.linspace(-1.0, 1.0, face_size, dtype=np.float32)
        uu, vv = np.meshgrid(u_coords, v_coords)

        sx, sy, sz = _face_to_sphere(fid, uu, vv)

        lon = np.arctan2(sy, sx) % (2 * np.pi)
        lat = np.arcsin(np.clip(sz, -1.0, 1.0))

        erp_h, erp_w = erp_data.shape[:2]
        col = (lon / (2 * np.pi) * erp_w).astype(np.float64) % erp_w
        row = ((np.pi / 2 - lat) / np.pi * erp_h).astype(np.float64)
        row = np.clip(row, 0, erp_h - 1)

        col0 = np.floor(col).astype(np.int32) % erp_w
        col1 = (col0 + 1) % erp_w
        row0 = np.clip(np.floor(row).astype(np.int32), 0, erp_h - 1)
        row1 = np.clip(row0 + 1, 0, erp_h - 1)

        fc = (col - np.floor(col)).astype(np.float32)
        fr = (row - np.floor(row)).astype(np.float32)

        if is_rgb:
            v00 = erp_data[row0, col0]
            v01 = erp_data[row0, col1]
            v10 = erp_data[row1, col0]
            v11 = erp_data[row1, col1]
            face = (
                v00 * (1 - fc[..., None]) * (1 - fr[..., None])
                + v01 * fc[..., None] * (1 - fr[..., None])
                + v10 * (1 - fc[..., None]) * fr[..., None]
                + v11 * fc[..., None] * fr[..., None]
            )
        else:
            v00 = erp_data[row0, col0]
            v01 = erp_data[row0, col1]
            v10 = erp_data[row1, col0]
            v11 = erp_data[row1, col1]
            face = (
                v00 * (1 - fc) * (1 - fr)
                + v01 * fc * (1 - fr)
                + v10 * (1 - fc) * fr
                + v11 * fc * fr
            )

        cube_faces.append(face.astype(np.float32))

    return cube_faces
