from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_HAS_RASTERIO = False
try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    _HAS_RASTERIO = True
except ImportError:
    pass


def elevation_to_geotiff(
    elevation: np.ndarray,
    output_path: str | Path | None = None,
    crs: str = "EPSG:4326",
    bounds: tuple[float, float, float, float] = (-180.0, -85.0, 180.0, 85.0),
) -> bytes | Path:
    if not _HAS_RASTERIO:
        return _elevation_to_geotiff_fallback(elevation, output_path)

    h, w = elevation.shape
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], w, h)
    crs_obj = CRS.from_string(crs)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            str(output_path), "w",
            driver="GTiff",
            height=h, width=w,
            count=1,
            dtype=elevation.dtype,
            crs=crs_obj,
            transform=transform,
            compress="deflate",
        ) as dst:
            dst.write(elevation, 1)
        return output_path

    buf = io.BytesIO()
    with rasterio.open(
        buf, "w",
        driver="GTiff",
        height=h, width=w,
        count=1,
        dtype=elevation.dtype,
        crs=crs_obj,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(elevation, 1)
    return buf.getvalue()


def geotiff_to_elevation(input_path: str | Path | bytes) -> np.ndarray:
    if not _HAS_RASTERIO:
        raise ImportError("rasterio is required for GeoTIFF reading")

    if isinstance(input_path, bytes):
        buf = io.BytesIO(input_path)
        with rasterio.open(buf) as src:
            return src.read(1).astype(np.float32)

    with rasterio.open(str(input_path)) as src:
        return src.read(1).astype(np.float32)


def elevation_to_cog(
    elevation: np.ndarray,
    output_path: str | Path,
    crs: str = "EPSG:4326",
    bounds: tuple[float, float, float, float] = (-180.0, -85.0, 180.0, 85.0),
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not _HAS_RASTERIO:
        logger.warning("rasterio not available, falling back to NPZ storage")
        np.savez_compressed(str(output_path).replace(".tif", ".npz"), elevation=elevation)
        return Path(str(output_path).replace(".tif", ".npz"))

    h, w = elevation.shape
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], w, h)
    crs_obj = CRS.from_string(crs)

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with rasterio.open(
            tmp_path, "w",
            driver="GTiff",
            height=h, width=w,
            count=1,
            dtype=elevation.dtype,
            crs=crs_obj,
            transform=transform,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
        ) as dst:
            dst.write(elevation, 1)
            dst.overviews(1)
        import shutil
        shutil.move(tmp_path, str(output_path))
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    return output_path


def _elevation_to_geotiff_fallback(
    elevation: np.ndarray,
    output_path: str | Path | None = None,
) -> bytes | Path:
    if output_path is not None:
        output_path = Path(output_path)
        np.savez_compressed(str(output_path), elevation=elevation)
        return output_path
    buf = io.BytesIO()
    np.savez_compressed(buf, elevation=elevation)
    return buf.getvalue()
