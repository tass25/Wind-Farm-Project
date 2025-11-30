from __future__ import annotations
import os
from dataclasses import dataclass
from math import cos, radians
from typing import Optional, List, Literal

import geopandas as gpd
import numpy as np
import osmnx as ox
import rioxarray as rxr
import xarray as xr
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pyproj import CRS
from rasterio.features import rasterize
from shapely.geometry import box, mapping
from shapely.ops import unary_union


# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------

DEFAULT_WIND_RASTER = os.getenv(
    "WIND_RASTER_PATH",
    "app/data/TUN_wind-speed_100m.tif",
)

OUT_DIR = os.getenv("CONTEXT_OUT_DIR", "app/data/site_artifacts")
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------------------------------
# MODELS
# -----------------------------------------------------------

@dataclass
class SiteContext:
    name: str
    aoi: gpd.GeoDataFrame
    wind_clip: xr.DataArray        # normalized wind (0–1)
    allow_mask: xr.DataArray       # 0/1 mask
    wind_path: str                 # raw clipped wind path
    allow_path: str
    context_path: str              # 2-band [wind_norm, allow]


class ContextRequest(BaseModel):
    min_lon: Optional[float] = None
    min_lat: Optional[float] = None
    max_lon: Optional[float] = None
    max_lat: Optional[float] = None

    center_lon: Optional[float] = None
    center_lat: Optional[float] = None
    size_km: Optional[float] = 20.0

    name: Optional[str] = None
    return_mode: Literal["paths"] = "paths"


# -----------------------------------------------------------
# HELPERS
# -----------------------------------------------------------

def slugify(text: str) -> str:
    return (
        text.strip()
        .lower()
        .replace(",", "")
        .replace(" ", "_")
        .replace("/", "_")
    )


def bbox_from_center(lon: float, lat: float, size_km: float) -> List[float]:
    half_lat = size_km / 111.32
    half_lon = size_km / (111.32 * cos(radians(lat)))
    return [lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat]


def get_aoi_from_bbox(bbox: List[float]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        geometry=[box(*bbox)],
        crs="EPSG:4326"
    )


def _get_utm_crs_for_aoi(aoi: gpd.GeoDataFrame) -> CRS:
    centroid = aoi.geometry.unary_union.centroid
    lon, lat = centroid.x, centroid.y
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg(32600 + zone if lat >= 0 else 32700 + zone)


def _empty_gdf(crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[], crs=crs)


def _safe_geoms(gdf: Optional[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """Normalize to a valid (possibly empty) GeoDataFrame."""
    if gdf is None:
        return _empty_gdf()
    gdf = gdf[gdf.geometry.notnull()]
    if len(gdf) == 0:
        return _empty_gdf(gdf.crs or "EPSG:4326")
    return gdf


def _safe_features_from_polygon(poly, tags) -> gpd.GeoDataFrame:
    """
    Wrap osmnx.features_from_polygon so that:
    - "no matching features" doesn't crash
    - any other OSM error becomes an empty layer instead of 500
    """
    try:
        gdf = ox.features_from_polygon(poly, tags=tags)
        return _safe_geoms(gdf)
    except Exception as e:
        print(f"[OSM] tags={tags} returned no features or error: {e}")
        return _empty_gdf()


# -----------------------------------------------------------
# MASK BUILDER
# -----------------------------------------------------------

def build_allow_mask_from_osm(aoi, wind_clip, road_buffer_m=200.0, urban_buffer_m=0.0):

    aoi_wgs84 = aoi.to_crs("EPSG:4326")
    poly = aoi_wgs84.geometry.unary_union

    utm = _get_utm_crs_for_aoi(aoi_wgs84)

    # ---------------------------
    # Fetch all layers (safe)
    # ---------------------------

    water = _safe_features_from_polygon(
        poly, tags={"natural": ["water"], "waterway": True}
    )

    urban = _safe_features_from_polygon(
        poly,
        tags={
            "landuse": ["residential", "industrial", "commercial"],
            "place": ["city", "town", "village"],
        },
    )

    roads = _safe_features_from_polygon(
        poly, tags={"highway": True}
    )

    airports = _safe_features_from_polygon(
        poly, tags={"aeroway": ["aerodrome", "airport"]}
    )

    protected = _safe_features_from_polygon(
        poly,
        tags={"leisure": "nature_reserve", "boundary": "national_park"},
    )

    # ---------------------------
    # Build forbidden geometries
    # ---------------------------
    forbidden = []

    if len(water) > 0:
        forbidden.append(unary_union(water.to_crs(utm).geometry))

    if len(urban) > 0:
        urb = urban.to_crs(utm)
        forbidden.append(
            unary_union(
                urb.buffer(
                    urban_buffer_m) if urban_buffer_m > 0 else urb.geometry
            )
        )

    if len(roads) > 0:
        forbidden.append(unary_union(roads.to_crs(utm).buffer(road_buffer_m)))

    if len(airports) > 0:
        forbidden.append(unary_union(
            airports.to_crs(utm).buffer(road_buffer_m)))

    if len(protected) > 0:
        forbidden.append(unary_union(protected.to_crs(utm).geometry))

    # No constraints → everything allowed
    if len(forbidden) == 0:
        allow = xr.full_like(wind_clip.squeeze(), 1, dtype="uint8")
        return allow.rio.write_crs(wind_clip.rio.crs)

    # Union forbidden zones
    forbidden_union = unary_union(forbidden)

    # Reproject forbidden polygon to wind CRS
    wind_crs = wind_clip.rio.crs
    forbidden_wind = (
        gpd.GeoSeries([forbidden_union], crs=utm)
        .to_crs(wind_crs)
        .iloc[0]
    )

    transform = wind_clip.rio.transform()
    shape = (wind_clip.rio.height, wind_clip.rio.width)

    forbidden_raster = rasterize(
        [(mapping(forbidden_wind), 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    allow_array = 1 - forbidden_raster

    da = xr.DataArray(
        allow_array,
        coords={"y": wind_clip.y, "x": wind_clip.x},
        dims=("y", "x"),
        name="allow",
    )
    return da.rio.write_crs(wind_clip.rio.crs)


# -----------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------

def prepare_site_from_bbox(bbox, wind_raster_path=None, out_dir=OUT_DIR, name=None):

    if wind_raster_path is None:
        wind_raster_path = DEFAULT_WIND_RASTER

    if not os.path.exists(wind_raster_path):
        raise FileNotFoundError(f"Wind raster not found: {wind_raster_path}")

    aoi = get_aoi_from_bbox(bbox)

    name_slug = slugify(name) if name else slugify(
        f"bbox_{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}"
    )

    os.makedirs(out_dir, exist_ok=True)

    # 1) Clip raw wind
    wind = rxr.open_rasterio(wind_raster_path, masked=True)
    wind_clip_raw = wind.rio.clip(
        aoi.geometry, aoi.crs).squeeze().rename("wind")

    wind_path = os.path.join(out_dir, f"wind_{name_slug}.tif")
    wind_clip_raw.rio.to_raster(wind_path)

    # 2) Build allow mask on raw grid
    allow = build_allow_mask_from_osm(aoi, wind_clip_raw)

    allow_path = os.path.join(out_dir, f"allow_{name_slug}.tif")
    allow.rio.to_raster(allow_path)

    # 3) Normalize wind (per tile)
    w = wind_clip_raw
    w_min = w.min()
    w_max = w.max()
    wind_norm = (w - w_min) / (w_max - w_min + 1e-8)
    wind_norm = wind_norm.astype("float32")

    # 4) Combined 2-band context [wind_norm, allow]
    wind_data = wind_norm.values.astype("float32")
    allow_data = allow.values.astype("float32")

    stacked = np.stack([wind_data, allow_data], axis=0)

    context = xr.DataArray(
        stacked,
        coords={"band": [1, 2], "y": wind_norm.y, "x": wind_norm.x},
        dims=("band", "y", "x"),
        name="context",
    ).rio.write_crs(wind_norm.rio.crs)

    context_path = os.path.join(out_dir, f"context_{name_slug}.tif")
    context.rio.to_raster(context_path)

    return SiteContext(
        name=name_slug,
        aoi=aoi,
        wind_clip=wind_norm,      # normalized wind
        allow_mask=allow,         # 0/1
        wind_path=wind_path,      # raw clipped wind
        allow_path=allow_path,
        context_path=context_path,
    )


# -----------------------------------------------------------
# FASTAPI
# -----------------------------------------------------------

app = FastAPI(title="Context Map API", version="0.1.0")


@app.post("/context")
def create_context(req: ContextRequest):

    # Use explicit bbox if provided
    if all(
        v is not None
        for v in [req.min_lon, req.min_lat, req.max_lon, req.max_lat]
    ):
        bbox = [req.min_lon, req.min_lat, req.max_lon, req.max_lat]

    elif req.center_lon is not None and req.center_lat is not None:
        bbox = bbox_from_center(
            req.center_lon, req.center_lat, req.size_km or 20.0
        )

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either bbox or center_lon+center_lat.",
        )

    try:
        site = prepare_site_from_bbox(bbox, name=req.name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to prepare site: {e}")

    return {
        "name": site.name,
        "wind_raster": site.wind_path,          # raw clipped wind
        "allow_mask": site.allow_path,
        "context_raster": site.context_path,    # [wind_norm, allow]
    }
