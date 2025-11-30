import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
import rasterio.transform as rio_transform
from pyproj import Transformer

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

# Path to your trained synthetic model checkpoint
MODEL_PATH = os.getenv("DDPM_MODEL_PATH", "app/models/ddpm_unet_synthetic.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Globals that will be filled when loading the model
_model = None
_betas = None
_alphas = None
_alphas_cumprod = None
_alphas_cumprod_prev = None
_T_STEPS = None
_IMG_SIZE = None


# -------------------------------------------------------------------
# UNet DEFINITION (must match training)
# -------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if shapes mismatch by 1 pixel
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(
            x1,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, out_ch=1):
        super().__init__()
        self.inc = DoubleConv(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)

        self.up1 = Up(base_ch * 8, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 2)
        self.up3 = Up(base_ch * 2, base_ch)

        self.outc = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x, t_emb=None):
        # t_emb is ignored; model is unconditional wrt time in this simple version.
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


# -------------------------------------------------------------------
# DIFFUSION UTILITIES
# -------------------------------------------------------------------

def _init_diffusion_from_betas(betas: torch.Tensor):
    """
    Given betas (T,), precompute alphas, cumulative products, etc.
    """
    global _betas, _alphas, _alphas_cumprod, _alphas_cumprod_prev

    _betas = betas.to(device)
    _alphas = 1.0 - _betas
    _alphas_cumprod = torch.cumprod(_alphas, dim=0)
    _alphas_cumprod_prev = torch.cat(
        [torch.tensor([1.0], device=device), _alphas_cumprod[:-1]]
    )


@torch.no_grad()
def _sample_layout(
    model: nn.Module,
    context_tensor: torch.Tensor,  # (1, 2, H, W)
    steps: int,
) -> torch.Tensor:
    """
    DDPM-style sampling loop.
    Returns layout tensor in [0,1], shape (1, 1, H, W).
    """
    model.eval()

    betas = _betas
    alphas = _alphas
    alphas_cum = _alphas_cumprod
    alphas_cum_prev = _alphas_cumprod_prev

    B, C_ctx, H, W = context_tensor.shape
    assert B == 1, "Only batch size 1 supported in this simple sampler."

    y_t = torch.randn(1, 1, H, W, device=device)

    for i in reversed(range(steps)):
        t = torch.full((1,), i, device=device, dtype=torch.long)

        x_in = torch.cat([context_tensor.to(device), y_t],
                         dim=1)  # (1, 3, H, W)
        # (1, 1, H, W)
        eps_theta = model(x_in, t_emb=None)

        beta_t = betas[t].reshape(-1, 1, 1, 1)
        alpha_t = alphas[t].reshape(-1, 1, 1, 1)
        alpha_cum_t = alphas_cum[t].reshape(-1, 1, 1, 1)
        alpha_cum_prev_t = alphas_cum_prev[t].reshape(-1, 1, 1, 1)

        # Estimate x0
        y0_pred = (y_t - torch.sqrt(1 - alpha_cum_t) *
                   eps_theta) / torch.sqrt(alpha_cum_t)

        # Posterior mean p(x_{t-1} | x_t)
        coef1 = torch.sqrt(alpha_cum_prev_t) * beta_t / (1 - alpha_cum_t)
        coef2 = torch.sqrt(alpha_t) * \
            (1 - alpha_cum_prev_t) / (1 - alpha_cum_t)
        mean = coef1 * y0_pred + coef2 * y_t

        if i > 0:
            z = torch.randn_like(y_t)
            sigma = torch.sqrt(beta_t)
            y_t = mean + sigma * z
        else:
            y_t = mean

    return y_t.clamp(0, 1)


# -------------------------------------------------------------------
# MODEL LOADING
# -------------------------------------------------------------------

def load_ddpm_model(ckpt_path: str = MODEL_PATH) -> Tuple[UNet, int, int]:
    """
    Load the DDPM/UNet model and diffusion config from checkpoint.
    Returns:
      model, T_STEPS, IMG_SIZE
    """
    global _model, _T_STEPS, _IMG_SIZE

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    betas = ckpt["betas"]

    in_ch = cfg["in_ch"]      # should be 3
    out_ch = cfg["out_ch"]    # should be 1
    _T_STEPS = cfg["T_STEPS"]
    _IMG_SIZE = cfg["IMG_SIZE"]

    model = UNet(in_ch=in_ch, out_ch=out_ch).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _init_diffusion_from_betas(betas)

    _model = model
    return model, _T_STEPS, _IMG_SIZE


# Load at import time (you can also call load_ddpm_model() manually if you prefer)
if _model is None:
    try:
        load_ddpm_model(MODEL_PATH)
        print(
            f"[inference_service] Loaded model from {MODEL_PATH} on {device}.")
    except Exception as e:
        print(f"[inference_service] Warning: could not load model: {e}")


# -------------------------------------------------------------------
# CORE INFERENCE FUNCTION (LAYOUT MASK)
# -------------------------------------------------------------------

def predict_layout_from_context_geotiff(
    context_path: str,
    threshold: float = None,
) -> np.ndarray:
    """
    Main function you will call from your app.

    - Reads the context GeoTIFF created by context_api.py
      (2 bands: wind_norm, allow)
    - Resamples to the training IMG_SIZE
    - Runs diffusion sampling
    - Returns layout (H, W) float32 in [0,1]
      If threshold is not None, returns binary mask in {0,1}.

    Parameters
    ----------
    context_path : str
        Path to context_<name>.tif from context_api.
    threshold : float, optional
        If set (e.g. 0.5), output is binarized.

    Returns
    -------
    np.ndarray
        Layout mask, shape (IMG_SIZE, IMG_SIZE), values in [0,1] or {0,1}.
    """
    if _model is None:
        raise RuntimeError(
            "Model not loaded. Check MODEL_PATH or call load_ddpm_model().")

    if not os.path.exists(context_path):
        raise FileNotFoundError(f"context raster not found: {context_path}")

    # Read 2-band context and resample to training size
    with rasterio.open(context_path) as src:
        bands = src.read()  # (count, H, W)
        count, H, W = bands.shape

        if count < 2:
            raise ValueError(
                f"Expected at least 2 bands in context raster, got {count}.")

        # Resample first 2 bands [wind_norm, allow] to model size
        ctx_resampled = src.read(
            out_shape=(2, _IMG_SIZE, _IMG_SIZE),
            resampling=Resampling.bilinear,
        ).astype("float32")

    ctx_tensor = torch.from_numpy(ctx_resampled)[
        None, ...].to(device)  # (1, 2, H, W)

    layout_tensor = _sample_layout(
        _model, ctx_tensor, steps=_T_STEPS)  # (1, 1, H, W)
    layout = layout_tensor[0, 0].detach().cpu(
    ).numpy()                 # (H, W), [0,1]

    if threshold is not None:
        layout = (layout >= float(threshold)).astype("float32")

    return layout


# -------------------------------------------------------------------
# MASK → TURBINE POINTS (LON/LAT)
# -------------------------------------------------------------------

def predict_turbine_points_from_context_geotiff(
    context_path: str,
    threshold: float = 0.5,
):
    """
    High-level helper:
      - runs the DDPM model on the context raster
      - converts '1' pixels to geographic coordinates (lon/lat)

    Returns a list of dicts: [{"lon": ..., "lat": ...}, ...]
    """
    # 1) Get layout mask in _IMG_SIZE x _IMG_SIZE
    layout = predict_layout_from_context_geotiff(
        context_path=context_path,
        threshold=threshold,
    )  # shape: (IMG_SIZE, IMG_SIZE)

    # 2) Build the transform for the *resampled* grid
    with rasterio.open(context_path) as src:
        bounds = src.bounds
        src_crs = src.crs

    transform_resampled = from_bounds(
        bounds.left, bounds.bottom, bounds.right, bounds.top,
        _IMG_SIZE, _IMG_SIZE
    )

    # 3) Find turbine pixels (== 1.0)
    ys, xs = np.where(layout == 1.0)
    if len(ys) == 0:
        return []

    xs_geo, ys_geo = rio_transform.xy(transform_resampled, ys, xs)
    xs_geo = np.array(xs_geo)
    ys_geo = np.array(ys_geo)

    # 4) Reproject to WGS84 (lon/lat) if needed
    if src_crs is None:
        # Assume already lon/lat (fallback)
        lons, lats = xs_geo, ys_geo
    else:
        crs_str = src_crs.to_string()
        if crs_str in ("EPSG:4326", "WGS84"):
            lons, lats = xs_geo, ys_geo
        else:
            transformer = Transformer.from_crs(
                src_crs, "EPSG:4326", always_xy=True)
            lons, lats = transformer.transform(xs_geo, ys_geo)

    points = [
        {"lon": float(lon), "lat": float(lat)}
        for lon, lat in zip(lons, lats)
    ]
    return points


# -------------------------------------------------------------------
# FASTAPI WRAPPER
# -------------------------------------------------------------------

app = FastAPI(title="Wind Layout Inference API", version="0.1.0")


class PredictRequest(BaseModel):
    context_path: str
    threshold: float | None = 0.5


class PredictPointsRequest(BaseModel):
    context_path: str
    threshold: float | None = 0.5


@app.post("/predict-layout")
def predict_layout(req: PredictRequest):
    """
    Simple endpoint:
      Input:  {"context_path": "...", "threshold": 0.5}
      Output: {"height": H, "width": W, "mask": [[... rows ...]]}
    In a real app you might instead return a file path or a vector of turbine points.
    """
    try:
        layout = predict_layout_from_context_geotiff(
            req.context_path,
            threshold=req.threshold,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    H, W = layout.shape
    # For demo: convert small masks directly to JSON. For bigger areas, you'd write GeoTIFF or GeoJSON.
    return {
        "height": int(H),
        "width": int(W),
        "threshold": req.threshold,
        "mask": layout.tolist(),
    }


@app.post("/predict-points")
def predict_points(req: PredictPointsRequest):
    """
    Returns turbine locations as lon/lat points for a given context raster.
      Input:  {"context_path": "...", "threshold": 0.5}
      Output: {"count": N, "points": [{"lon": ..., "lat": ...}, ...]}
    """
    try:
        points = predict_turbine_points_from_context_geotiff(
            context_path=req.context_path,
            threshold=req.threshold if req.threshold is not None else 0.5,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return {
        "count": len(points),
        "points": points,
    }
