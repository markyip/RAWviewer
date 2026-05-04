"""
Focus-area saliency heatmap for single-image viewing (no scipy/opencv).

Fuses DRDF-style directional kernels (Surh CVPR17 / DRDF discrete kernels),
gradient energy (Tenengrad), and summed modified Laplacian, plus two cues
that reduce false positives from strong but defocused edges/patterns:

- Blur-attenuation residual: fine structure that collapses after light blur
  (optical sharpness) vs broad contrast that survives blurring.
- Directional entropy over DRDF kernel responses: fewer dominant directions
  (e.g. stripes) get down-weighted vs isotropic micro-texture.

Produces a jet-like full-image RGBA overlay and a peak that targets the
sharpest *region* via mass-centroid above a high percentile.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image


def _drdf_kernels() -> list[np.ndarray]:
    z = np.float32(0)
    t = np.float32(2)
    n = np.float32(-1)
    h1 = np.array(
        [
            [z, z, z, z, z],
            [z, z, z, z, z],
            [n, z, t, z, n],
            [z, z, z, z, z],
            [z, z, z, z, z],
        ],
        dtype=np.float32,
    )
    h2 = np.array(
        [
            [z, z, z, z, z],
            [z, z, z, z, n],
            [z, z, t, z, z],
            [n, z, z, z, z],
            [z, z, z, z, z],
        ],
        dtype=np.float32,
    )
    h3 = np.array(
        [
            [z, z, z, n, z],
            [z, z, z, z, z],
            [z, z, t, z, z],
            [z, z, z, z, z],
            [z, n, z, z, z],
        ],
        dtype=np.float32,
    )
    h4 = np.array(
        [
            [z, z, n, z, z],
            [z, z, z, z, z],
            [z, z, t, z, z],
            [z, z, z, z, z],
            [z, z, n, z, z],
        ],
        dtype=np.float32,
    )
    h5 = np.array(
        [
            [z, n, z, z, z],
            [z, z, z, z, z],
            [z, z, t, z, z],
            [z, z, z, z, z],
            [z, z, z, n, z],
        ],
        dtype=np.float32,
    )
    h6 = np.array(
        [
            [z, z, z, z, z],
            [n, z, z, z, z],
            [z, z, t, z, z],
            [z, z, z, z, n],
            [z, z, z, z, z],
        ],
        dtype=np.float32,
    )
    return [h1, h2, h3, h4, h5, h6]


def _conv2d_reflect_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    k = np.asarray(kernel, dtype=np.float32)
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="reflect")
    windows = sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, k, dtype=np.float32, optimize=True)


def _box_blur(gray: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        return gray.astype(np.float32, copy=False)
    x = gray.astype(np.float32, copy=False)
    k = 2 * radius + 1
    for axis in (0, 1):
        pad = np.pad(
            x,
            tuple((0, 0) if i != axis else (radius, radius) for i in range(2)),
            mode="reflect",
        )
        c = np.cumsum(pad, axis=axis, dtype=np.float32)
        # Same-size box filter: pad length is H + 2*radius = H + k - 1 along this axis.
        # Use cumsum differences k-1 apart (not k) so output matches input geometry.
        hi = tuple(slice(None) if i != axis else slice(k - 1, None) for i in range(2))
        lo = tuple(slice(None) if i != axis else slice(None, -(k - 1)) for i in range(2))
        x = (c[hi] - c[lo]) / float(k)
    return x


def _resize_gray_bilinear(gray: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if gray.shape[0] == out_h and gray.shape[1] == out_w:
        return gray.astype(np.float32, copy=False)
    im = Image.fromarray(gray.astype(np.float32), mode="F")
    res = im.resize((out_w, out_h), Image.Resampling.BILINEAR)
    return np.asarray(res, dtype=np.float32)


def _focus_drdf(gray: np.ndarray) -> np.ndarray:
    acc = np.zeros_like(gray, dtype=np.float32)
    for ker in _drdf_kernels():
        acc += np.abs(_conv2d_reflect_same(gray, ker))
    return acc


def _modified_laplacian(gray: np.ndarray) -> np.ndarray:
    """|∂²I/∂x²|+|∂²I/∂y²| on a 3×3 stencil."""
    kx = np.array([[1, -2, 1]], dtype=np.float32)
    ky = kx.reshape(3, 1)
    lx = np.abs(_conv2d_reflect_same(gray, kx))
    ly = np.abs(_conv2d_reflect_same(gray, ky))
    return lx + ly


def _tenengrad(gray: np.ndarray) -> np.ndarray:
    """Squared gradient magnitude (Sobel)."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    gx = _conv2d_reflect_same(gray, kx)
    gy = _conv2d_reflect_same(gray, ky)
    return gx * gx + gy * gy


def _gradient_blur_residual(gray: np.ndarray, blur_r: int = 2) -> np.ndarray:
    """Positive drop in Tenengrad energy after light blur — favors fine detail."""
    sm = _box_blur(gray, blur_r)
    t0 = _tenengrad(gray)
    t1 = _tenengrad(sm)
    return np.sqrt(np.maximum(t0 - t1, np.float32(0.0)))


def _laplacian_blur_residual(gray: np.ndarray, blur_r: int = 2) -> np.ndarray:
    """Modified Laplacian lost to light blur — suppresses broad defocus edges."""
    sm = _box_blur(gray, blur_r)
    m0 = _modified_laplacian(gray)
    m1 = _modified_laplacian(sm)
    return np.maximum(m0 - m1, np.float32(0.0))


def _drdf_direction_entropy(gray: np.ndarray) -> np.ndarray:
    """
    Normalized Shannon entropy (0..1) across the six DRDF responses per pixel.
    Stripes / one-sided patterns → low entropy; isotropic texture → high.
    """
    stacks: list[np.ndarray] = []
    for ker in _drdf_kernels():
        stacks.append(np.abs(_conv2d_reflect_same(gray, ker)))
    d = np.stack(stacks, axis=0).astype(np.float32)
    s = np.sum(d, axis=0, keepdims=True) + np.float32(1e-8)
    p = np.clip(d / s, np.float32(1e-10), np.float32(1.0))
    ent = np.sum(-(p * np.log(p)), axis=0).astype(np.float32)
    return ent / np.float32(np.log(6.0))


def _pct_normalize(x: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    lo = float(np.percentile(x, low_pct))
    hi = float(np.percentile(x, high_pct))
    if hi <= lo + 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0).astype(np.float32)


_JET_BREAKS_X = np.array(
    [
        0.0 / 765,
        3.0 / 765,
        6.0 / 765,
        20.0 / 765,
        89.0 / 765,
        223.0 / 765,
        591.0 / 765,
        705.0 / 765,
        764.0 / 765,
        765.0 / 765,
    ],
    dtype=np.float64,
)


def _matplotlib_jet_lookup(n: int = 768) -> np.ndarray:
    """Jet-like LUT, RGB floats in [0,1]."""
    r_k = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.125, 0.0], dtype=np.float64)
    g_k = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    b_k = np.array(
        [
            0.5628,
            0.7347,
            1.0000,
            1.0000,
            1.0000,
            0.5975,
            0.3596,
            0.0896,
            0.0586,
            0.0,
        ],
        dtype=np.float64,
    )
    xx = (_JET_BREAKS_X * 765).astype(np.float64)
    t = np.linspace(0.0, 765.0, n)
    rf = np.interp(t, xx, r_k).astype(np.float32)
    gf = np.interp(t, xx, g_k).astype(np.float32)
    bf = np.interp(t, xx, b_k).astype(np.float32)
    return np.clip(np.stack([rf, gf, bf], axis=1), 0.0, 1.0)


_LUT_JET_RGB = None


def _jet_rgba_from_heat(heat01: np.ndarray, alpha_max: float = 0.55) -> np.ndarray:
    global _LUT_JET_RGB  # pylint: disable=global-statement
    if _LUT_JET_RGB is None:
        _LUT_JET_RGB = _matplotlib_jet_lookup(768)

    t = np.clip(heat01, 0.0, 1.0).astype(np.float64)
    idx_f = np.clip(t * 767.0, 0.0, 767.0).astype(np.float64)
    i0 = np.floor(idx_f).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, 767)
    w = (idx_f - i0.astype(np.float64)).astype(np.float32)[..., np.newaxis]
    v0 = _LUT_JET_RGB[i0.flatten()].reshape(*(heat01.shape + (3,)))
    v1 = _LUT_JET_RGB[i1.flatten()].reshape(*(heat01.shape + (3,)))
    rgb = (v0.astype(np.float32) * (1.0 - w) + v1.astype(np.float32) * w).clip(0.0, 1.0)
    gamma = np.power(t.astype(np.float32), np.float32(0.92))
    alph = gamma * np.float32(alpha_max) * np.float32(255.0)
    rgba = np.zeros((*heat01.shape, 4), dtype=np.uint8)
    rgba[:, :, :3] = (rgb * 255.0 + 0.5).astype(np.uint8)
    rgba[:, :, 3] = alph.astype(np.uint8)
    return rgba


def _sharp_peak_xy(fm_smooth: np.ndarray, fm_peak: np.ndarray) -> tuple[float, float]:
    """
    Coarse centroid of high-response region in work-pixel coords:
    returns (px, py) with px = column, py = row.
    """
    h, wch = fm_smooth.shape
    thr = float(np.percentile(fm_smooth.astype(np.float64), 93.5))
    mask = fm_smooth >= thr
    need = max(48, min(h, wch))
    if int(mask.sum()) < need:
        thr = float(np.percentile(fm_smooth.astype(np.float64), 85.0))
        mask = fm_smooth >= thr

    fy, fx = np.where(mask)
    if fx.size >= 24:
        wts = fm_peak[fy, fx].astype(np.float64)
        wsum = float(np.maximum(wts.sum(), 1e-18))
        wts /= wsum
        py = float(np.dot(fy.astype(np.float64), wts))
        px = float(np.dot(fx.astype(np.float64), wts))
        return px, py

    r, c = np.unravel_index(int(np.argmax(fm_peak)), fm_peak.shape)
    return float(c), float(r)


def compute_focus_heatmap(
    rgb_uint8: np.ndarray,
    max_side: int = 896,
    fuse_half_scale: bool = True,
) -> tuple[int, int, np.ndarray]:
    """
    :param rgb_uint8: HxWx3 uint8
    :return: (peak_x, peak_y, rgba_uint8) — jet-style overlay covering the full image:
             cold (blue/purple) = least sharp; hot (yellow/red) = sharpest.
    """
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] < 3:
        raise ValueError("rgb_uint8 must be HxWx3")
    h0, w0 = rgb_uint8.shape[0], rgb_uint8.shape[1]
    r = rgb_uint8[:, :, 0].astype(np.float32)
    gch = rgb_uint8[:, :, 1].astype(np.float32)
    bch = rgb_uint8[:, :, 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * gch + 0.114 * bch

    scale = min(1.0, float(max_side) / max(h0, w0))
    nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
    nh, nw = max(16, nh), max(16, nw)
    work = _resize_gray_bilinear(gray, nh, nw)

    drdf = _focus_drdf(work)
    mlap = _modified_laplacian(work)
    ten = _tenengrad(work)
    ten_sqrt = np.sqrt(np.clip(ten, 1e-20, np.finfo(np.float32).max))

    if fuse_half_scale and min(nh, nw) >= 32:
        sh, sw = max(16, nh // 2), max(16, nw // 2)
        sm = _resize_gray_bilinear(work, sh, sw)
        dr_up = _resize_gray_bilinear(_focus_drdf(sm), nh, nw)
        drdf = np.maximum(drdf, 0.7 * dr_up)

    d_n = _pct_normalize(drdf, 3.5, 99.25)
    m_n = _pct_normalize(mlap, 3.5, 99.25)
    t_n = _pct_normalize(ten_sqrt, 3.5, 99.25)

    g_res = _gradient_blur_residual(work, blur_r=2)
    l_res = _laplacian_blur_residual(work, blur_r=2)
    g_rn = _pct_normalize(g_res, 4.0, 99.0)
    l_rn = _pct_normalize(l_res, 4.0, 99.0)
    blur_att = np.sqrt(np.maximum(g_rn * l_rn, np.float32(1e-20)))

    ent = _drdf_direction_entropy(work)
    ent_w = np.float32(0.28) + np.float32(0.72) * np.clip(ent, 0.0, 1.0)

    fm = np.maximum(
        np.cbrt(np.maximum(d_n * m_n * t_n * blur_att, np.float32(1e-20))),
        np.float32(0.35)
        * np.maximum(
            d_n,
            np.maximum(m_n, np.maximum(t_n, blur_att)),
        ),
    )
    fm = fm * ent_w

    fm_smooth = _box_blur(fm, 2)
    p_low = float(np.percentile(fm_smooth, 1.0))
    p_high = float(np.percentile(fm_smooth, 99.0))
    if p_high <= p_low + 1e-14:
        heat = np.zeros((nh, nw), dtype=np.float32)
        peak_x = max(0, w0 // 2)
        peak_y = max(0, h0 // 2)
    else:
        p_lo = float(np.percentile(fm_smooth, 5.5))
        p_hi = float(np.percentile(fm_smooth, 99.1))
        if p_hi <= p_lo + 1e-14:
            heat = np.zeros((nh, nw), dtype=np.float32)
            peak_x = max(0, w0 // 2)
            peak_y = max(0, h0 // 2)
        else:
            heat = np.clip((fm_smooth - p_lo) / (p_hi - p_lo + 1e-14), 0.0, 1.0).astype(
                np.float32
            )
            heat = np.power(heat + 1e-6, np.float32(0.82))

            cx_f, cy_f = _sharp_peak_xy(fm_smooth, fm)

            rad = max(28, min(nh, nw) // 8)
            wx0 = max(0, int(cx_f - rad))
            wx1 = min(nw, int(cx_f + rad + 1))
            wy0 = max(0, int(cy_f - rad))
            wy1 = min(nh, int(cy_f + rad + 1))

            fm_clip = fm[wy0:wy1, wx0:wx1].astype(np.float32)
            if fm_clip.size > 80:
                thr_p = np.percentile(fm_clip, 82.0)
                patch_weight = np.where(fm_clip >= thr_p, fm_clip, np.float32(0.0))
                s_mass = float(patch_weight.sum())
                if s_mass > 1e-12:
                    yy, xx = np.indices(patch_weight.shape, dtype=np.float32)
                    py = wy0 + float(np.dot(yy.ravel(), patch_weight.ravel()) / s_mass)
                    px = wx0 + float(np.dot(xx.ravel(), patch_weight.ravel()) / s_mass)
                else:
                    rr, cc = np.unravel_index(int(np.argmax(fm_clip)), fm_clip.shape)
                    px = float(wx0 + cc)
                    py = float(wy0 + rr)
            else:
                px, py = cx_f, cy_f

            px = float(np.clip(px, 0.0, float(max(0, nw - 1))))
            py = float(np.clip(py, 0.0, float(max(0, nh - 1))))

            sx = (w0 - 1) / max(nw - 1, 1)
            sy = (h0 - 1) / max(nh - 1, 1)
            peak_x = int(max(0, min(round(px * sx), max(0, w0 - 1))))
            peak_y = int(max(0, min(round(py * sy), max(0, h0 - 1))))

    heat_full = _resize_gray_bilinear(heat, h0, w0)
    rgba = _jet_rgba_from_heat(heat_full.astype(np.float32))
    return peak_x, peak_y, rgba
