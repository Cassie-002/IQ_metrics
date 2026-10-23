#!/usr/bin/env python3
"""
Noise power spectrum (NPS) inside a contrast-filled vessel.

Input:  a CT volume and a binary mask of a large vessel (we used the aorta).
Output: a CSV with the 1D (radially averaged) NPS, plus noise magnitude,
        average frequency and peak frequency in the header.

Steps:
  1. On each slice, place the largest square ROI that fits inside the vessel.
     The mask is eroded first so the ROI stays away from the vessel wall.
  2. If the ROI contains a calcification (HU > 500), erode further and try again.
  3. Crop all ROIs to the same size (the smallest one found).
  4. Per ROI: remove a 2nd order polynomial trend, zero-pad, FFT, square.
  5. Average all 2D spectra, then radially average to a 1D NPS.

Example:
    python nps.py --ct scan.nii.gz --mask aorta.nii.gz --out scan_nps.csv
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.integrate import trapezoid
from skimage.measure import label

# ROI placement
MARGIN_START = 0.85        # fraction of the vessel radius that is kept after erosion
MARGIN_STEP = 0.05         # how much extra erosion per try when a calcification is found
MARGIN_FLOOR = 0.4         # stop eroding here
CALC_HU_THRESHOLD = 500.0  # voxels above this count as calcification
MAX_ROIS_PER_SLICE = 2     # e.g. ascending + descending aorta on the same slice
MIN_COMPONENT_AREA = 30    # px, smaller blobs in the mask are skipped
MIN_ROI_SIDE = 16          # px, smaller ROIs are not used

# NPS
PAD_FACTOR = 8             # zero-padding factor before the FFT


# ---------------------------------------------------------------- ROI placement

def inscribed_square(mask_2d, margin):
    """Largest square inside the mask after eroding it by (1 - margin) * max radius."""
    if mask_2d.sum() < 25:
        return None
    dist = ndimage.distance_transform_edt(mask_2d)
    if dist.max() < 3:
        return None

    eroded = dist >= (1.0 - margin) * dist.max()
    if eroded.sum() < 9:
        return None

    # centre of the square = deepest point of the eroded mask
    dist2 = ndimage.distance_transform_edt(eroded)
    if dist2.max() < 2:
        return None
    cy, cx = np.unravel_index(np.argmax(dist2), dist2.shape)
    half = int(np.floor(dist2.max() / np.sqrt(2)))
    if half < 3:
        return None

    return {
        "y_min": max(0, cy - half), "y_max": min(mask_2d.shape[0], cy + half),
        "x_min": max(0, cx - half), "x_max": min(mask_2d.shape[1], cx + half),
    }


def roi_without_calcification(comp_mask, ct_slice):
    """Shrink the ROI step by step until it contains no calcification.
    If it never gets clean, the smallest ROI is kept anyway."""
    margin = MARGIN_START
    best = None
    while margin >= MARGIN_FLOOR - 1e-9:
        roi = inscribed_square(comp_mask, margin)
        if roi is None:
            break
        pixels = ct_slice[roi["y_min"]:roi["y_max"], roi["x_min"]:roi["x_max"]]
        roi["margin_used"] = round(margin, 3)
        roi["calc_clean"] = pixels.size > 0 and pixels.max() <= CALC_HU_THRESHOLD
        best = roi
        if roi["calc_clean"]:
            break
        margin -= MARGIN_STEP
    return best


def find_rois(mask, ct):
    """Returns {slice_index: [roi, ...]}, max MAX_ROIS_PER_SLICE ROIs per slice
    (one per connected part of the mask, largest first)."""
    rois = {}
    for z in range(mask.shape[0]):
        if not mask[z].any():
            continue
        labeled = label(mask[z], connectivity=2)
        parts = [(i, (labeled == i).sum()) for i in range(1, labeled.max() + 1)]
        parts = sorted([p for p in parts if p[1] >= MIN_COMPONENT_AREA], key=lambda p: -p[1])

        slice_rois = []
        for part_id, _ in parts[:MAX_ROIS_PER_SLICE]:
            roi = roi_without_calcification(labeled == part_id, ct[z])
            if roi is not None:
                slice_rois.append(roi)
        if slice_rois:
            rois[z] = slice_rois
    return rois


# ---------------------------------------------------------------- NPS

def detrend(roi):
    """Subtract a 2nd order polynomial surface fit."""
    ny, nx = roi.shape
    y, x = np.mgrid[0:ny, 0:nx]
    x, y = x.ravel().astype(float), y.ravel().astype(float)
    A = np.column_stack([np.ones_like(x), x, y, x**2, y**2, x * y])
    coeffs, *_ = np.linalg.lstsq(A, roi.ravel().astype(float), rcond=None)
    return roi - (A @ coeffs).reshape(ny, nx)


def nps_2d(roi, pixel_mm):
    """2D NPS of one square ROI: |FFT|^2 * dx*dy / (Nx*Ny)."""
    n = roi.shape[0]
    n_pad = n * PAD_FACTOR
    F = np.fft.fftshift(np.fft.fft2(detrend(roi), s=(n_pad, n_pad)))
    nps = np.abs(F) ** 2 * pixel_mm**2 / (n * n)
    freqs = np.fft.fftshift(np.fft.fftfreq(n_pad, d=pixel_mm))
    return nps, freqs


def radial_average(nps, freqs):
    """2D NPS -> 1D NPS, from 0 up to Nyquist."""
    fy, fx = np.meshgrid(freqs, freqs, indexing="ij")
    fr = np.hypot(fx, fy)
    f_nyq = 0.5 * len(freqs) * abs(freqs[1] - freqs[0])
    n_bins = max(nps.shape[0] // 2, 32)

    edges = np.linspace(0.0, f_nyq, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    nps_1d = np.full(n_bins, np.nan)
    for i in range(n_bins):
        in_bin = (fr >= edges[i]) & (fr < edges[i + 1])
        if in_bin.any():
            nps_1d[i] = nps[in_bin].mean()
    return centers, nps_1d


def compute_nps(ct, mask, pixel_mm):
    rois = find_rois(mask, ct)
    candidates = []
    for z, roi_list in rois.items():
        for r in roi_list:
            side = min(r["y_max"] - r["y_min"], r["x_max"] - r["x_min"])
            if side >= MIN_ROI_SIDE:
                candidates.append((z, r, side))
    if not candidates:
        raise RuntimeError("no usable ROIs found in the mask")

    # all ROIs get the size of the smallest one, so the spectra can be averaged
    side = min(c[2] for c in candidates)
    h = side // 2
    spectra = []
    for z, r, _ in candidates:
        cy = (r["y_min"] + r["y_max"]) // 2
        cx = (r["x_min"] + r["x_max"]) // 2
        roi = ct[z, cy - h:cy + h, cx - h:cx + h]
        if roi.shape == (side, side):
            nps, freqs = nps_2d(roi, pixel_mm)
            spectra.append(nps)

    f, nps_1d = radial_average(np.mean(spectra, axis=0), freqs)

    # noise magnitude = sqrt(area under the 1D NPS), average frequency = NPS-weighted mean
    ok = (f > 0) & np.isfinite(nps_1d)
    area = trapezoid(nps_1d[ok], f[ok])
    return {
        "f": f,
        "nps": nps_1d,
        "noise_hu": float(np.sqrt(area)),
        "f_av": float(trapezoid(f[ok] * nps_1d[ok], f[ok]) / area),
        "f_peak": float(f[ok][np.argmax(nps_1d[ok])]),
        "n_rois": len(spectra),
        "roi_side_px": side,
        "n_slices": len({c[0] for c in candidates}),
        "n_rois_not_clean": sum(not c[1]["calc_clean"] for c in candidates),
    }


# ---------------------------------------------------------------- in/out

def save_csv(path, res, pixel_mm):
    with open(path, "w", newline="") as fh:
        fh.write(f"# pixel_spacing_mm,{pixel_mm:.4f}\n")
        fh.write(f"# n_rois,{res['n_rois']}\n")
        fh.write(f"# roi_side_px,{res['roi_side_px']}\n")
        fh.write(f"# n_slices,{res['n_slices']}\n")
        fh.write(f"# n_rois_not_clean,{res['n_rois_not_clean']}\n")
        fh.write(f"# noise_HU,{res['noise_hu']:.4f}\n")
        fh.write(f"# f_av_mm-1,{res['f_av']:.4f}\n")
        fh.write(f"# f_peak_mm-1,{res['f_peak']:.4f}\n")
        fh.write("frequency_mm-1,nps_HU2mm2\n")
        for fi, ni in zip(res["f"], res["nps"]):
            fh.write(f"{fi:.6f},{ni:.6f}\n" if np.isfinite(ni) else f"{fi:.6f},\n")


def main():
    parser = argparse.ArgumentParser(description="NPS inside a vessel mask.")
    parser.add_argument("--ct", required=True, help="CT volume (NIfTI), in HU")
    parser.add_argument("--mask", required=True, help="binary vessel mask (NIfTI), same grid as the CT")
    parser.add_argument("--out", required=True, help="output CSV")
    args = parser.parse_args()

    img = sitk.ReadImage(args.ct)
    ct = sitk.GetArrayFromImage(img).astype(np.float32)  # [z, y, x]
    mask = sitk.GetArrayFromImage(sitk.ReadImage(args.mask)) > 0
    if mask.shape != ct.shape:
        raise SystemExit(f"mask shape {mask.shape} does not match CT shape {ct.shape}")
    pixel_mm = float(img.GetSpacing()[0])  # assumes square in-plane pixels

    res = compute_nps(ct, mask, pixel_mm)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_csv(args.out, res, pixel_mm)
    print(f"noise = {res['noise_hu']:.2f} HU, f_av = {res['f_av']:.3f} mm-1, "
          f"f_peak = {res['f_peak']:.3f} mm-1 ({res['n_rois']} ROIs)")


if __name__ == "__main__":
    main()
