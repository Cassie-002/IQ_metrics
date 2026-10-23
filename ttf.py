#!/usr/bin/env python3
"""
Task transfer function (TTF) measured at the wall of the descending aorta.

Based on IQ-PAT: Lacombe et al., Physica Medica 147 (2026) 105796,
sections 2.1.3.1 - 2.1.3.5.

Input:  a CT volume and a binary aorta mask.
Output: a CSV with the TTF curve, and f50 in the header.

Steps:
  1. Find the descending aorta: slices where the mask is one round blob
     whose outline length is within +-20% of the median.
  2. Take 100 consecutive slices around the median slice (or the longest
     run available if there are fewer).
  3. Draw radial profiles from the centre of the aorta through each point of
     the outline. Keep only profiles that cross the wall at a right angle
     (|cos(theta)| <= 0.1).
  4. Reject profiles with odd HU values: lumen outside mean +-2 SD, tail
     outside the fat range (-195 to 45 HU), a peak above 600 HU, or more
     than 2 edges.
  5. Fit a Fermi function to each profile to find the exact edge position.
     Reject bad fits (R² < 0.95 near the edge; this check is our own addition).
  6. Upsample 10x, shift all profiles so the edges line up, keep the central 50%.
  7. Drop profiles whose tail HU is more than 2 SD from the median.
  8. ESF = median of the profiles. LSF = derivative of the ESF (Hann window).
     TTF = |FFT(LSF)|, normalised to 1 at f = 0.

Example:
    python ttf.py --ct scan.nii.gz --mask aorta.nii.gz --out scan_ttf.csv
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.optimize import OptimizeWarning, curve_fit
from skimage.measure import find_contours, label

# bad fits are caught by the R² check below, no need for the warning
warnings.filterwarnings("ignore", category=OptimizeWarning)

# from the IQ-PAT paper
COS_THETA_MAX = 0.10            # profile must be (almost) perpendicular to the wall
MEDIAN_TOLERANCE = 0.20         # slice selection: outline length within +-20% of median
BLOCK_SIZE = 100                # number of consecutive slices
OVERSAMPLE = 10
CROP_FRACTION = 0.50
TAIL_OUTLIER_SD = 2.0
FAT_HU_RANGE = (-195.0, 45.0)   # HU outside the aorta should look like fat
MAX_EDGES = 2

# not given in the paper, our choice
TANGENT_OFFSET_MM = 1.5         # distance to neighbours used for the outline direction
END_LENGTH_MM = 2.0             # length at each profile end used for lumen / tail HU
PEAK_HU_MAX = 600.0             # profiles with a higher peak are rejected

# extra check on the Fermi fit, not in the paper
FIT_WINDOW_MM = 5.0
FIT_R2_MIN = 0.95

PAD_FACTOR = 8


# ---------------------------------------------------------------- slice selection

def outline(mask_2d):
    contours = find_contours(mask_2d.astype(float), 0.5)
    return max(contours, key=len) if contours else None


def single_blob_slices(mask):
    """Slices where the mask is exactly one blob. Returns outline lengths and masks."""
    lengths, masks = {}, {}
    for z in range(mask.shape[0]):
        labeled = label(mask[z], connectivity=2)
        if labeled.max() != 1 or (labeled == 1).sum() < 50:
            continue
        c = outline(labeled == 1)
        if c is None:
            continue
        lengths[z] = len(c) - 1
        masks[z] = labeled == 1
    return lengths, masks


def select_slices(lengths):
    median = np.median(list(lengths.values()))
    lo, hi = median * (1 - MEDIAN_TOLERANCE), median * (1 + MEDIAN_TOLERANCE)
    valid = sorted(z for z, n in lengths.items() if lo <= n <= hi)

    if len(valid) < BLOCK_SIZE:
        # not enough slices, use the longest run of consecutive slices
        print(f"warning: only {len(valid)} valid slices, using the longest consecutive run")
        runs, run = [], [valid[0]]
        for z in valid[1:]:
            if z == run[-1] + 1:
                run.append(z)
            else:
                runs.append(run)
                run = [z]
        runs.append(run)
        return max(runs, key=len)

    # 100 consecutive valid slices that include the slice closest to the median
    center = min(valid, key=lambda z: abs(lengths[z] - median))
    valid_set = set(valid)
    for start in range(center - BLOCK_SIZE + 1, center + 1):
        block = list(range(start, start + BLOCK_SIZE))
        if all(z in valid_set for z in block):
            return block
    raise RuntimeError(f"no {BLOCK_SIZE} consecutive valid slices found around the median")


# ---------------------------------------------------------------- profiles

def radial_profiles(ct_slice, mask_2d, pixel_mm, half_length_mm):
    """Profiles from inside the aorta (negative distance) to outside (positive)."""
    c = outline(mask_2d)
    if c is None:
        return []
    if np.linalg.norm(c[0] - c[-1]) < 1e-9:
        c = c[:-1]
    # light smoothing of the outline, otherwise the pixel steps mess up the tangent
    c = np.column_stack([gaussian_filter1d(c[:, 0], 1.0, mode="wrap"),
                         gaussian_filter1d(c[:, 1], 1.0, mode="wrap")])

    ys, xs = np.where(mask_2d)
    cy, cx = ys.mean(), xs.mean()
    offset = max(3, round(TANGENT_OFFSET_MM / pixel_mm))
    n_half = round(half_length_mm / pixel_mm)
    t = np.arange(-n_half, n_half + 1, dtype=float)
    n = len(c)

    profiles = []
    for i in range(n):
        py, px = c[i]
        ry, rx = py - cy, px - cx               # radial direction
        r = np.hypot(ry, rx)
        ty, tx = c[(i + offset) % n] - c[(i - offset) % n]   # outline direction
        tl = np.hypot(ty, tx)
        if r < 1e-9 or tl < 1e-9:
            continue
        ry, rx, ty, tx = ry / r, rx / r, ty / tl, tx / tl

        if abs(ry * ty + rx * tx) > COS_THETA_MAX:
            continue
        hu = map_coordinates(ct_slice, [py + t * ry, px + t * rx],
                             order=1, mode="constant", cval=np.nan)
        if np.isfinite(hu).sum() < 0.9 * len(hu):
            continue  # profile runs out of the image
        profiles.append({"d": t * pixel_mm, "hu": hu})
    return profiles


def end_means(p):
    d, hu = p["d"], p["hu"]
    lumen = np.nanmean(hu[d <= d.min() + END_LENGTH_MM])
    tail = np.nanmean(hu[d >= d.max() - END_LENGTH_MM])
    return lumen, tail


def is_clean(p, lumen_lo, lumen_hi):
    lumen, tail = p["lumen"], p["tail"]
    if not (lumen_lo <= lumen <= lumen_hi):
        return False
    if not (FAT_HU_RANGE[0] <= tail <= FAT_HU_RANGE[1]):
        return False
    if np.nanmax(p["hu"]) > PEAK_HU_MAX:
        return False
    # count how often the profile crosses halfway between lumen and tail
    above = (p["hu"] > 0.5 * (lumen + tail)).astype(np.int8)
    return np.count_nonzero(np.diff(above)) <= MAX_EDGES


# ---------------------------------------------------------------- Fermi fit + alignment

def fermi(x, a, b, c):
    return a / (1.0 + np.exp(np.clip((x - b) / c, -700, 700)))


def fit_edge(d, hu, tail):
    """Fit a / (1 + exp((x-b)/c)). The tail HU is subtracted first, because the
    function goes to 0 on the outside. Returns (b, params) or (None, None)."""
    ok = np.isfinite(hu)
    if ok.sum() < 8:
        return None, None
    x, y = d[ok], hu[ok] - tail
    a0 = y.max() - y.min()
    if a0 <= 0:
        return None, None
    b0 = x[np.argmin(np.abs(y - (y.min() + 0.5 * a0)))]
    try:
        params, _ = curve_fit(fermi, x, y, p0=[a0, b0, 0.7], maxfev=5000)
    except (RuntimeError, ValueError):
        return None, None
    a, b, c = params
    if not (np.isfinite(b) and c > 0 and x.min() < b < x.max()):
        return None, None

    # R² only close to the edge (the flat parts always fit well)
    near = (np.abs(d - b) <= FIT_WINDOW_MM) & np.isfinite(hu)
    if near.sum() < 4:
        return None, None
    y_near = hu[near]
    ss_res = np.sum((y_near - (fermi(d[near], *params) + tail)) ** 2)
    ss_tot = np.sum((y_near - y_near.mean()) ** 2)
    if ss_tot <= 0 or 1 - ss_res / ss_tot < FIT_R2_MIN:
        return None, None
    return b, params


def align(p):
    """Upsample, put the edge at 0 and keep the central part. Returns (d, hu) or None."""
    b, _ = fit_edge(p["d"], p["hu"], p["tail"])
    if b is None:
        return None
    d, hu = p["d"], p["hu"]
    ok = np.isfinite(hu)
    d_up = np.linspace(d[0], d[-1], (len(d) - 1) * OVERSAMPLE + 1)
    hu_up = np.interp(d_up, d[ok], hu[ok])
    d_up = d_up - b

    keep = np.abs(d_up) <= 0.5 * CROP_FRACTION * (d[-1] - d[0])
    if keep.sum() < 10:
        return None
    return d_up[keep], hu_up[keep]


# ---------------------------------------------------------------- ESF -> TTF

def crossing(freqs, ttf, level):
    """First frequency where the TTF drops below `level` (linear interpolation)."""
    idx = np.where((ttf[:-1] >= level) & (ttf[1:] < level))[0]
    if len(idx) == 0:
        return float("nan")
    i = idx[0]
    if ttf[i + 1] == ttf[i]:
        return float(freqs[i])
    frac = (level - ttf[i]) / (ttf[i + 1] - ttf[i])
    return float(freqs[i] + frac * (freqs[i + 1] - freqs[i]))


def compute_ttf(ct, mask, pixel_mm):
    lengths, masks = single_blob_slices(mask)
    if not lengths:
        raise RuntimeError("no slices where the aorta mask is a single blob")
    block = select_slices(lengths)

    # profile length = radius of the smallest aorta in the block
    half_length = min(np.sqrt(masks[z].sum() / np.pi) for z in block) * pixel_mm

    profiles = []
    for z in block:
        for p in radial_profiles(ct[z], masks[z], pixel_mm, half_length):
            p["lumen"], p["tail"] = end_means(p)
            profiles.append(p)
    if not profiles:
        raise RuntimeError("no radial profiles found")

    # lumen HU limits: mean +- 2 SD over all profiles
    lumens = np.array([p["lumen"] for p in profiles])
    lumen_lo, lumen_hi = lumens.mean() - 2 * lumens.std(), lumens.mean() + 2 * lumens.std()

    aligned = []
    for p in profiles:
        if is_clean(p, lumen_lo, lumen_hi):
            a = align(p)
            if a is not None:
                aligned.append((a[0], a[1], p["tail"]))
    if not aligned:
        raise RuntimeError("no usable profiles left after filtering")

    # drop tail outliers
    tails = np.array([a[2] for a in aligned])
    med, sd = np.median(tails), tails.std()
    kept = [a for a in aligned if sd == 0 or abs(a[2] - med) <= TAIL_OUTLIER_SD * sd]

    # common grid, median -> ESF
    dx = float(np.median([np.median(np.diff(d)) for d, _, _ in kept]))
    grid = np.arange(max(d[0] for d, _, _ in kept), min(d[-1] for d, _, _ in kept) + 0.5 * dx, dx)
    esf = np.median([np.interp(grid, d, hu) for d, hu, _ in kept], axis=0)

    # LSF with Hann window -> TTF
    lsf = np.gradient(esf, grid) * np.hanning(len(esf))
    n_pad = len(lsf) * PAD_FACTOR
    F = np.abs(np.fft.rfft(lsf, n=n_pad))
    ttf = F / F[0]
    freqs = np.fft.rfftfreq(n_pad, d=dx)

    below_nyq = freqs <= 0.5 / pixel_mm
    return {
        "f": freqs,
        "ttf": ttf,
        "f50": crossing(freqs[below_nyq], ttf[below_nyq], 0.5),
        "n_slices": len(block),
        "z_range": (block[0], block[-1]),
        "n_profiles": len(kept),
    }


# ---------------------------------------------------------------- in/out

def save_csv(path, res, pixel_mm):
    with open(path, "w", newline="") as fh:
        fh.write(f"# pixel_spacing_mm,{pixel_mm:.6f}\n")
        fh.write(f"# slices,{res['z_range'][0]}-{res['z_range'][1]}\n")
        fh.write(f"# n_profiles,{res['n_profiles']}\n")
        fh.write(f"# f50_mm-1,{res['f50']:.6f}\n")
        fh.write("frequency_mm-1,ttf\n")
        for fi, ti in zip(res["f"], res["ttf"]):
            fh.write(f"{fi:.6f},{ti:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="TTF at the descending aorta wall (IQ-PAT).")
    parser.add_argument("--ct", required=True, help="CT volume (NIfTI), in HU")
    parser.add_argument("--mask", required=True, help="binary aorta mask (NIfTI), same grid as the CT")
    parser.add_argument("--out", required=True, help="output CSV")
    args = parser.parse_args()

    img = sitk.ReadImage(args.ct)
    ct = sitk.GetArrayFromImage(img).astype(np.float32)  # [z, y, x]
    mask = sitk.GetArrayFromImage(sitk.ReadImage(args.mask)) > 0
    if mask.shape != ct.shape:
        raise SystemExit(f"mask shape {mask.shape} does not match CT shape {ct.shape}")
    pixel_mm = float(img.GetSpacing()[0])  # assumes square in-plane pixels

    res = compute_ttf(ct, mask, pixel_mm)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_csv(args.out, res, pixel_mm)
    print(f"f50 = {res['f50']:.4f} mm-1 ({res['n_profiles']} profiles, {res['n_slices']} slices)")


if __name__ == "__main__":
    main()
