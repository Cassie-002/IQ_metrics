#!/usr/bin/env python3
"""
Detectability index d' (non-prewhitening model observer, optional eye filter).

           [ ∫∫ TTF² W² E² du dv ]²
    d'² = -------------------------
           ∫∫ NPS TTF² W² E⁴ du dv

  TTF  from ttf.py
  NPS  from nps.py
  W    spectrum of the task: a round lesion with a soft (sigmoid) edge,
       scaled by the lesion contrast in HU
  E    eye filter (Eckstein et al. 2003). Off by default, then E = 1.

TTF, NPS and W are all assumed to be the same in every direction, so the
2D integrals become 1D: ∫∫ g du dv = 2π ∫ g(f) f df.

Example:
    python dprime.py --ttf scan_ttf.csv --nps scan_nps.csv --contrast-hu 300
"""

import argparse

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.special import j0

# task (lesion) defaults
LESION_DIAMETER_MM = 3.0
EDGE_STEEPNESS_MM = 1.0    # width of the sigmoid edge
TASK_R_MAX_MM = 15.0       # radius up to which the lesion profile is integrated
TASK_N_R = 2000

# eye filter: E(f) = f^1.3 * exp(-0.0016 f²), with f in cycles/degree
ECKSTEIN_EXPONENT = 1.3
ECKSTEIN_COEFF = 0.0016
VIEWING_DISTANCE_MM = 500.0


def lesion_profile(r, diameter_mm, steepness_mm):
    """1 inside the lesion, 0 outside, with a sigmoid edge at r = diameter/2."""
    return 1.0 / (1.0 + np.exp((r - diameter_mm / 2.0) / steepness_mm))


def task_spectrum(f, diameter_mm, steepness_mm):
    """2D Fourier transform of the round lesion (Hankel transform):
    W(f) = 2π ∫ w(r) J0(2π f r) r dr. Unit contrast."""
    r = np.linspace(0.0, TASK_R_MAX_MM, TASK_N_R)
    w = lesion_profile(r, diameter_mm, steepness_mm)
    return np.array([2 * np.pi * simpson(w * j0(2 * np.pi * fi * r) * r, x=r) for fi in f])


def eye_filter(f_mm, viewing_distance_mm):
    """Eckstein eye filter. cycles/mm -> cycles/degree using the viewing distance."""
    f_deg = f_mm * viewing_distance_mm * np.tan(np.deg2rad(1.0))
    return f_deg**ECKSTEIN_EXPONENT * np.exp(-ECKSTEIN_COEFF * f_deg**2)


def compute_dprime(ttf_f, ttf, nps_f, nps, contrast_hu,
                   diameter_mm=LESION_DIAMETER_MM, steepness_mm=EDGE_STEEPNESS_MM,
                   use_eye_filter=False, viewing_distance_mm=VIEWING_DISTANCE_MM):
    # use the NPS frequencies (they stop at a lower frequency than the TTF), skip f = 0
    keep = (nps_f > 0) & np.isfinite(nps)
    f = nps_f[keep]
    nps_f_grid = nps[keep]
    ttf_f_grid = np.interp(f, ttf_f, ttf)

    w = task_spectrum(f, diameter_mm, steepness_mm) * contrast_hu
    e = eye_filter(f, viewing_distance_mm) if use_eye_filter else np.ones_like(f)

    num = 2 * np.pi * simpson(ttf_f_grid**2 * w**2 * e**2 * f, x=f)
    den = 2 * np.pi * simpson(nps_f_grid * ttf_f_grid**2 * w**2 * e**4 * f, x=f)
    return float(np.sqrt(num**2 / den))


def read_curve(path, ycol):
    df = pd.read_csv(path, comment="#")
    return df["frequency_mm-1"].to_numpy(), df[ycol].to_numpy()


def main():
    parser = argparse.ArgumentParser(description="d' from a TTF and an NPS curve.")
    parser.add_argument("--ttf", required=True, help="TTF CSV from ttf.py")
    parser.add_argument("--nps", required=True, help="NPS CSV from nps.py")
    parser.add_argument("--contrast-hu", type=float, required=True,
                        help="lesion contrast in HU (we used the mean HU in the pulmonary artery)")
    parser.add_argument("--diameter-mm", type=float, default=LESION_DIAMETER_MM)
    parser.add_argument("--steepness-mm", type=float, default=EDGE_STEEPNESS_MM)
    parser.add_argument("--eye-filter", action="store_true", help="use the Eckstein eye filter")
    parser.add_argument("--viewing-distance-mm", type=float, default=VIEWING_DISTANCE_MM)
    args = parser.parse_args()

    ttf_f, ttf = read_curve(args.ttf, "ttf")
    nps_f, nps = read_curve(args.nps, "nps_HU2mm2")
    d = compute_dprime(ttf_f, ttf, nps_f, nps, args.contrast_hu,
                       diameter_mm=args.diameter_mm, steepness_mm=args.steepness_mm,
                       use_eye_filter=args.eye_filter,
                       viewing_distance_mm=args.viewing_distance_mm)
    print(f"d' = {d:.3f}")


if __name__ == "__main__":
    main()
