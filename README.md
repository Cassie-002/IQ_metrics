# CT image quality metrics

Scripts to calculate three image quality metrics on patient CT scans:

| script      | metric | where it is measured |
|-------------|--------|----------------------|
| `nps.py`    | noise power spectrum (NPS), noise magnitude, average and peak frequency | inside the aorta lumen |
| `ttf.py`    | task transfer function (TTF), f50 | wall of the descending aorta (IQ-PAT method) |
| `dprime.py` | detectability index d' (NPWE model observer) | uses the NPS and TTF output |

The scans need an aorta segmentation first (for example from TotalSegmentator).
The CT and mask have to be NIfTI files on the same grid.

## Install

```
pip install numpy scipy pandas scikit-image SimpleITK
```

## Use

```
python nps.py    --ct scan.nii.gz --mask aorta.nii.gz --out scan_nps.csv
python ttf.py    --ct scan.nii.gz --mask aorta.nii.gz --out scan_ttf.csv
python dprime.py --ttf scan_ttf.csv --nps scan_nps.csv --contrast-hu 300
```

For d', `--contrast-hu` is the contrast of the lesion. We used the mean HU in
the pulmonary artery of each patient. Other options: `--diameter-mm` (default 3),
`--steepness-mm` (default 1), `--eye-filter` (Eckstein, off by default) and
`--viewing-distance-mm` (default 500).

To run a whole dataset, loop over the scans, e.g. in bash:

```
for d in data/*/; do
    python nps.py --ct "$d/ct.nii.gz" --mask "$d/aorta.nii.gz" --out "results/$(basename $d)_nps.csv"
done
```

## Reference

TTF method: Lacombe et al., IQ-PAT: An innovative tool for noise power spectrum
and task-based transfer function assessment in patient CT images,
Physica Medica 147 (2026) 105796.
