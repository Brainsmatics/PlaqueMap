# Module 3 — Plaque Analysis Tools

Module 3 of a plaque image analysis pipeline: three **standalone** analysis tools for plaque characterization. Unlike Module 2 (which is a sequential pipeline), the three scripts here are **independent of each other** — there is no data flow between them, and each can be run on its own.

| Script | Purpose |
|--------|---------|
| `feature similarity.py` | Quantify how similar the feature spaces of two plaque subtypes/morphotypes are |
| `post processing.py` | Merge the feature / coordinate / classification tables into one summary table |
| `distribution difference.py` | Test whether subtype distributions differ across brain regions / between subtypes |

## Repository Structure

```
module3_analysis/
├── feature similarity.py           # feature-space similarity (multiple metrics + KDE)
├── post processing.py              # merge feature / coordinate / classification tables
├── distribution difference.py      # KS test + multiple tests (t-test / Hotelling T² / permutation)
├── test/
│   ├── test_2D.xlsx                # input for feature similarity (2D features + "multi" cluster labels)
│   └── output/                     # feature similarity outputs (figures, txt, summary csv)
└── distribution/
    ├── test_data/
    │   ├── features_test.csv       # input for the multiple tests
    │   └── count_regions/*.xlsx    # input for the KS test (region counts per brain area)
    ├── out_ks/                     # KS test outputs (auto-created)
    └── stat_t.xlsx                 # multiple-test output
```

## Dependencies

```bash
pip install pandas numpy scipy scikit-learn matplotlib umap-learn openpyxl
```

> `umap-learn` is only required by `feature similarity.py`; note the pip package is `umap-learn` but it is imported as `import umap`. `openpyxl` is needed for `.xlsx` read/write.

| Script | Required packages |
|--------|-------------------|
| `feature similarity.py` | `numpy pandas scipy scikit-learn matplotlib umap-learn openpyxl` |
| `post processing.py` | `pandas openpyxl` |
| `distribution difference.py` | `numpy pandas scipy openpyxl` |

## Usage

Run each script from inside `module3_analysis/` so the relative paths resolve. The three scripts are independent — run whichever you need.

```bash
python feature\ similarity.py      # similarity between two subtypes / morphotypes
python post\ processing.py         # merge tables into one summary
python distribution\ difference.py # distribution difference tests
```

### 1. Feature Similarity (`feature similarity.py`)

Compares the feature spaces of two plaque subtypes (or morphotypes) using several metrics — manifold overlap (UMAP + KDE), statistical distribution (effect size, distance distributions, correlation difference, CDF), separability score (SVM), feature importance (Random Forest), JS divergence, an overall similarity score, and a KDE overlap ratio.

- **Input**: `test/test_2D.xlsx` (must contain the `feature_names_2D` columns and a `multi` cluster-label column).
- **Output**: `test/output/` — master figures (`M{cls1}-M{cls2}.png`), per-pair text summaries, KDE plots, and `kde_comparison_summary.csv`.
- It loops over all pairs of cluster labels (`multi` 0–20 × 0–20), so it is the most compute-heavy script.

### 2. Post Processing (`post processing.py`)

Merges three per-plaque tables — features, coordinates/registration, and morphotype classification — together with the per-group cluster labels (`multi`/`texture`/`shape`/`gray`), producing a single `unite.csv`.

- **Input**: `features.csv`, `swc_all.csv`, `cls_correct.csv`, and `{group}/unite_5xcls.csv`.
- **Output**: `unite.csv`.

> The three input files (`features.csv`, `swc_all.csv`, `cls_correct.csv`) are analysis results generated during an actual run; adjust the file names in the script to match your own generated files.

### 3. Distribution Difference (`distribution difference.py`)

Tests distribution differences in two ways:

1. **Kolmogorov–Smirnov test** (`cal_differ`) — compares the distribution of subtype counts across brain regions. Outputs to `distribution/out_ks/{area}_ks.xlsx` with `small`/`medium` region sheets.
2. **Multiple tests** — per-feature t-tests (with Levene's test to choose equal/unequal variance) plus a multivariate mean test (Hotelling's T², or a permutation test when samples are small). Outputs to `distribution/stat_t.xlsx`.

- **Input**: `distribution/test_data/features_test.csv` and `distribution/test_data/count_regions/*.xlsx`.
- **Output**: `distribution/out_ks/*_ks.xlsx` and `distribution/stat_t.xlsx`.

## Citation

```
[Citation information to be added when published]
```

## License

MIT License.
