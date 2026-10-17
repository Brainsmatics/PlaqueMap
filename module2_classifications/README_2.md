# Module 2 — Plaque Classification & Clustering Pipeline

Module 2 of a plaque image analysis pipeline: **feature selection**, **morphological classification**, and **cluster analysis** (2D and 3D) for plaque subtyping. It is part of a research project on plaque characterization using radiomics and machine learning.

## Analysis Pipeline

```text
test/test.xlsx
      │
      ▼
┌─────────────────────┐
│ feature selection.py │  Step 1 — screen morphotype-relevant features
└─────────────────────┘  (Wasserstein / JS / Kruskal-Wallis / PCA / KDE)
      │
      ▼
┌─────────────────────┐   ┌─────────────────────┐
│  classification.py   │   │ cluster analysis.py │  Step 3 — subtype discovery
└─────────────────────┘   │ cluster analysis_2D │  (K-Means on Gray/Shape/
  Step 2 — 3-class       └─────────────────────┘   Texture/Multi groups)
  prediction (Diffuse /
  Core / CAA)
```

- **`feature selection.py`** — entry point: evaluates features against the three morphotypes (Diffuse / Core / CAA) to guide feature selection.
- **`classification.py`** — supervised: classifies each plaque into a morphotype using the selected features, and outputs a confidence score.
- **`cluster analysis.py` / `cluster analysis_2D.py`** — unsupervised: uses all features (four groups) to discover finer subtypes, independently of the morphotype label. The 2D version maps 3D features to 2D first.

## Repository Structure

```
module2_classifications/
├── feature selection.py           # Step 1: feature screening
├── classification.py              # Step 2: morphological classification
├── cluster analysis.py            # Step 3: 3D clustering
├── cluster analysis_2D.py         # Step 3 (2D): 2D clustering
├── test/                          # feature-selection data & outputs
│   └── test.xlsx                  # input (requires a "cls" column: 1/2/3)
├── cls_test/                      # classification test data
│   └── test.xlsx
├── cls_model/                     # classification model
│   └── final2_model.model
├── cluster_test/                  # 3D clustering test data
│   └── test.xlsx
├── cluster_model/                 # 3D models + normalization params
│   ├── {group}_kmp-{k}.pkl        # gray=2, shape=5, texture=8, multi=21
│   ├── {group}_f_minmax.xlsx
│   └── 672areasvolume.xlsx        # region-number mapping table
├── cluster_test_2D/               # 2D clustering test data
│   └── test_2D.xlsx
└── cluster_model_2D/              # 2D clustering models
    └── {group}_kmp-{k}.pkl
```

## Dependencies

```bash
pip install pandas numpy scikit-learn xgboost scipy matplotlib seaborn joblib openpyxl
```

> `openpyxl` is needed for `.xlsx` read/write; `seaborn` only for `feature selection.py` (KDE).

## Scripts

| Script | What it does | Input | Output |
|--------|-------------|-------|--------|
| `feature selection.py` | Screen morphotype-relevant features | `test/test.xlsx` | `test/test_{wd,js,kw,pca}.xlsx`, `KDE_plot/*.png` |
| `classification.py` | 3-class morphological classification | `cls_test/test.xlsx` + `cls_model/final2_model.model` | `cls_test/cls_*.csv` |
| `cluster analysis.py` | 3D subtype clustering | `cluster_test/test.xlsx` + `cluster_model/*` | `cluster_test/{group}/cls_*.csv` |
| `cluster analysis_2D.py` | 2D subtype clustering | `cluster_test_2D/test_2D.xlsx` + `cluster_model_2D/*` | `cluster_test_2D/{group}/cls_*.csv` |

## Usage

Run from inside `module2_classifications/` so relative paths resolve.

```bash
python feature\ selection.py     # Step 1: screen features
python classification.py         # Step 2: classify plaques (Diffuse / Core / CAA)
python "cluster analysis.py"     # Step 3: 3D clustering
python "cluster analysis_2D.py"  # Step 3: 2D clustering
```

Each script is a runnable entry point. `classification.py` and both cluster scripts guard their main logic with `if __name__ == '__main__':`; `feature selection.py` runs at module top level. Training blocks inside the cluster scripts are commented out but fully self-contained (they take an `out_dir` argument and auto-create output directories via `os.makedirs`).

## Citation

```
[Citation information to be added when published]
```

## License

MIT License.
