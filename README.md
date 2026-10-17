# PlaqueMap

This repository contains the complete codebase for a plaque morphological analysis project. It is organized into three modules that together form an end-to-end pipeline, from raw images to downstream analysis.

```
raw plaque image
      │
      ▼
┌─────────────────────────────┐
│  Module 1 · Feature         │  segmentation → feature construction → feature extraction
│  Extraction                 │
└─────────────────────────────┘
      │  feature tables (CSV)
      ▼
┌─────────────────────────────┐
│  Module 2 · Classification  │  feature selection → morphotype classification → subtype clustering
│  & Clustering               │
└─────────────────────────────┘
      │  class / cluster labels
      ▼
┌─────────────────────────────┐
│  Module 3 · Analysis        │  similarity, post-processing, and distribution-difference tools
│                             │
└─────────────────────────────┘
```

## Modules

| Module | Directory | What it does | Readme |
|--------|-----------|--------------|--------|
| Module 1 | `module1_feature extraction/` | Extract quantitative features from raw plaque images (2D/3D) | [`README_1.md`](module1_feature%20extraction/README_1.md) |
| Module 2 | `module2_classifications/` | Classify plaques into morphotypes and discover subtypes | [`README_2.md`](module2_classifications/README_2.md) |
| Module 3 | `module3_analysis/` | Standalone downstream analysis tools | [`README_3.md`](module3_analysis/README_3.md) |

Each module is self-contained: it has its own scripts, data, and dependencies. To use a module, open its directory and follow the corresponding `README_*.md`.

---

### Module 1 — Feature Extraction (`module1_feature extraction/`)

Turns raw plaque images into quantitative feature tables through three stages — **segmentation**, **low-level feature construction**, and **radiomics feature extraction** — for both 2D and 3D data.

- **Scripts**: `segmentation_2D/3D.py`, `feature_construction_2D/3D.py`, `feature_extraction_2D/3D.py`.
- **Run**: the `feature_extraction_*` scripts are batch entry points; the `segmentation_*` and `feature_construction_*` scripts expose reusable functions.

```bash
pip install numpy opencv-python SimpleITK scikit-image scipy pandas tifffile imutils matplotlib
pip install pyradiomics   # only needed for feature extraction
```

See [`module1_feature extraction/README_1.md`](module1_feature%20extraction/README_1.md).

### Module 2 — Classification & Clustering (`module2_classifications/`)

Classifies each plaque into one of three morphotypes (**Diffuse / Core / CAA**) and further partitions plaques into subtypes via clustering. The workflow is sequential: **feature selection → classification → cluster analysis** (2D/3D).

- **Scripts**: `feature selection.py`, `classification.py`, `cluster analysis.py`, `cluster analysis_2D.py`.
- **Run**: all four are runnable entry points.

```bash
pip install pandas numpy scikit-learn xgboost scipy matplotlib seaborn joblib openpyxl
```

See [`module2_classifications/README_2.md`](module2_classifications/README_2.md).

### Module 3 — Analysis Tools (`module3_analysis/`)

Three **standalone** analysis tools with no data flow between them: **feature similarity**, **post processing**, and **distribution difference**.

- **Scripts**: `feature similarity.py`, `post processing.py`, `distribution difference.py`.
- **Run**: run whichever tool you need, independently.

```bash
pip install pandas numpy scipy scikit-learn matplotlib umap-learn openpyxl
```

See [`module3_analysis/README_3.md`](module3_analysis/README_3.md).

---

## Getting Started

1. Clone the repository.
2. Choose the module you need and install its dependencies (listed above and in the module's `README_*.md`).
3. Enter the module directory and follow the instructions in its `README_*.md`.

> Note: the pip package `umap-learn` (Module 3) is imported as `import umap`; `pyradiomics` (Module 1) is only required for feature extraction.

## License

MIT License.
