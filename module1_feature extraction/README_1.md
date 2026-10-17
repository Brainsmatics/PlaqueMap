# Module 1 — Plaque Feature Extraction Pipeline

This repository contains **Module 1** of a plaque image analysis pipeline: it turns raw plaque images into quantitative features through three stages — **segmentation**, **feature construction**, and **feature extraction**. Both **2D** and **3D** data are supported.

## Pipeline Overview

Each plaque image goes through the same three-stage workflow:

```
Raw image (2D / 3D)
      │
      ▼
[1] Segmentation ──────────────► foreground mask
      │
      ▼
[2] Feature Construction ──────► low-level geometric / intensity features
      │
      ▼
[3] Feature Extraction ────────► radiomics features  ──► CSV output
```

### 1. Segmentation

Binarize the raw plaque image to obtain the plaque **foreground mask**.
- **2D** (`segmentation_2D.py`): Otsu thresholding on a grayscale image.
- **3D** (`segmentation_3D.py`): maximum-entropy thresholding, followed by connected-component filtering and a Gaussian blur.

### 2. Feature Construction

Compute **low-level features** (geometry and intensity based) directly from the segmented foreground, without any external radiomics dependency.

### 3. Feature Extraction

Combine the original image with the segmentation mask to extract **high-level radiomics features** (via PyRadiomics), merge them with the low-level features, and write a per-image CSV.

Each stage has a corresponding **2D** and **3D** script (see below).

## Repository Structure

```
module1_feature_extraction/
├── segmentation_2D.py               # 2D segmentation
├── segmentation_3D.py               # 3D segmentation
├── feature_construction_2D.py       # 2D low-level features
├── feature_construction_3D.py       # 3D low-level features
├── feature_extraction_2D.py         # 2D radiomics features
├── feature_extraction_3D.py         # 3D radiomics features
├── param/
│   ├── param.yaml                   # PyRadiomics config (3D)
│   └── param_AD.yaml                # PyRadiomics config (2D)
├── test_2D/                         # 2D demo data and outputs
│   └── test_AD.jpg
├── test/                            # 3D demo data and outputs
│   └── test_tif.tif
├── extract_2D/                      # 2D extraction input and outputs
│   ├── test_AD.jpg
│   └── output/
└── extract/                         # 3D extraction input and outputs
    ├── *.tif
    ├── *_label.csv                  # plaque bounding-box labels
    └── output/
```

## Usage

The `segmentation_*` and `feature_construction_*` scripts expose reusable functions; the `feature_extraction_*` scripts are run directly as batch processes.

### Segmentation

```python
# 2D
import SimpleITK as sitk
from segmentation_2D import outline, jpg_to_8bit_grayscale_advanced

img = sitk.ReadImage("./test_2D/test_AD.jpg")
img_gray = jpg_to_8bit_grayscale_advanced(img, "./test_2D/test_AD_gray.tif")
arr = sitk.GetArrayFromImage(img_gray)
thresh, cnt, mask, volume = outline(arr, "test_AD", "./test_2D/")
```

```python
# 3D
import tifffile
from segmentation_3D import segmentation

arr = tifffile.imread("./test/test_tif.tif")
threshImg, compont_arr, im_outline, gaus_arr, volume = segmentation(arr)
```

### Feature Construction

```python
# 2D
from feature_construction_2D import analysis_single
out = analysis_single(cnt, thresh, arr, "./test_2D/", "test_AD")

# 3D
from feature_construction_3D import analysis_single
out = analysis_single(arr, "./test/")
```

### Feature Extraction

```bash
python feature_extraction_2D.py   # reads extract_2D/*.jpg  -> extract_2D/output/
python feature_extraction_3D.py   # reads extract/*.tif + extract/*_label.csv  -> extract/output/cut/
```

## Configuration

PyRadiomics parameters live in `param/`:
- `param/param.yaml` — 3D extraction (`feature_extraction_3D.py`)
- `param/param_AD.yaml` — 2D extraction (`feature_extraction_2D.py`)

## Dependencies

```bash
pip install numpy opencv-python SimpleITK scikit-image scipy pandas tifffile imutils matplotlib
pip install pyradiomics   # only needed for feature extraction
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
