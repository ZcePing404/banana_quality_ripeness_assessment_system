# Project Structure

```text
banana_quality_project_organised/
├── app.py                              # Thin Streamlit entry point
├── streamlit_ui/                       # Streamlit-only interface code
│   ├── camera.py                       # USB camera handling
│   ├── controls.py                     # Sidebar parameter controls
│   ├── image_io.py                     # Uploaded-image decoding
│   ├── input_section.py                # Upload/camera input section
│   ├── roi.py                          # Mode B ROI drawing
│   └── results.py                      # Metrics and processing-stage display
│
├── banana_quality/                     # Classical image-processing logic
│   ├── config.py                       # Central validated/default parameter values
│   ├── image_utils.py                  # Resize/kernel helpers
│   ├── preprocessing.py                # Resize, denoise and colour conversion
│   ├── morphology.py                   # Morphological/component cleanup
│   ├── segmentation/
│   │   ├── kmeans.py                   # Mode A automatic banana segmentation
│   │   ├── grabcut.py                  # Mode B ROI + GrabCut segmentation
│   │   └── mask_refinement.py          # Shared banana-mask refinement
│   ├── blemish/
│   │   └── detection.py                # Lab a,b K-means + blemish rules
│   ├── surface_quality.py              # Blemish percentage and quality grade
│   ├── visualisation.py                # Overlay rendering
│   └── pipeline.py                     # High-level orchestration/public API
│
├── data/
│   └── annotations/                    # Classical evaluation ground truth
│       ├── images/                     # All 30 annotated images in one folder
│       ├── banana_masks/               # Ground-truth banana masks
│       ├── blemish_masks/              # Ground-truth blemish masks
│       ├── annotations.coco.json       # Combined source annotations
│       └── README.md                    # Dataset source and usage notes
│
├── evaluation/
│   ├── notebooks/
│   │   ├── 01_parameter_tuning.ipynb  # 21-image staged parameter tuning
│   │   └── 02_final_evaluation.ipynb  # 9-image untouched final evaluation
│   ├── evaluate_segmentation.py        # Optional fixed-set command-line evaluator
│   └── results/.gitkeep                # Generated tuning/evaluation CSVs go here
│
├── machine_learning/
│   └── ripeness_classification/        # Reserved for the future ML module
│       ├── data/raw/.gitkeep
│       ├── data/processed/.gitkeep
│       ├── models/.gitkeep
│       ├── notebooks/.gitkeep
│       ├── training/.gitkeep
│       └── evaluation/.gitkeep
│
├── README.md
├── requirements.txt
└── .gitignore
```

The classical processing behaviour remains separated by responsibility. `banana_quality/pipeline.py` is the public orchestration layer used by the Streamlit UI and the evaluation script.

The 30 existing annotated images remain together under `data/annotations/images/`. The classical tuning notebook creates a reproducible logical 21-image tuning / 9-image final-evaluation split without moving files. A separate train/validation/test split should still be created later **inside the machine-learning ripeness module** when the ripeness classifier is actually trained.
