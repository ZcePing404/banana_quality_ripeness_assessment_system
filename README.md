# Start here

See [START_HERE.md](START_HERE.md) for the updated Flask application, calibration, PDF and video instructions.

# Banana Surface Quality Analysis

A Streamlit-based classical image-processing prototype for banana segmentation and blemish quantification.

## Current processing modes

- **Mode A — Automatic K-means:** whole-image colour clustering followed by banana-mask refinement.
- **Mode B — ROI + GrabCut:** the user draws a rectangular region of interest and GrabCut segments the banana.
- Both modes use the same second-stage blemish detector inside the final banana mask.

## Run the application

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Ground-truth annotations

The 30 supplied annotated images have been consolidated into:

```text
data/annotations/images/
data/annotations/banana_masks/
data/annotations/blemish_masks/
```

The original COCO annotations are retained as `data/annotations/annotations.coco.json` for traceability.

## Tune and evaluate the classical segmentation

Use the Jupyter notebooks under `evaluation/notebooks/`:

1. `01_parameter_tuning.ipynb` creates a reproducible 21-image tuning / 9-image final-evaluation split, compares candidate parameter sets in stages, and reports the best tuning-set defaults. It does **not** edit `config.py` automatically.
2. Copy the selected defaults into `banana_quality/config.py`.
3. Restart the notebook kernel / Streamlit app.
4. Run `02_final_evaluation.ipynb` once on the untouched 9-image final set.

The Streamlit slider starting values are read from `ProcessingConfig`, so edited defaults in `config.py` become the starting values shown to the user. The UI sliders remain available for optional interactive refinement, but per-image manual adjustments should not be mixed into the reported automatic IoU/Dice evaluation.

The original `evaluation/evaluate_segmentation.py` is retained as an optional command-line evaluator.

## Future ripeness-classification module

The empty `machine_learning/ripeness_classification/` folders are reserved for the later ML component. The ML dataset should be split into training, validation and test sets at that stage rather than reusing the old Roboflow folder split automatically.


Run on mobile
Terminal: ipconfig