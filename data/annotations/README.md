# Annotated Ground-Truth Data

This folder contains the **30 annotated banana images** supplied in the earlier project dataset. The original Roboflow export separated the files into `train`, `valid`, and `test`; those folders have been merged here because this dataset is currently used as **ground truth for classical image-processing evaluation**, not for model training.

## Folder contents

```text
annotations/
├── images/                 # 30 original input images
├── banana_masks/           # Binary ground-truth banana-surface masks
├── blemish_masks/          # Binary ground-truth blemish masks
├── annotations.coco.json   # Combined original COCO segmentation annotations
└── README.md
```

Each mask uses the same filename stem as its image. For example:

```text
images/example.jpg
banana_masks/example.png
blemish_masks/example.png
```

The PNG masks are derived directly from the supplied COCO segmentation annotations. White pixels (`255`) represent the annotated region and black pixels (`0`) represent background.

## Why there is no train/valid/test folder here

A train/validation/test split is required when fitting a machine-learning model. The current classical banana/blemish image-processing pipeline does not learn weights from these images, so storing the ground-truth evaluation images together is clearer and avoids treating the original Roboflow export split as an ML experiment split.

When the ripeness-classification model is developed later, its dataset should be placed under `machine_learning/ripeness_classification/data/` and split there into training, validation, and test sets as part of the ML workflow.

## Dataset source

- Dataset: **Banana Blemish Ground Truth**
- Roboflow Universe: `https://universe.roboflow.com/tams-workspace-iuock/banana-blemish-ground-truth`
- Export date recorded in the supplied files: 5 August 2026
- Format: COCO Segmentation
- Images: 30
- Pre-processing/augmentation recorded in the supplied export: none
- Licence recorded in the supplied dataset: **CC BY 4.0**
