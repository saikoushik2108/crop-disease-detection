# Dataset Setup Guide

This project uses the **Paddy Doctor** dataset for crop disease classification. To maintain repository size and privacy, the dataset is not included in this repository.

## How to Obtain the Dataset
1. Visit the [Paddy Disease Classification Kaggle Competition](https://www.kaggle.com/competitions/paddy-disease-classification).
2. Download the competition data.
3. Extract the files into the project directory.

## Expected Folder Structure
The system expects the data to be placed as follows:

```
data/
└── raw/
    ├── train.csv           # The main metadata file
    └── train_images/       # Folder containing all leaf images
        ├── image_1.jpg
        ├── image_2.jpg
        └── ...
```

## Validation
After placing the data, you can verify its integrity by running the audit script:
```bash
python src/utils/audit_dataset.py
```
This will check for missing images, label consistency, and corrupted files.
