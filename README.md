# Ethnicity-DA
# Supervised Domain Adaptation Mitigates Cross-Ethnicity Prediction Error in Neuroimaging-Based Cognitive Prediction

Code repository for the paper:

**"Supervised Domain Adaptation Mitigates Cross-Ethnicity Prediction Error in Neuroimaging-Based Cognitive Prediction"**

Preprint: DOI_TO_BE_ADDED

---

## Overview

This repository contains the code used for neuroimaging-based cognitive prediction and supervised domain adaptation experiments across ethnically diverse populations.

The project evaluates whether supervised domain adaptation methods can reduce cross-ethnicity prediction error in MRI-based cognitive prediction models by incorporating small numbers of target-domain participants during training.

The repository includes:

- Baseline predictive modeling
- Multiple supervised domain adaptation methods
- Incremental target-sample adaptation experiments
- Analysis and plotting notebooks

---

## Repository Structure

### Core Scripts

| File | Description |
|---|---|
| `1_Typ-Nesi.py` | Baseline predictive modeling without domain adaptation |
| `2_AdaptBW-Nesi.py` | Balanced Weighting domain adaptation |
| `3_AdaptTRB-Nesi.py` | TwoStageTrAdaBoostR2 domain adaptation |
| `4_AdaptLinInt-Nesi.py` | Linear Interpolation (LinInt) domain adaptation |
| `5_AdapPred-Nesi.py` | PRED feature augmentation domain adaptation |
| `6_Analysis&Plotting.ipynb` | Statistical analysis, evaluation, and figure generation |

---

## Methods

The repository includes implementations of several supervised domain adaptation approaches using the ADAPT library:

- TwoStageTrAdaBoostR2
- Balanced Weighting
- LinInt
- PRED Feature Augmentation

All methods were implemented using Partial Least Squares (PLS) regression as the base estimator.

---

## Data

This repository does not include the neuroimaging or behavioral datasets used in the study.

The analyses were conducted using preprocessed MRI-derived features and cognitive measures from the ABCD Study dataset.

Users must obtain access to the relevant datasets independently.

---

## Requirements

Main Python dependencies include:

- Python 3.x
- numpy
- pandas
- scikit-learn
- scipy
- matplotlib
- seaborn
- joblib
- ADAPT

Example installation:

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn joblib adapt
