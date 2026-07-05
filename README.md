# ECG Cross-Dataset Generalization

Feature-based pipeline for measuring cross-dataset generalization of ECG arrhythmia classifiers between MIT-BIH and PTB-XL.

## Background

Electrocardiogram (ECG) arrhythmia classifiers are typically trained and evaluated on a single database, often reporting accuracy above 95%. This design cannot distinguish whether a model has learned transferable cardiac physiology or has overfit to one recording setup's equipment, patient population, and annotation conventions -- a problem known as domain shift. This project measures cross-dataset generalization in ECG arrhythmia classification, using the MIT-BIH Arrhythmia Database as a training source and PTB-XL as an independent cross-dataset test source.

## What this does

1. Extracts seven hand-engineered beat-level features (R-R intervals, R-R ratios, QRS amplitude, QRS-width proxy) from MIT-BIH using expert beat annotations
2. Extracts the same features from PTB-XL using WFDB's `xqrs` beat detector, with each beat inheriting its 10-second strip's diagnostic label
3. Harmonizes both databases' labels into a shared three-class AAMI grouping (Normal, Supraventricular ectopic, Ventricular ectopic) via an explicit crosswalk from PTB-XL SCP-ECG codes
4. Trains a random forest on MIT-BIH's inter-patient DS1 split, evaluating in-dataset (DS2) and cross-dataset (PTB-XL)
5. Runs a controlled comparison downsampling MIT-BIH to PTB-XL's 100 Hz rate, to isolate whether cross-dataset degradation is driven by acquisition differences or genuine population/label shift

## Results summary

| | In-dataset (MIT-BIH DS2) | Cross-dataset (PTB-XL) |
|---|---|---|
| Macro-F1 | 0.61 | 0.35 |
| N (normal) F1 | 0.96 | 0.50 |
| S (supraventricular) F1 | 0.09 | 0.34 |
| V (ventricular) F1 | 0.77 | 0.23 |

The ventricular class, strong in-dataset, is the least robust cross-dataset -- 60% of true PTB-XL V beats are misclassified as normal. A sampling-rate-matched control experiment shows this is not explained by the 360 Hz vs 100 Hz acquisition difference between datasets.

## Installation

Requires Python 3.9+:

pip install -r requirements.txt

wfdb
numpy
pandas
scipy
scikit-learn
joblib

## Usage

Run scripts in order:
1. `extract_mitbih.py` -- MIT-BIH feature extraction
2. `train_model.py` -- train random forest on DS1
3. `evaluate_indataset.py` -- evaluate on DS2 (in-dataset)
4. `extract_ptbxl.py` -- PTB-XL feature extraction
5. `evaluate_crossdataset.py` -- evaluate trained model on PTB-XL
6. `control_100hz.py` -- sampling-rate-matched control experiment

Data (MIT-BIH and PTB-XL) is downloaded automatically from PhysioNet by the extraction scripts.

## License

MIT License (see LICENSE)


