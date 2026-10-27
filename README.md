# Avocado Ripening Classification

## Project Overview

This project explores image classification using two computer vision approaches:

1. Classical machine learning using HSV and LAB color histograms, Local Binary Patterns (LBP), and a Support Vector Machine (SVM).
2. Deep learning using a pretrained MobileNetV2 model with transfer learning.

The experiments were implemented and evaluated in Google Colab.

## Important Dataset Note

The supplied dataset contains orange images rather than avocado images. Its four classes are:

- blackspot
- canker
- fresh
- grenning

Therefore, the reported experiments classify the supplied orange-condition categories. They should not be interpreted as validated avocado-ripening predictions.

## Models

### Approach 1: Classical Machine Learning

The classical pipeline extracts color and texture features from each image and uses an SVM classifier to predict its class.

Features include:
- HSV color histograms
- LAB color histograms
- Local Binary Patterns (LBP)

### Approach 2: Deep Learning

MobileNetV2 was initialized with pretrained ImageNet weights. Its feature-extraction layers were frozen, and the final classification layer was adapted to the four dataset classes.

Training used data augmentation and class-weighted cross-entropy loss.

## Test Results

| Metric | SVM | MobileNetV2 |
|---|---:|---:|
| Accuracy | 96.97% | 94.95% |
| Macro F1-score | 96.59% | 94.45% |
| Correct predictions | 96/99 | 94/99 |

These results apply only to the supplied orange dataset and the evaluated test split.

The SVM achieved higher test accuracy and macro F1-score in this experiment. This result alone does not establish that SVM will perform better on other datasets.

## Project Structure

```text
avocado-ripening-classification/
├── models/
│   ├── classical_svm.joblib
│   └── mobilenet_v2_best.pth
├── notebooks/
│   └── Avocado_Ripening_Classification.ipynb
├── report/
│   └── final_report.md
├── results/
│   ├── figures/
│   └── metrics/
├── src/
│   ├── evaluate.py
│   ├── feature_extraction.py
│   ├── predict.py
│   ├── train_mobilenet.py
│   └── train_svm.py
├── requirements.txt
└── README.md