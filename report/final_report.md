# Computer Vision Classification: Experimental Report

## 1. Introduction

This project implements and compares two computer vision approaches for image classification: classical machine learning using a Support Vector Machine (SVM) and deep learning using MobileNetV2.

The objective is to investigate image preprocessing, feature extraction, model training, and evaluation using accuracy, precision, recall, and F1-score.

**Dataset clarification:** Although the assignment concerns avocado ripening, the supplied dataset contains orange images. The four available categories are `blackspot`, `canker`, `fresh`, and `grenning`. Consequently, the experiments reported here classify orange-condition categories rather than avocado-ripening stages.

## 2. Dataset Overview

The dataset contains 1,090 images distributed across four classes.

| Class | Training images | Test images | Total |
|---|---:|---:|---:|
| blackspot | 184 | 22 | 206 |
| canker | 179 | 22 | 201 |
| fresh | 281 | 33 | 314 |
| grenning | 347 | 22 | 369 |
| **Total** | **991** | **99** | **1,090** |

The original training set was divided into training and validation subsets using stratified sampling.

| Split | Images |
|---|---:|
| Training | 842 |
| Validation | 149 |
| Test | 99 |
| **Total** | **1,090** |

The test set was kept separate from the training and validation sets.

## 3. Exploratory Data Analysis

The initial data inspection covered image readability, image dimensions, class distribution, and exact duplicate detection.

Key findings:

- All 1,090 images were readable during the inspection.
- No exact duplicate images were detected using file-content hashes.
- Image dimensions varied, with a minimum of 144 × 138 pixels and a maximum of 800 × 800 pixels.
- The largest class contained 369 images, while the smallest contained 201 images.
- The largest-to-smallest class ratio was approximately 1.84:1, indicating moderate class imbalance.

These findings motivated consistent image resizing, stratified data splitting, and class-weighted learning where appropriate.

The absence of exact duplicate files does not guarantee that all images are independent or free from near-duplicates.

## 4. Approach One: Classical Machine Learning

### 4.1 Feature Extraction

Each image was resized to 128 × 128 pixels before feature extraction.

The feature vector combined three types of information:

- HSV color histograms: 56 features.
- LAB color histograms: 48 features.
- Local Binary Patterns (LBP): 19 texture features.

The resulting feature vector contained 123 features per image.

### 4.2 Classification

An SVM classifier with an RBF kernel was trained using a pipeline containing feature standardization and classification.

Class weights were balanced to reduce the effect of differences in class frequencies.

### 4.3 Validation Results

The SVM achieved:

- Validation accuracy: 98.66%.
- Validation macro F1-score: 98.31%.

These metrics were used during model assessment before evaluating the model on the held-out test set.

## 5. Approach Two: MobileNetV2

### 5.1 Model Architecture

MobileNetV2 was initialized using pretrained ImageNet weights.

The original feature-extraction layers were frozen, and the final classification layer was replaced with a four-class output layer.

Images were resized to 224 × 224 pixels and normalized using ImageNet mean and standard deviation values.

### 5.2 Training Configuration

The training process used:

- Random horizontal flipping.
- Random rotation.
- Moderate color jitter.
- Class-weighted cross-entropy loss.
- Adam optimizer with a learning rate of 0.001.
- Batch size of 32.
- Eight training epochs.

The best validation macro F1-score was 1.0000 at epoch 6.

This perfect validation score should not be interpreted as evidence of perfect generalization. Performance on the independent test set provides a more useful assessment.

## 6. Test Evaluation

Both approaches were evaluated on the same 99-image test set.

| Metric | SVM | MobileNetV2 |
|---|---:|---:|
| Accuracy | 96.97% | 94.95% |
| Macro F1-score | 96.59% | 94.45% |
| Correct predictions | 96/99 | 94/99 |
| Incorrect predictions | 3/99 | 5/99 |

### 6.1 SVM Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| blackspot | 0.91 | 0.95 | 0.93 | 22 |
| canker | 0.95 | 0.91 | 0.93 | 22 |
| fresh | 1.00 | 1.00 | 1.00 | 33 |
| grenning | 1.00 | 1.00 | 1.00 | 22 |
| **Macro average** | **0.97** | **0.97** | **0.97** | **99** |

### 6.2 MobileNetV2 Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| blackspot | 0.91 | 0.91 | 0.91 | 22 |
| canker | 0.90 | 0.86 | 0.88 | 22 |
| fresh | 0.97 | 1.00 | 0.99 | 33 |
| grenning | 1.00 | 1.00 | 1.00 | 22 |
| **Macro average** | **0.95** | **0.94** | **0.94** | **99** |

The displayed class-level values are rounded; aggregate metrics may differ slightly from calculations using rounded values.

## 7. Model Comparison and Interpretation

The SVM achieved higher test accuracy and macro F1-score than MobileNetV2 in this experiment.

Both models correctly classified every `fresh` and `grenning` test image. Their remaining errors occurred in the `blackspot` and `canker` categories.

The SVM made three incorrect predictions, compared with five for MobileNetV2.

One possible explanation is that color and texture features provide useful information for distinguishing these particular image categories. However, the results do not establish why the SVM performed better, and the small test set limits the strength of this conclusion.

The comparison also does not establish that SVM is generally superior to deep learning.

## 8. Outliers, Imbalance, and Potential Bias

The exploratory analysis identified moderate class imbalance and variation in image dimensions.

Potential limitations include:

- Unequal numbers of examples per class.
- Differences in lighting, backgrounds, camera quality, and image framing.
- Possible correlations between image collection conditions and class labels.
- Limited evidence about performance on images from new sources.
- The possibility of near-duplicate or related images that exact hash matching would not detect.

Stratified splitting and class-weighted learning help address some issues, but they cannot eliminate dataset bias.

## 9. Production Performance

The current experiments demonstrate an offline image-classification workflow rather than a production-ready service.

The classical approach uses handcrafted features followed by an SVM. MobileNetV2 uses a pretrained neural network with a modified classification layer.

The training time recorded for MobileNetV2 was approximately 83.2 seconds over eight epochs in the Google Colab environment used for the experiment.

This is a training-time measurement, not an inference-latency benchmark. Production deployment would require additional measurements of per-image latency, throughput, memory usage, and performance on target hardware.

Neither model has been validated for real-world avocado-ripening decisions.

## 10. Drawbacks

The main drawbacks are:

1. The supplied dataset contains orange-condition labels rather than avocado-ripening labels.
2. The independent test set contains only 99 images.
3. The evaluation does not establish generalization across different farms, cameras, lighting conditions, or backgrounds.
4. The validation performance, especially the perfect MobileNetV2 validation macro F1-score at epoch 6, may not reflect performance on unseen data from other sources.
5. Confidence scores from the neural network should not be treated as calibrated probabilities without additional evaluation.
6. The current experiments do not include a comprehensive inference-speed or resource-usage comparison.

## 11. Future Improvements

The next steps should be:

- Obtain a correctly labeled avocado-ripening dataset.
- Repeat the experiments with labels that represent avocado-ripening stages.
- Increase the dataset size and collect images under varied conditions.
- Evaluate on an independently collected test set.
- Investigate near-duplicates and potential data leakage.
- Tune model hyperparameters using an appropriate validation procedure.
- Evaluate model calibration and inference latency.
- Build a prediction interface only after validating the model on the intended avocado dataset.

## 12. Where Time Was Spent

The work focused on:

- Inspecting and organizing the supplied image dataset.
- Performing exploratory data analysis and checking image quality.
- Implementing color and texture feature extraction.
- Training and validating the SVM classifier.
- Adapting and training MobileNetV2 using transfer learning.
- Evaluating both models using classification metrics and confusion matrices.
- Comparing test performance and documenting limitations.

## 13. Conclusion

Two approaches were implemented and evaluated on the supplied four-class image dataset.

The SVM achieved 96.97% test accuracy and a 96.59% macro F1-score. MobileNetV2 achieved 94.95% test accuracy and a 94.45% macro F1-score.

The SVM performed better on this particular test set, although broader testing is required to determine whether this difference persists on other data.

Most importantly, the supplied dataset contains orange-condition categories rather than avocado-ripening stages. Therefore, these results should be considered an experimental comparison on the provided dataset, not a validated avocado-ripening classification solution.