# Dry Bean Variety Classification — ML Assignment 2

## a. Problem Statement
The goal is to automatically identify the variety of a dry bean from measurements of its shape.
Given 16 geometric features extracted from images of individual beans, the task is a
**multi-class classification problem**: predict which of 7 registered bean varieties
(Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira) a bean belongs to.
Five classification models are trained on the same data, evaluated with six metrics, and
compared, and the trained models are served through an interactive Streamlit web app.

## b. Dataset Description
| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository — Dry Bean Dataset (id 602) |
| Instances | 13,611 |
| Features | 16, all numeric |
| Target | `Class` — 7 bean varieties |
| Missing values | None |

**Features**
- *12 dimensional features:* Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation,
  Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness
- *4 shape-form features:* ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4

**Class distribution** (full dataset → test set)

| Class | Total | Test set |
|---|---|---|
| Dermason | 3,546 | 709 |
| Sira | 2,636 | 527 |
| Seker | 2,027 | 406 |
| Horoz | 1,928 | 386 |
| Cali | 1,630 | 326 |
| Barbunya | 1,322 | 265 |
| Bombay | 522 | 104 |

The classes are moderately imbalanced (Dermason is about 7× larger than Bombay).

**Preprocessing**
- 80% train / 20% test split, stratified by class (`random_state=27`) → 10,888 train / 2,723 test rows
- Features standardised with `StandardScaler` (fitted on training data only)
- Labels encoded with `LabelEncoder`
- The 20% test split is saved as `test_data.csv` for use in the Streamlit app

## c. GitHub Repository Link
TODO: https://github.com/<your-username>/dry-bean-classifier

**Live Streamlit App:** TODO: https://<your-app>.streamlit.app

## d. Models Used
1. Logistic Regression (`max_iter=2000`)
2. Decision Tree Classifier (`max_depth=10`)
3. K-Nearest Neighbors Classifier (`n_neighbors=7`)
4. Naive Bayes Classifier — Gaussian
5. Support Vector Machine (RBF kernel)

### Comparison Table (test set, 2,723 samples)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9188 | 0.9941 | 0.9190 | 0.9188 | 0.9189 | 0.9019 |
| Decision Tree | 0.9053 | 0.9640 | 0.9059 | 0.9053 | 0.9054 | 0.8854 |
| kNN | 0.9148 | 0.9876 | 0.9151 | 0.9148 | 0.9149 | 0.8969 |
| Naive Bayes | 0.8909 | 0.9901 | 0.8923 | 0.8909 | 0.8910 | 0.8686 |
| Support Vector Machine | **0.9269** | **0.9947** | **0.9270** | **0.9269** | **0.9269** | **0.9116** |

*Precision, Recall and F1 are weighted averages across the 7 classes; AUC is one-vs-rest (OvR).*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second-best overall (Accuracy 91.88%, MCC 0.9019), with an AUC of 0.9941 almost equal to SVM. After scaling, most bean classes are close to linearly separable in this feature space, so a simple linear model performs very well while being the fastest to train. |
| Decision Tree | Lowest AUC of all models (0.9640) and second-lowest accuracy (90.53%). A single tree gives coarse, step-like probabilities from its leaves, which hurts ranking quality (AUC), and its axis-aligned splits approximate the curved class boundaries less smoothly than the other models. |
| kNN | Strong performance (Accuracy 91.48%, MCC 0.8969), close to Logistic Regression. Works well because the features are standardised and beans of the same variety form compact clusters; errors occur mainly where similar varieties overlap. |
| Naive Bayes | Lowest Accuracy (89.09%) and MCC (0.8686), yet a high AUC (0.9901). Its feature-independence assumption is clearly violated — Area, Perimeter, ConvexArea and EquivDiameter are all strongly correlated size measures — so its hard predictions suffer even though it still ranks classes reasonably well. |
| Support Vector Machine | Best on all six metrics (Accuracy 92.69%, AUC 0.9947, MCC 0.9116). The RBF kernel captures non-linear boundaries between similar varieties. Its confusion matrix shows Bombay is classified perfectly (104/104); most remaining errors are between Sira and Dermason (45 Sira → Dermason, 36 Dermason → Sira) and between Barbunya and Cali, varieties with very similar shapes. |
| **Overall winner** | **Support Vector Machine** — highest Accuracy, F1 and MCC. Logistic Regression is a close, cheaper alternative (≈0.8% lower accuracy). The ranking is consistent across F1 and MCC, so the result is not driven by class imbalance. |

## Streamlit App Features
- Upload test data as CSV
- Model selection dropdown (all 5 models)
- Display of all 6 evaluation metrics
- Confusion matrix heatmap and classification report
- Side-by-side comparison of all models

## Repository Structure
```
dry-bean-classifier/
├── app.py                 # Streamlit app
├── requirements.txt       # dependencies
├── README.md
├── test_data.csv          # 20% held-out test split
└── model/
    ├── train_models.ipynb # data prep, training, evaluation
    └── *.pkl              # 5 saved models + scaler + label encoder
```

## How to Run Locally
```
pip install -r requirements.txt
streamlit run app.py
```
